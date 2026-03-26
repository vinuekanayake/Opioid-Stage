import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, T5Tokenizer, set_seed, get_linear_schedule_with_warmup

from src.utils.config_loader import ConfigLoader
from src.contrastive.augmentations import TextAugmenter
from src.contrastive.scl_dataset import SCLDataset
from src.contrastive.scl_sampler import BalancedBatchSampler
from src.contrastive.scl_loss import supervised_contrastive_loss
from src.models.scl_models import DeBERTaSCLModel, T5SCLModel


def collate_fn(batch_items, tokenizer, max_len):
    """Collate function for SCL training."""
    view1_texts = [item["text"] for item in batch_items]
    view2_texts = [item["text_aug"] for item in batch_items]
    labels = torch.tensor([item["label"] for item in batch_items], dtype=torch.long)
    
    tok1 = tokenizer(
        view1_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    tok2 = tokenizer(
        view2_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    return {
        "input_ids_1": tok1["input_ids"],
        "attention_mask_1": tok1["attention_mask"],
        "input_ids_2": tok2["input_ids"],
        "attention_mask_2": tok2["attention_mask"],
        "labels": labels,
    }


def train_scl(args):
    # Load configs
    config_loader = ConfigLoader()
    configs = config_loader.load_all_configs(
        f"{args.model}_scl",
        args.data_type
    )
    
    # Load SCL training config
    scl_config = config_loader.load_yaml("training_configs/scl_pretrain.yaml")
    configs['training'].update(scl_config)
    
    # Set seed
    seed = configs['training']['seed']
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup paths
    output_dir = Path(configs['paths']['output']['checkpoints']) / \
        f"scl_{args.model}_{args.data_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    label2id = {label: i for i, label in enumerate(configs['training']['labels'])}
    
    augmenter = TextAugmenter(configs['training'])
    dataset = SCLDataset(
        csv_path=configs['data_paths']['train'],
        label2id=label2id,
        augmenter=augmenter
    )
    
    # Create balanced sampler
    pretrain_config = configs['model']['pretrain']
    sampler = BalancedBatchSampler(
        class_to_indices=dataset.class_to_indices,
        n_classes=pretrain_config['n_classes_per_batch'],
        n_samples=pretrain_config['batch_size_per_class']
    )
    
    # Load tokenizer and model
    print(f"Initializing {configs['model']['model_name']}...")
    cache_dir = configs['paths'].get('cache', {}).get('hf_cache')
    
    if configs['model']['model_type'] == 'encoder':
        tokenizer = AutoTokenizer.from_pretrained(configs['model']['model_name'])
        model = DeBERTaSCLModel(
            configs['model']['model_name'],
            projection_dim=configs['model']['projection_dim']
        )
    else:  # encoder-decoder (T5)
        tokenizer = T5Tokenizer.from_pretrained(configs['model']['model_name'])
        model = T5SCLModel(
            configs['model']['model_name'],
            projection_dim=configs['model']['projection_dim'],
            cache_dir=cache_dir
        )
    
    model.to(device)
    
    # Collate function with tokenizer
    def collate_batch(batch_indices):
        items = []
        for idx in batch_indices:
            text, label = dataset.get_raw(idx)
            view1 = text
            view2 = augmenter.augment_post(text)
            items.append({"text": view1, "text_aug": view2, "label": label})
        return collate_fn(items, tokenizer, configs['model']['max_length'])
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pretrain_config['learning_rate'],
        weight_decay=pretrain_config['weight_decay']
    )
    
    total_steps = (len(sampler) * pretrain_config['epochs']) // \
        pretrain_config['gradient_accumulation_steps']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )
    
    # Mixed precision training
    use_fp16 = configs['model'].get('use_fp16', False)
    scaler = GradScaler(enabled=use_fp16)
    
    # Training loop
    print("Starting pretraining...")
    model.train()
    global_step = 0
    
    for epoch in range(pretrain_config['epochs']):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_indices in sampler:
            batch = collate_batch(batch_indices)
            
            # Move to device
            input_ids_1 = batch["input_ids_1"].to(device, non_blocking=True)
            attn_1 = batch["attention_mask_1"].to(device, non_blocking=True)
            input_ids_2 = batch["input_ids_2"].to(device, non_blocking=True)
            attn_2 = batch["attention_mask_2"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=not use_fp16):
                z1 = model.get_projection(input_ids_1, attn_1)
                z2 = model.get_projection(input_ids_2, attn_2)
                
                features = torch.cat([z1, z2], dim=0)
                labels_long = torch.cat([labels, labels], dim=0)
                
                loss = supervised_contrastive_loss(
                    features,
                    labels_long,
                    temperature=configs['model']['temperature']
                )
                loss = loss / pretrain_config['gradient_accumulation_steps']
            
            # Backward pass
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (global_step + 1) % pretrain_config['gradient_accumulation_steps'] == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * pretrain_config['gradient_accumulation_steps']
            batch_count += 1
            
            if global_step % configs['training']['logging_steps'] == 0:
                print(f"Epoch {epoch+1}/{pretrain_config['epochs']} "
                      f"step {global_step} "
                      f"loss {epoch_loss / batch_count:.4f}")
        
        avg_loss = epoch_loss / (batch_count + 1e-12)
        print(f"=== Epoch {epoch+1} average loss: {avg_loss:.4f} ===")
        
        # Save checkpoint
        if (epoch + 1) % configs['training']['epoch_save_interval'] == 0:
            encoder_path = output_dir / f"encoder_epoch{epoch+1}.pth"
            proj_path = output_dir / f"projection_epoch{epoch+1}.pth"
            torch.save(model.encoder.state_dict(), encoder_path)
            torch.save(model.projection.state_dict(), proj_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Final save
    torch.save(model.encoder.state_dict(), output_dir / "encoder.pth")
    torch.save(model.projection.state_dict(), output_dir / "projection.pth")
    print(f"Training complete! Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCL Pretraining")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["deberta_base", "deberta_large", "t5_3b", "t5_11b"],
        help="Model to pretrain"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="wo",
        choices=["w", "wo"],
        help="Data type: 'w' (with explanation) or 'wo' (without explanation)"
    )
    
    args = parser.parse_args()
    train_scl(args)
