import argparse
import os
from pathlib import Path
from transformers import set_seed, Trainer, Seq2SeqTrainer

from src.utils.config_loader import ConfigLoader
from src.utils.metrics import print_classification_report
from src.data_loader import OUDDataLoader
from src.models.deberta_classifier import DeBERTaClassifier
from src.models.t5_classifier import T5Classifier

import numpy as np

def main(args):
    # Load configs
    config_loader = ConfigLoader()
    configs = config_loader.load_all_configs(args.model, args.data_type)
    
    # Set seed
    set_seed(configs['training']['seed'])
    
    # Setup paths
    output_dir = Path(configs['paths']['output']['checkpoints']) / f"{args.model}_{args.data_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data with {args.data_type} explanations...")
    data_loader = OUDDataLoader(
        data_paths=configs['data_paths'],
        labels=configs['training']['labels'],
        val_split=configs['training']['val_split'],
        seed=configs['training']['seed']
    )
    
    train_ds, valid_ds, worker_test_ds, expert_test_ds = data_loader.load_and_prepare_data()
    
    # Initialize model
    print(f"Initializing {configs['model']['model_name']}...")
    model_type = configs['model']['model_type']
    
    if model_type == "encoder":
        model_wrapper = DeBERTaClassifier(
            configs['model'],
            data_loader.label2id,
            data_loader.id2label
        )
        trainer_class = Trainer
    elif model_type == "encoder-decoder":
        cache_dir = configs['paths'].get('cache', {}).get('hf_cache')
        model_wrapper = T5Classifier(
            configs['model'],
            data_loader.label2id,
            data_loader.id2label,
            cache_dir=cache_dir
        )
        trainer_class = Seq2SeqTrainer
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    remove_cols = ["label", "text", "label_str"] if "label_str" in train_ds.column_names else ["label", "text"]
    if model_type == "encoder":
        remove_cols = [c for c in remove_cols if c != "label_str"] + ["label_id"]
        train_ds = train_ds.rename_column("label_id", "label")
        valid_ds = valid_ds.rename_column("label_id", "label")
        worker_test_ds = worker_test_ds.rename_column("label_id", "label")
        expert_test_ds = expert_test_ds.rename_column("label_id", "label")
        remove_cols.remove("label")
    
    train_ds = train_ds.map(model_wrapper.tokenize_function, batched=True, remove_columns=remove_cols)
    valid_ds = valid_ds.map(model_wrapper.tokenize_function, batched=True, remove_columns=remove_cols)
    worker_test_ds = worker_test_ds.map(model_wrapper.tokenize_function, batched=True, remove_columns=remove_cols)
    expert_test_ds = expert_test_ds.map(model_wrapper.tokenize_function, batched=True, remove_columns=remove_cols)
    
    # Get training arguments
    training_args = model_wrapper.get_training_args(str(output_dir), configs['training'])
    
    # Initialize trainer
    trainer = trainer_class(
        model=model_wrapper.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=model_wrapper.tokenizer,
        data_collator=model_wrapper.data_collator,
        compute_metrics=model_wrapper.compute_metrics
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\n→ Validation metrics:")
    print(trainer.evaluate())
    
    print("\n→ Worker-held-out test metrics:")
    worker_results = trainer.predict(worker_test_ds)
    print(worker_results.metrics)
    
    print("\n→ Expert-held-out test metrics:")
    expert_results = trainer.predict(expert_test_ds)
    print(expert_results.metrics)
    
    # Detailed classification reports
    def get_predictions_and_labels(prediction_output, model_type):
        if model_type == "encoder":
            preds = np.argmax(prediction_output.predictions, axis=1)
            labels = prediction_output.label_ids
        else:  # encoder-decoder
            # Decode predictions
            pred_tokens = np.where(
                prediction_output.predictions != -100,
                prediction_output.predictions,
                model_wrapper.tokenizer.pad_token_id
            )
            label_tokens = np.where(
                prediction_output.label_ids != -100,
                prediction_output.label_ids,
                model_wrapper.tokenizer.pad_token_id
            )
            
            decoded_preds = model_wrapper.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
            decoded_labels = model_wrapper.tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
            
            preds = np.array([data_loader.label2id.get(p.strip(), -1) for p in decoded_preds])
            labels = np.array([data_loader.label2id[l.strip()] for l in decoded_labels])
            
            # Filter out invalid predictions
            valid_mask = preds != -1
            preds = preds[valid_mask]
            labels = labels[valid_mask]
        
        return labels, preds
    
    worker_labels, worker_preds = get_predictions_and_labels(worker_results, model_type)
    expert_labels, expert_preds = get_predictions_and_labels(expert_results, model_type)
    
    print_classification_report("Worker-held-out Test", worker_labels, worker_preds, configs['training']['labels'])
    print_classification_report("Expert-held-out Test", expert_labels, expert_preds, configs['training']['labels'])
    
    print(f"\n Training complete! Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline OUD classification models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["deberta_base", "deberta_large", "t5_3b", "t5_11b"],
        help="Model to train"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="wo",
        choices=["w", "wo"],
        help="Data type: 'w' (with explanation) or 'wo' (without explanation)"
    )
    
    args = parser.parse_args()
    main(args)
