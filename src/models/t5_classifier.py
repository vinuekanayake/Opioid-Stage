import numpy as np
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

class T5Classifier:
    def __init__(self, config, label2id, id2label, cache_dir=None):
        self.config = config
        self.label2id = label2id
        self.id2label = id2label
        self.prefix = config.get('prefix', 'classify:')
        
        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(
            config['model_name'],
            cache_dir=cache_dir
        )
        
        model_kwargs = {'cache_dir': cache_dir}
        
        # Add T5-11B specific configurations
        if '11b' in config['model_name'].lower():
            model_kwargs.update({
                'torch_dtype': torch.bfloat16,
                'device_map': 'auto'
            })
        
        self.model = T5ForConditionalGeneration.from_pretrained(
            config['model_name'],
            **model_kwargs
        )
        
        # Enable gradient checkpointing for large models
        if config.get('use_gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model
        )
    
    def tokenize_function(self, batch):
        """Tokenize input text with prefix."""
        inputs = [f"{self.prefix} {txt}" for txt in batch["text"]]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config['max_length'],
            truncation=True
        )
        
        # Tokenize labels
        labels = self.tokenizer(
            batch["label_str"],
            max_length=10,
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def compute_metrics(self, eval_preds):
        """Compute metrics for evaluation."""
        from src.utils.metrics import compute_classification_metrics
        
        gen_tokens, label_tokens = eval_preds
        
        # Replace -100 with pad token
        label_tokens = np.where(
            label_tokens != -100,
            label_tokens,
            self.tokenizer.pad_token_id
        )
        gen_tokens = np.where(
            gen_tokens != -100,
            gen_tokens,
            self.tokenizer.pad_token_id
        )
        
        # Decode
        decoded_preds = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
        
        # Map to IDs
        pred_ids = [self.label2id.get(pred, -1) for pred in decoded_preds]
        label_ids = [self.label2id[label] for label in decoded_labels]
        
        return compute_classification_metrics(label_ids, pred_ids)
    
    def get_training_args(self, output_dir, training_config):
        """Get Seq2SeqTrainingArguments."""
        args_dict = {
            'output_dir': output_dir,
            'per_device_train_batch_size': self.config['batch_size'],
            'per_device_eval_batch_size': self.config['batch_size'],
            'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 1),
            'learning_rate': self.config['learning_rate'],
            'num_train_epochs': self.config['epochs'],
            'weight_decay': training_config.get('weight_decay', 0.01),
            'eval_strategy': "epoch",
            'save_strategy': "epoch",
            'save_total_limit': training_config['save_total_limit'],
            'logging_steps': training_config['logging_steps'],
            'predict_with_generate': True,
            'load_best_model_at_end': True,
            'metric_for_best_model': f"eval_{training_config['metric_for_best_model']}",
            'report_to': []
        }
        
        # Add bf16 and optimizer for T5-11B
        if self.config.get('use_bf16', False):
            args_dict['bf16'] = True
            args_dict['optim'] = 'adafactor'
            args_dict['weight_decay'] = 0.0
        
        return Seq2SeqTrainingArguments(**args_dict)
