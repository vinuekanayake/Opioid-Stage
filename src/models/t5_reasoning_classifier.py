import numpy as np
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
)
from src.utils.reasoning_utils import compute_reasoning_metrics


class T5ReasoningClassifier:
    """T5 Classifier with reasoning distillation."""
    
    def __init__(self, config, training_config, label2id, id2label, cache_dir=None):
        self.config = config
        self.training_config = training_config
        self.label2id = label2id
        self.id2label = id2label
        self.prefix = config.get('prefix', 'classify:')
        
        # Reasoning-specific settings
        self.reasoning_column = training_config['reasoning_column']
        self.train_prompt = training_config['train_prompt']
        self.test_prompt = training_config['test_prompt']
        self.target_format = training_config['target_format']
        self.max_gen_length = training_config.get(
            'max_generation_length', 
            config.get('max_generation_length', 512)
        )
        
        # Initialize tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            config['model_name'],
            cache_dir=cache_dir
        )
        
        # Initialize model
        model_kwargs = {'cache_dir': cache_dir}
        
        if '11b' in config['model_name'].lower():
            model_kwargs.update({
                'torch_dtype': torch.bfloat16,
                'device_map': 'auto'
            })
        
        self.model = T5ForConditionalGeneration.from_pretrained(
            config['model_name'],
            **model_kwargs
        )
        
        # Configure generation
        self.model.generation_config.max_new_tokens = self.max_gen_length
        self.model.generation_config.min_new_tokens = 0
        
        # Gradient checkpointing for large models
        if config.get('use_gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model
        )
    
    def tokenize_train_function(self, batch):
        """Tokenize training data with reasoning."""
        # Format inputs
        inputs = [
            self.train_prompt.format(text=txt) 
            for txt in batch["text"]
        ]
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config['max_length'],
            truncation=True
        )
        
        # Format targets with reasoning
        targets = [
            self.target_format.format(label=label, reasoning=reasoning)
            for label, reasoning in zip(batch["label_str"], batch[self.reasoning_column])
        ]
        
        labels = self.tokenizer(
            targets,
            max_length=self.max_gen_length,
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def tokenize_test_function(self, batch):
        """Tokenize test data (no reasoning available)."""
        inputs = [
            self.test_prompt.format(text=txt)
            for txt in batch["text"]
        ]
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config['max_length'],
            truncation=True
        )
        
        # Only tokenize label (no reasoning)
        labels = self.tokenizer(
            batch["label_str"],
            max_length=10,
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def compute_metrics(self, eval_preds):
        """Compute metrics wrapper."""
        return compute_reasoning_metrics(
            eval_preds, 
            self.tokenizer, 
            self.label2id
        )
    
    def get_training_args(self, output_dir):
        """Get Seq2SeqTrainingArguments."""
        args_dict = {
            'output_dir': output_dir,
            'per_device_train_batch_size': self.config['batch_size'],
            'per_device_eval_batch_size': self.config['batch_size'],
            'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 1),
            'learning_rate': self.config['learning_rate'],
            'num_train_epochs': self.config['epochs'],
            'weight_decay': self.training_config.get('weight_decay', 0.01),
            'eval_strategy': "epoch",
            'save_strategy': "epoch",
            'save_total_limit': self.training_config['save_total_limit'],
            'logging_steps': self.training_config['logging_steps'],
            'predict_with_generate': True,
            'generation_max_length': self.max_gen_length,
            'load_best_model_at_end': True,
            'metric_for_best_model': f"eval_{self.training_config['metric_for_best_model']}",
            'report_to': []
        }
        
        # Add bf16 and optimizer for T5-11B
        if self.config.get('use_bf16', False):
            args_dict['bf16'] = True
            args_dict['optim'] = 'adafactor'
            args_dict['weight_decay'] = 0.0
        
        return Seq2SeqTrainingArguments(**args_dict)
