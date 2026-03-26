import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

class DeBERTaClassifier:
    def __init__(self, config, label2id, id2label):
        self.config = config
        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = len(label2id)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_name'],
            use_fast=config.get('use_fast_tokenizer', False)
        )
        
        # Add special tokens if needed
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[title]", "[text]", "[Rationale]"]}
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
    
    def tokenize_function(self, batch):
        """Tokenize input text."""
        return self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.config['max_length']
        )
    
    def compute_metrics(self, pred):
        """Compute metrics for evaluation."""
        from src.utils.metrics import compute_classification_metrics
        preds = np.argmax(pred.predictions, axis=1)
        return compute_classification_metrics(pred.label_ids, preds)
    
    def get_training_args(self, output_dir, training_config):
        """Get TrainingArguments."""
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            learning_rate=self.config['learning_rate'],
            num_train_epochs=self.config['epochs'],
            weight_decay=training_config['weight_decay'],
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=training_config['save_total_limit'],
            logging_steps=training_config['logging_steps'],
            load_best_model_at_end=True,
            metric_for_best_model=training_config['metric_for_best_model'],
            disable_tqdm=False,
            report_to=[]
        )
