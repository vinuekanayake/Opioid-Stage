import argparse
from pathlib import Path
from transformers import set_seed, Seq2SeqTrainer

from src.utils.config_loader import ConfigLoader
from src.utils.reasoning_utils import print_reasoning_report, save_reasoning_predictions
from src.data_loader import ReasoningDataLoader
from src.models.t5_reasoning_classifier import T5ReasoningClassifier


def main(args):
    # Load configs
    config_loader = ConfigLoader()
    configs = config_loader.load_all_configs(
        f"{args.model}_reasoning",
        args.data_type
    )
    
    # Override with reasoning config
    reasoning_config_path = f"training_configs/reasoning_{args.reasoning_type}.yaml"
    reasoning_config = config_loader.load_yaml(reasoning_config_path)
    configs['training'].update(reasoning_config)
    
    # Set seed
    set_seed(configs['training']['seed'])
    
    # Setup output directory
    output_dir = Path(configs['paths']['output']['checkpoints']) / \
                 f"{args.model}_reasoning_{args.reasoning_type}_{args.data_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup results directory
    results_dir = Path(configs['paths']['output']['predictions'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Update data paths for reasoning
    base_name = f"train_{args.data_type}_explanation_deepseek_reasoning_forced.csv"
    configs['data_paths']['train'] = f"data/worker_data/{base_name}"
    
    # Load data
    print(f"Loading data with {args.reasoning_type} reasoning...")
    data_loader = ReasoningDataLoader(
        data_paths=configs['data_paths'],
        labels=configs['training']['labels'],
        reasoning_column=configs['training']['reasoning_column'],
        val_split=configs['training']['val_split'],
        seed=configs['training']['seed']
    )
    
    train_ds, valid_ds, worker_test_ds, expert_test_ds = \
        data_loader.load_and_prepare_data()
    
    # Initialize model
    print(f"Initializing {configs['model']['model_name']} with reasoning distillation...")
    cache_dir = configs['paths'].get('cache', {}).get('hf_cache')
    
    model_wrapper = T5ReasoningClassifier(
        configs['model'],
        configs['training'],
        data_loader.label2id,
        data_loader.id2label,
        cache_dir=cache_dir
    )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    
    # Train and valid use reasoning
    remove_cols_train = ["label", "text", "label_str", configs['training']['reasoning_column']]
    train_ds = train_ds.map(
        model_wrapper.tokenize_train_function,
        batched=True,
        remove_columns=remove_cols_train
    )
    valid_ds = valid_ds.map(
        model_wrapper.tokenize_train_function,
        batched=True,
        remove_columns=remove_cols_train
    )
    
    # Test sets don't have reasoning
    remove_cols_test = ["label"]
    worker_test_ds = worker_test_ds.map(
        model_wrapper.tokenize_test_function,
        batched=True,
        remove_columns=remove_cols_test
    )
    expert_test_ds = expert_test_ds.map(
        model_wrapper.tokenize_test_function,
        batched=True,
        remove_columns=remove_cols_test
    )
    
    # Get training arguments
    training_args = model_wrapper.get_training_args(str(output_dir))
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
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
    try:
        trainer.train()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
    
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
    print_reasoning_report(
        "Validation",
        trainer.predict(valid_ds),
        model_wrapper.tokenizer,
        data_loader.label2id,
        configs['training']['labels']
    )
    
    print_reasoning_report(
        "Worker-held-out Test",
        worker_results,
        model_wrapper.tokenizer,
        data_loader.label2id,
        configs['training']['labels']
    )
    
    print_reasoning_report(
        "Expert-held-out Test",
        expert_results,
        model_wrapper.tokenizer,
        data_loader.label2id,
        configs['training']['labels']
    )
    
    # Save predictions
    print("\n" + "="*50)
    print("SAVING PREDICTIONS")
    print("="*50)
    
    suffix = f"{args.data_type}_explanation_{args.reasoning_type}_reasoning"
    
    save_reasoning_predictions(
        worker_test_ds,
        worker_results,
        model_wrapper.tokenizer,
        results_dir / f"worker_test_{suffix}.csv",
        data_loader.label2id
    )
    
    save_reasoning_predictions(
        expert_test_ds,
        expert_results,
        model_wrapper.tokenizer,
        results_dir / f"expert_test_{suffix}.csv",
        data_loader.label2id
    )
    
    print(f"\n Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train T5 models with reasoning distillation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["t5_3b", "t5_11b"],
        help="Model to train"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="wo",
        choices=["w", "wo"],
        help="Data type: 'w' (with explanation) or 'wo' (without explanation)"
    )
    parser.add_argument(
        "--reasoning_type",
        type=str,
        required=True,
        choices=["summarized", "stepbystep"],
        help="Type of reasoning: 'summarized' or 'stepbystep'"
    )
    
    args = parser.parse_args()
    main(args)
