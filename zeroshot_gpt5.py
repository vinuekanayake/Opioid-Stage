import argparse
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from src.utils.config_loader import ConfigLoader
from src.zeroshot.prompter import ZeroShotPrompter
from src.zeroshot.evaluator import ZeroShotEvaluator


def classify_post(text: str, client: OpenAI, prompter: ZeroShotPrompter, config: dict) -> str:
    """
    Classify a single post using GPT-5.
    
    Args:
        text: Post text
        client: OpenAI client
        prompter: ZeroShotPrompter instance
        config: Configuration dictionary
        
    Returns:
        Predicted label or "ERROR"
    """
    prompt = prompter.build_prompt(text)
    
    try:
        response = client.chat.completions.create(
            model=config['api']['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=config['api']['temperature'],
            # max_tokens=config['api']['max_tokens']
        )
        label = response.choices[0].message.content.strip()
        print(label)
        return label
    except Exception as e:
        print(f"Error: {e}")
        return "ERROR"


def run_zeroshot_classification(args):
    """Main function for zero-shot classification."""
    
    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_yaml("training_configs/zeroshot_gpt5.yaml")
    
    # Setup OpenAI client
    if args.api_key:
        client = OpenAI(api_key=args.api_key)
    else:
        client = OpenAI()  # Uses OPENAI_API_KEY env variable
    
    # Initialize components
    prompter = ZeroShotPrompter(
        class_descriptions=config['class_descriptions'],
        labels=config['labels']
    )
    evaluator = ZeroShotEvaluator(valid_labels=config['labels'])
    
    # Determine data path
    dataset_key = f"{args.dataset}_eval_{args.data_type}"
    if dataset_key not in config['data']:
        print(f"ERROR: Unknown dataset combination: {args.dataset} + {args.data_type}")
        print(f"Available: {list(config['data'].keys())}")
        return
    
    data_path = config['data'][dataset_key]
    
    # Check if file exists
    if not Path(data_path).exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Note: This script expects filtered test sets (ICL examples removed)")
        return
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} examples")
    
    # Setup output
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.dataset}_{args.data_type}_zeroshot_gpt5.csv"
    
    # Check if we should resume
    if output_file.exists() and not args.overwrite:
        print(f"\nOutput file already exists: {output_file}")
        if args.resume:
            print("Resuming from existing predictions...")
            df = pd.read_csv(output_file)
            if 'predicted_label' not in df.columns:
                df['predicted_label'] = None
        else:
            print("Use --resume to continue or --overwrite to start fresh")
            return
    else:
        df['predicted_label'] = None
    
    # Classify posts
    print("\nClassifying posts...")
    
    # Determine which rows need classification
    if args.resume and 'predicted_label' in df.columns:
        mask = df['predicted_label'].isna() | (df['predicted_label'] == 'ERROR')
        to_classify = df[mask].index.tolist()
        print(f"Resuming: {len(to_classify)} posts remaining")
    else:
        to_classify = df.index.tolist()
    
    for idx in tqdm(to_classify, desc="Classifying"):
        text = str(df.at[idx, 'text'])
        
        # Skip if already classified (when resuming)
        if pd.notna(df.at[idx, 'predicted_label']) and df.at[idx, 'predicted_label'] != 'ERROR':
            continue
        
        # Classify
        label = classify_post(text, client, prompter, config)
        df.at[idx, 'predicted_label'] = label
        
        # Rate limiting
        time.sleep(config['api']['rate_limit_delay'])
        
        # Save periodically (every 10 predictions)
        if (idx + 1) % 10 == 0:
            df.to_csv(output_file, index=False)
    
    # Final save
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING ZERO-SHOT PERFORMANCE")
    print("="*60)
    
    results = evaluator.evaluate(df, gold_col="label", pred_col="predicted_label")
    
    # Save confusion analysis
    if args.save_errors:
        misclassified = evaluator.get_confusion_analysis(
            df, gold_col="label", pred_col="predicted_label"
        )
        if len(misclassified) > 0:
            error_file = output_dir / f"{args.dataset}_{args.data_type}_errors.csv"
            misclassified.to_csv(error_file, index=False)
            print(f"\nSaved {len(misclassified)} misclassified examples to: {error_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Data type: {args.data_type}")
    print(f"Model: {config['api']['model']}")
    print(f"Total examples: {results['total_predictions']}")
    print(f"Valid predictions: {results['valid_predictions']}")
    print(f"Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot classification with GPT-5"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["worker", "expert"],
        help="Dataset to evaluate (worker or expert)"
    )
    
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["w", "wo"],
        help="Data type: 'w' (with explanation) or 'wo' (without explanation)"
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env variable)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing predictions"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions"
    )
    
    parser.add_argument(
        "--save_errors",
        action="store_true",
        help="Save misclassified examples to separate file"
    )
    
    args = parser.parse_args()
    
    run_zeroshot_classification(args)
