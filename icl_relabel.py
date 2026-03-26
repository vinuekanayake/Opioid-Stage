import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from src.utils.config_loader import ConfigLoader
from src.icl.prompter import ICLPrompter
from src.icl.parser import ICLOutputParser

def relabel_with_icl(args):
    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_yaml("training_configs/icl_relabel.yaml")
    
    # Setup OpenAI client
    if args.api_key:
        client = OpenAI(api_key=args.api_key)
    else:
        # Try to load from environment
        client = OpenAI()
    
    # Initialize components
    prompter = ICLPrompter(config['guidelines'])
    parser = ICLOutputParser()
    
    # Load training data
    print(f"Loading training data from {config['data']['train_path']}...")
    train_df = pd.read_csv(config['data']['train_path'])
    
    # Output directory
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which ICL sets to process
    if args.icl_set is not None:
        icl_sets = [args.icl_set]
    else:
        icl_sets = range(1, config['icl']['num_sets'] + 1)
    
    # Process each ICL set
    for icl_idx in icl_sets:
        print(f"\n{'='*60}")
        print(f"Processing ICL Set {icl_idx}")
        print(f"{'='*60}")
        
        # Load ICL examples
        icl_path = Path(config['data']['icl_sets_dir']) / f"icl_set_{icl_idx}.csv"
        if not icl_path.exists():
            print(f"Warning: {icl_path} not found. Skipping.")
            continue
        
        icl_df = pd.read_csv(icl_path)
        
        # Format ICL examples
        icl_examples = []
        for _, row in icl_df.iterrows():
            post_text, rationale = prompter.extract_post_and_rationale(row["text"])
            label = row["label"]
            icl_examples.append(
                prompter.format_icl_example(post_text, label, rationale)
            )
        
        print(f"Loaded {len(icl_examples)} ICL examples")
        
        # Relabel all training examples
        results = []
        failed_count = 0
        
        for idx, row in tqdm(
            train_df.iterrows(),
            total=len(train_df),
            desc=f"Relabeling with ICL set {icl_idx}"
        ):
            post_text = row["text"]
            
            # Build prompt
            prompt = prompter.build_prompt(icl_examples, post_text)
            
            try:
                # Call API
                response = client.chat.completions.create(
                    model=config['api']['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config['api'].get('temperature', 0.0),
                    # max_tokens=config['api'].get('max_tokens', 500)
                )
                print(response)
                raw_output = response.choices[0].message.content
                
                # Parse output
                label, rationale = parser.parse_output(raw_output)
                
                if label and rationale:
                    formatted_text = parser.format_with_rationale(post_text, rationale)
                    results.append({
                        "text": formatted_text,
                        "label": label,
                        "raw_output": raw_output
                    })
                else:
                    failed_count += 1
                    if args.verbose:
                        print(f"\nFailed to parse row {idx}: {raw_output}")
                    results.append({
                        "text": "",
                        "label": "",
                        "raw_output": raw_output
                    })
            
            except Exception as e:
                failed_count += 1
                if args.verbose:
                    print(f"\nError processing row {idx}: {e}")
                results.append({
                    "text": "",
                    "label": "",
                    "raw_output": str(e)
                })
        
        # Save results
        out_path = output_dir / f"relabel_set_{icl_idx}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        
        print(f"\nSaved {len(results)} relabeled posts to {out_path}")
        print(f"Failed parses: {failed_count}/{len(results)} ({100*failed_count/len(results):.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICL-based relabeling with GPT-5")
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env variable)"
    )
    parser.add_argument(
        "--icl_set",
        type=int,
        default=None,
        help="Process specific ICL set only (1-10). If not specified, processes all."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed error messages"
    )
    
    args = parser.parse_args()
    relabel_with_icl(args)
