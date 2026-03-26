import argparse
from pathlib import Path
import pandas as pd

from src.utils.config_loader import ConfigLoader
from src.icl.majority_voter import MajorityVoter

def perform_majority_voting(args):
    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_yaml("training_configs/icl_relabel.yaml")
    
    # Setup paths
    # results_dir = Path(config['data']['icl_sets_dir'])
    results_dir = Path(config['data']['output_dir'])
    output_dir = Path("data/icl_relabeled/combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original data
    print("Loading original training data...")
    original_df = pd.read_csv(config['data']['train_path'])
    
    # Initialize voter
    voter = MajorityVoter(num_icl_sets=config['icl']['num_sets'])
    
    # Load all relabeling results
    print(f"Loading relabeling results from {results_dir}...")
    all_labels = voter.load_relabel_results(str(results_dir))
    
    # Perform voting
    print("Performing majority voting...")
    combined_df = voter.combine_and_vote(original_df, all_labels)
    
    # Save detailed results
    combined_path = output_dir / "combined_labels_detailed.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved detailed results to {combined_path}")
    
    # Create final dataset
    final_df = voter.create_final_dataset(
        combined_df,
        min_agreement=args.min_agreement
    )
    
    # Save final dataset
    final_path = output_dir / "train_icl_relabeled.csv"
    final_df.to_csv(final_path, index=False)
    print(f"Saved final ICL-relabeled dataset to {final_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("VOTING STATISTICS")
    print("="*60)
    print(f"Total examples: {len(combined_df)}")
    print(f"Final examples (after filtering): {len(final_df)}")
    print("\nAgreement distribution:")
    print(combined_df['mode_count'].value_counts().sort_index())
    print("\nLabel distribution (original):")
    print(original_df['label'].value_counts())
    print("\nLabel distribution (ICL-relabeled):")
    print(final_df['label'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Majority voting for ICL relabeling")
    parser.add_argument(
        "--min_agreement",
        type=int,
        default=None,
        help="Minimum number of ICL sets that must agree (default: no filtering)"
    )
    
    args = parser.parse_args()
    perform_majority_voting(args)
