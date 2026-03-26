import argparse
import sys
from pathlib import Path

from src.utils.config_loader import ConfigLoader

def train_icl(args):
    """
    Wrapper script for training on ICL-relabeled data.
    
    Note: All 40 ICL examples come from expert data only.
    - Worker test sets: Use original (no ICL examples)
    - Expert test sets: Use filtered (40 ICL examples removed)
    """
    # Load configs
    config_loader = ConfigLoader()
    paths_config = config_loader.load_yaml("paths.yaml")
    icl_config = config_loader.load_yaml("training_configs/icl_finetune.yaml")
    
    # Update paths to use ICL data
    icl_train_path = icl_config['icl_data_path']
    
    # Check if ICL data exists
    if not Path(icl_train_path).exists():
        print(f"ERROR: ICL relabeled data not found at {icl_train_path}")
        print("Please run icl_majority_vote.py first to create the ICL dataset.")
        sys.exit(1)
    
    # Training data path
    paths_config['data'][f'worker_train_{args.data_type}_explanation'] = icl_train_path
    
    # Test set paths
    # Worker: use original (no ICL examples)
    # Expert: use filtered (40 ICL examples removed)
    if icl_config.get('use_filtered_expert_only', True):
        # Check if filtered expert sets exist
        filtered_expert_wo = Path(paths_config['data']['expert_eval_wo_explanation_filtered'])
        filtered_expert_w = Path(paths_config['data']['expert_eval_w_explanation_filtered'])
        
        if not filtered_expert_wo.exists() or not filtered_expert_w.exists():
            print("WARNING: Filtered expert test sets not found.")
            print(f"Expected: {filtered_expert_wo}")
            print(f"Expected: {filtered_expert_w}")
            print("Using original expert test sets (contains 40 ICL examples - data leakage!).")
        else:
            # Use filtered expert test sets
            paths_config['data'][f'expert_eval_{args.data_type}_explanation'] = \
                paths_config['data'][f'expert_eval_{args.data_type}_explanation_filtered']
            print(f"Using filtered expert test set (40 ICL examples removed)")
        
        # Worker test sets remain unchanged (no ICL examples)
        print(f"Using original worker test set (no ICL examples)")
    
    # Save temporary paths config
    temp_paths = Path("config/paths_icl_temp.yaml")
    import yaml
    with open(temp_paths, 'w') as f:
        yaml.dump(paths_config, f)
    
    # Import and run baseline training
    print("="*60)
    print("Training on ICL-relabeled data")
    print("="*60)
    print(f"Training data: {icl_train_path}")
    print(f"Model: {args.model}")
    print(f"Data type: {args.data_type}")
    print("="*60)
    print("Test sets:")
    print(f"  Worker: ORIGINAL (no ICL examples)")
    print(f"  Expert: FILTERED (40 ICL examples removed)")
    print("="*60)
    
    # Call baseline training script
    from train_baseline import main as train_baseline_main
    
    class Args:
        model = args.model
        data_type = args.data_type
    
    # Temporarily replace config loader to use ICL paths
    original_load = config_loader.load_yaml
    
    def load_with_icl(path):
        if 'paths.yaml' in path:
            return paths_config
        return original_load(path)
    
    config_loader.load_yaml = load_with_icl
    
    try:
        train_baseline_main(Args())
    finally:
        # Cleanup
        if temp_paths.exists():
            temp_paths.unlink()
        config_loader.load_yaml = original_load


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on ICL-relabeled data")
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
        help="Data type"
    )
    
    args = parser.parse_args()
    train_icl(args)
