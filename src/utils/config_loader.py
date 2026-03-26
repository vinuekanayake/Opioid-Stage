# import yaml
# from pathlib import Path
# from typing import Dict, Any

# class ConfigLoader:
#     def __init__(self, config_dir: str = "config"):
#         self.config_dir = Path(config_dir)
    
#     def load_yaml(self, file_path: str) -> Dict[str, Any]:
#         """Load a YAML config file."""
#         with open(self.config_dir / file_path, 'r') as f:
#             return yaml.safe_load(f)
    
#     def load_all_configs(self, model_name: str, data_type: str = "wo"):
#         """Load all necessary configs for training."""
#         paths = self.load_yaml("paths.yaml")
#         model_config = self.load_yaml(f"model_configs/{model_name}.yaml")
#         training_config = self.load_yaml("training_configs/baseline.yaml")
        
#         # Set data paths based on data_type
#         data_paths = {
#             'train': paths['data'][f'worker_train_{data_type}_explanation'],
#             'worker_eval': paths['data'][f'worker_eval_{data_type}_explanation'],
#             'expert_eval': paths['data'][f'expert_eval_{data_type}_explanation']
#         }
        
#         return {
#             'paths': paths,
#             'data_paths': data_paths,
#             'model': model_config,
#             'training': training_config
#         }


import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
    
    def load_yaml(self, file_path: str) -> Dict[str, Any]:
        """Load a YAML config file."""
        full_path = self.config_dir / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")
        
        with open(full_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_all_configs(
        self, 
        model_name: str, 
        data_type: str = "wo",
        training_config: str = "baseline",
        use_filtered_expert: bool = False
    ) -> Dict[str, Any]:
        """
        Load all necessary configs for training.
        
        Args:
            model_name: Name of model config (e.g., 'deberta_base', 't5_3b')
            data_type: 'w' or 'wo' (with/without explanations)
            training_config: Name of training config (e.g., 'baseline', 'icl_finetune')
            use_filtered_expert: Whether to use filtered expert test set (for ICL experiments)
        
        Returns:
            Dictionary with all loaded configs
        """
        paths = self.load_yaml("paths.yaml")
        model_config = self.load_yaml(f"model_configs/{model_name}.yaml")
        train_config = self.load_yaml(f"training_configs/{training_config}.yaml")
        
        # Set data paths based on data_type
        data_paths = {
            'train': paths['data'][f'worker_train_{data_type}_explanation'],
            'worker_eval': paths['data'][f'worker_eval_{data_type}_explanation'],
        }
        
        # Handle expert test set (filtered or original)
        if use_filtered_expert:
            expert_key = f'expert_eval_{data_type}_explanation_filtered'
            if expert_key in paths['data']:
                data_paths['expert_eval'] = paths['data'][expert_key]
            else:
                print(f"Warning: Filtered expert set not found, using original")
                data_paths['expert_eval'] = paths['data'][f'expert_eval_{data_type}_explanation']
        else:
            data_paths['expert_eval'] = paths['data'][f'expert_eval_{data_type}_explanation']
        
        return {
            'paths': paths,
            'data_paths': data_paths,
            'model': model_config,
            'training': train_config
        }
    
    def load_icl_configs(
        self, 
        model_name: str, 
        data_type: str = "wo"
    ) -> Dict[str, Any]:
        """
        Load configs specifically for ICL training.
        
        Args:
            model_name: Name of model config
            data_type: 'w' or 'wo'
        
        Returns:
            Dictionary with all ICL-specific configs
        """
        configs = self.load_all_configs(
            model_name=model_name,
            data_type=data_type,
            training_config="icl_finetune",
            use_filtered_expert=True  # ICL always uses filtered expert sets
        )
        
        # Override training data path with ICL relabeled data
        icl_config = configs['training']
        if icl_config.get('use_icl_data', False):
            icl_train_path = icl_config.get('icl_data_path')
            if icl_train_path and Path(icl_train_path).exists():
                configs['data_paths']['train'] = icl_train_path
            else:
                raise FileNotFoundError(
                    f"ICL relabeled data not found at {icl_train_path}. "
                    "Run icl_majority_vote.py first."
                )
        
        return configs
    
    def load_scl_configs(
        self,
        model_name: str,
        data_type: str = "wo",
        stage: str = "pretrain"
    ) -> Dict[str, Any]:
        """
        Load configs for SCL training.
        
        Args:
            model_name: Name of model config
            data_type: 'w' or 'wo'
            stage: 'pretrain' or 'finetune'
        
        Returns:
            Dictionary with SCL-specific configs
        """
        if stage not in ["pretrain", "finetune"]:
            raise ValueError(f"stage must be 'pretrain' or 'finetune', got {stage}")
        
        configs = self.load_all_configs(
            model_name=model_name,
            data_type=data_type,
            training_config=f"scl_{stage}",
            use_filtered_expert=False  # SCL uses original test sets
        )
        
        return configs
    
    def get_data_path(
        self,
        dataset: str,
        data_type: str,
        filtered: bool = False
    ) -> str:
        """
        Get specific data path.
        
        Args:
            dataset: 'worker_train', 'worker_eval', or 'expert_eval'
            data_type: 'w' or 'wo'
            filtered: Whether to use filtered version (only for expert_eval)
        
        Returns:
            Path to data file
        """
        paths = self.load_yaml("paths.yaml")
        
        # Build key
        key = f"{dataset}_{data_type}_explanation"
        if filtered and dataset == "expert_eval":
            key += "_filtered"
        
        if key not in paths['data']:
            raise KeyError(f"Data path not found for key: {key}")
        
        return paths['data'][key]
    
    def get_output_paths(self) -> Dict[str, str]:
        """Get all output directory paths."""
        paths = self.load_yaml("paths.yaml")
        return paths['output']
    
    def get_cache_path(self) -> str:
        """Get HuggingFace cache path."""
        paths = self.load_yaml("paths.yaml")
        return paths['cache']['hf_cache']
    
    def validate_paths(self, configs: Dict[str, Any]) -> bool:
        """
        Validate that all data paths exist.
        
        Args:
            configs: Config dictionary from load_all_configs
        
        Returns:
            True if all paths exist, raises FileNotFoundError otherwise
        """
        data_paths = configs['data_paths']
        
        for name, path in data_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{name} data not found at {path}"
                )
        
        return True
    
    def merge_configs(
        self, 
        base_config: Dict[str, Any], 
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two config dictionaries (override takes precedence).
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
        
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
