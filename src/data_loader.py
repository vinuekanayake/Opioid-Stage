import pandas as pd
from datasets import Dataset
from typing import Dict, Tuple

class OUDDataLoader:
    def __init__(self, data_paths: Dict[str, str], labels: list, val_split: float = 0.1, seed: int = 42):
        self.data_paths = data_paths
        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.val_split = val_split
        self.seed = seed
    
    def load_and_prepare_data(self) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
        """Load and prepare all datasets."""
        # Load CSVs
        train_df = pd.read_csv(self.data_paths['train'])
        worker_eval_df = pd.read_csv(self.data_paths['worker_eval'])
        expert_eval_df = pd.read_csv(self.data_paths['expert_eval'])
        
        # Map labels
        for df in (train_df, worker_eval_df, expert_eval_df):
            df["label_id"] = df["label"].map(self.label2id)
            df["label_str"] = df["label"]  # Keep string version for T5
        
        # Create HF Datasets
        full_train_ds = Dataset.from_pandas(train_df)
        worker_test_ds = Dataset.from_pandas(worker_eval_df)
        expert_test_ds = Dataset.from_pandas(expert_eval_df)
        
        # Split train → (train, valid)
        split = full_train_ds.train_test_split(test_size=self.val_split, seed=self.seed)
        train_ds = split["train"]
        valid_ds = split["test"]
        
        return train_ds, valid_ds, worker_test_ds, expert_test_ds
    
class ReasoningDataLoader(OUDDataLoader):
    """Extended data loader for reasoning distillation."""
    
    def __init__(self, data_paths, labels, reasoning_column, val_split=0.1, seed=42):
        super().__init__(data_paths, labels, val_split, seed)
        self.reasoning_column = reasoning_column
    
    def load_and_prepare_data(self):
        """Load data with reasoning column."""
        
        # Load CSVs
        train_df = pd.read_csv(self.data_paths['train'])
        worker_eval_df = pd.read_csv(self.data_paths['worker_eval'])
        expert_eval_df = pd.read_csv(self.data_paths['expert_eval'])
        
        # Verify reasoning column exists in training data
        if self.reasoning_column not in train_df.columns:
            raise ValueError(
                f"Reasoning column '{self.reasoning_column}' not found in training data. "
                f"Available columns: {train_df.columns.tolist()}"
            )
        
        # Map labels and prepare string versions
        for df in (train_df, worker_eval_df, expert_eval_df):
            df["label_id"] = df["label"].map(self.label2id)
            df["label_str"] = df["label"]
        
        # Create HF Datasets
        full_train_ds = Dataset.from_pandas(train_df)
        worker_test_ds = Dataset.from_pandas(worker_eval_df)
        expert_test_ds = Dataset.from_pandas(expert_eval_df)
        
        # Split train
        split = full_train_ds.train_test_split(test_size=self.val_split, seed=self.seed)
        train_ds = split["train"]
        valid_ds = split["test"]
        
        return train_ds, valid_ds, worker_test_ds, expert_test_ds

