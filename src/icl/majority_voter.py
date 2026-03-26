import pandas as pd
from collections import Counter
from typing import List, Tuple

class MajorityVoter:
    """Performs majority voting across multiple ICL relabeling results."""
    
    def __init__(self, num_icl_sets: int):
        self.num_icl_sets = num_icl_sets
    
    def load_relabel_results(self, results_dir: str) -> pd.DataFrame:
        """
        Load all relabeled datasets.
        
        Args:
            results_dir: Directory containing relabel_set_*.csv files
            
        Returns:
            DataFrame with all label columns
        """
        relabel_dfs = []
        
        for i in range(1, self.num_icl_sets + 1):
            temp = pd.read_csv(f"{results_dir}/relabel_set_{i}.csv")
            
            if 'label' in temp.columns:
                temp = temp[['label']].rename(columns={'label': f'label_{i}'})
                relabel_dfs.append(temp)
        
        # Concatenate all label columns horizontally
        return pd.concat(relabel_dfs, axis=1)
    
    @staticmethod
    def get_mode_and_count(label_row: pd.Series) -> Tuple[str, int]:
        """
        Get mode label and its count from a row of labels.
        
        Args:
            label_row: Series containing multiple label predictions
            
        Returns:
            Tuple of (mode_label, mode_count)
        """
        labels = label_row.tolist()
        counts = Counter(labels)
        mode_label, mode_count = counts.most_common(1)[0]
        return mode_label, mode_count
    
    def combine_and_vote(
        self,
        original_df: pd.DataFrame,
        all_labels: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine original data with relabeled results and perform voting.
        
        Args:
            original_df: Original dataset with 'text' and 'label' columns
            all_labels: DataFrame with all ICL label predictions
            
        Returns:
            Combined DataFrame with voting results
        """
        # Combine original with all labels
        combined = pd.concat(
            [original_df[['text', 'label']], all_labels],
            axis=1
        )
        combined = combined.rename(columns={'label': 'original_label'})
        
        # Compute mode label and count
        label_cols = [f'label_{i}' for i in range(1, self.num_icl_sets + 1)]
        combined[['mode_label', 'mode_count']] = combined[label_cols].apply(
            lambda row: pd.Series(self.get_mode_and_count(row)),
            axis=1
        )
        
        return combined
    
    def create_final_dataset(
        self,
        combined_df: pd.DataFrame,
        min_agreement: int = None
    ) -> pd.DataFrame:
        """
        Create final relabeled dataset.
        
        Args:
            combined_df: DataFrame with voting results
            min_agreement: Minimum number of agreeing labels to keep (optional)
            
        Returns:
            Final dataset with 'text' and 'label' columns
        """
        final_df = combined_df[['text', 'mode_label']].copy()
        final_df = final_df.rename(columns={'mode_label': 'label'})
        
        # Optional: filter by agreement threshold
        if min_agreement:
            mask = combined_df['mode_count'] >= min_agreement
            final_df = final_df[mask].reset_index(drop=True)
            print(f"Filtered to {len(final_df)} examples with >= {min_agreement} agreement")
        
        return final_df
