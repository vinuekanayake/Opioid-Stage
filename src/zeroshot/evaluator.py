import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, Any


class ZeroShotEvaluator:
    """Evaluates zero-shot classification performance."""
    
    def __init__(self, valid_labels: list):
        self.valid_labels = valid_labels
    
    def evaluate(
        self,
        df: pd.DataFrame,
        gold_col: str = "label",
        pred_col: str = "predicted_label"
    ) -> Dict[str, Any]:
        """
        Evaluate classifier performance.
        
        Args:
            df: DataFrame with predictions
            gold_col: Column with ground truth labels
            pred_col: Column with predicted labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Clean whitespace
        df[gold_col] = df[gold_col].astype(str).str.strip()
        df[pred_col] = df[pred_col].astype(str).str.strip()
        
        # Remove rows where model failed
        df_valid = df[df[pred_col].isin(self.valid_labels)].copy()
        
        total_rows = len(df)
        valid_rows = len(df_valid)
        dropped_rows = total_rows - valid_rows
        
        print(f"\nTotal rows: {total_rows}")
        print(f"Valid predictions: {valid_rows}")
        print(f"Dropped (invalid outputs): {dropped_rows}")
        
        if valid_rows == 0:
            print("ERROR: No valid predictions to evaluate!")
            return {
                "accuracy": 0.0,
                "valid_predictions": 0,
                "total_predictions": total_rows,
                "report": "No valid predictions"
            }
        
        y_true = df_valid[gold_col]
        y_pred = df_valid[pred_col]
        
        # Calculate accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # Generate classification report
        report = classification_report(
            y_true,
            y_pred,
            labels=self.valid_labels,
            digits=4,
            zero_division=0
        )
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nAccuracy: {acc:.4f}")
        print("\nPer-Class Metrics (Precision / Recall / F1):")
        print(report)
        
        return {
            "accuracy": acc,
            "valid_predictions": valid_rows,
            "total_predictions": total_rows,
            "report": report,
            "valid_df": df_valid
        }
    
    def get_confusion_analysis(
        self,
        df: pd.DataFrame,
        gold_col: str = "label",
        pred_col: str = "predicted_label"
    ) -> pd.DataFrame:
        """
        Get detailed confusion analysis.
        
        Args:
            df: DataFrame with predictions
            gold_col: Column with ground truth labels
            pred_col: Column with predicted labels
            
        Returns:
            DataFrame with misclassified examples
        """
        df_valid = df[df[pred_col].isin(self.valid_labels)].copy()
        misclassified = df_valid[df_valid[gold_col] != df_valid[pred_col]].copy()
        
        return misclassified[['text', gold_col, pred_col]]
