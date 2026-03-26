import re
from typing import List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def extract_label_from_output(output_text: str) -> str:
    """
    Extract the label from reasoning output.
    Format: "Label. Reasoning: ..." or "Label. Step by step reasoning: ..."
    """
    return output_text.split('.')[0].strip()


def extract_label_from_end(output_text: str) -> str:
    """
    Alternative extraction for formats like "Reasoning: ... Label: LABEL."
    """
    match = re.search(r"label\s*:\s*(.+?)(?:\.|$)", output_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # Fallback: last token chunk
        parts = [p.strip() for p in output_text.split('.') if p.strip()]
        return parts[-1] if parts else ""


def compute_reasoning_metrics(eval_preds, tokenizer, label2id):
    """
    Compute metrics for reasoning-based outputs.
    """
    gen_tokens, label_tokens = eval_preds
    
    # Replace -100 with pad token
    label_tokens = np.where(
        label_tokens != -100, 
        label_tokens, 
        tokenizer.pad_token_id
    )
    gen_tokens = np.where(
        gen_tokens != -100, 
        gen_tokens, 
        tokenizer.pad_token_id
    )
    
    # Decode
    decoded_preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
    
    # Extract labels
    pred_labels = [extract_label_from_output(pred) for pred in decoded_preds]
    true_labels = [extract_label_from_output(label) for label in decoded_labels]
    
    # Map to IDs
    pred_ids = [label2id.get(label, -1) for label in pred_labels]
    label_ids = [label2id.get(label, -1) for label in true_labels]
    
    return {
        "accuracy": accuracy_score(label_ids, pred_ids),
        "f1_macro": f1_score(label_ids, pred_ids, average="macro"),
    }


def print_reasoning_report(name, prediction_output, tokenizer, label2id, label_names):
    """
    Print classification report for reasoning outputs.
    """
    from sklearn.metrics import classification_report
    
    pred_token_seqs = prediction_output.predictions
    true_token_seqs = prediction_output.label_ids
    
    # Replace -100
    true_token_seqs = np.where(
        true_token_seqs != -100, 
        true_token_seqs, 
        tokenizer.pad_token_id
    )
    pred_token_seqs = np.where(
        pred_token_seqs != -100, 
        pred_token_seqs, 
        tokenizer.pad_token_id
    )
    
    # Decode
    decoded_preds = tokenizer.batch_decode(pred_token_seqs, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(true_token_seqs, skip_special_tokens=True)
    
    # Extract labels
    pred_labels = [extract_label_from_output(text.strip()) for text in decoded_preds]
    true_labels = [extract_label_from_output(text.strip()) for text in decoded_labels]
    
    pred_ids = [label2id.get(lbl, -1) for lbl in pred_labels]
    label_ids = [label2id.get(lbl, -1) for lbl in true_labels]
    
    # Find unrecognized labels
    unrecognized_preds = {lbl for lbl in pred_labels if lbl not in label2id}
    unrecognized_labels = {lbl for lbl in true_labels if lbl not in label2id}
    
    if unrecognized_preds:
        print(f"\nUnrecognized predicted labels: {unrecognized_preds}")
    if unrecognized_labels:
        print(f"Unrecognized gold labels: {unrecognized_labels}")
    
    # Filter valid samples
    valid = [
        (p, l) for p, l in zip(pred_ids, label_ids) 
        if (p in label2id.values() and l in label2id.values())
    ]
    
    if not valid:
        print(f"{name} (no valid samples to report!)")
        return
    
    pred_ids, label_ids = zip(*valid)
    
    print(f"\n→ {name} classification report")
    print(classification_report(
        label_ids,
        pred_ids,
        labels=list(label2id.values()),
        target_names=label_names,
        digits=4,
        zero_division=0
    ))


def save_reasoning_predictions(dataset, predictions, tokenizer, output_path, label2id):
    """
    Save predictions with reasoning to CSV.
    """
    import pandas as pd
    
    generated_ids = predictions.predictions
    decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    original_texts = dataset["text"]
    gold_labels = dataset["label_str"]
    
    predicted_labels = [extract_label_from_output(out) for out in decoded_outputs]
    
    df = pd.DataFrame({
        "text": original_texts,
        "gold_label": gold_labels,
        "predicted_output": decoded_outputs,
        "predicted_label": predicted_labels,
    })
    
    df.to_csv(output_path, index=False)
    print(f" Saved predictions to {output_path}")
