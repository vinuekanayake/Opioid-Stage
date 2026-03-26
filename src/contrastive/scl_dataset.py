import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from src.contrastive.augmentations import TextAugmenter


class SCLDataset(Dataset):
    """Dataset for supervised contrastive learning."""
    
    def __init__(self, csv_path: str, label2id: dict, augmenter: TextAugmenter):
        df = pd.read_csv(csv_path)
        
        self.texts = df["text"].astype(str).tolist()
        self.labels = [label2id[s] for s in df["label"].astype(str).tolist()]
        self.augmenter = augmenter
        
        # Build class-to-indices mapping for balanced sampler
        self.class_to_indices = defaultdict(list)
        for i, lab in enumerate(self.labels):
            self.class_to_indices[lab].append(i)
    
    def __len__(self):
        return len(self.texts)
    
    def get_raw(self, idx):
        """Get raw text and label."""
        return self.texts[idx], self.labels[idx]
    
    def __getitem__(self, idx):
        """Get two augmented views and label."""
        text, label = self.get_raw(idx)
        
        view1 = text  # Original view
        view2 = self.augmenter.augment_post(text)  # Augmented view
        
        return {
            "text": view1,
            "text_aug": view2,
            "label": label
        }
