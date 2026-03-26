import random
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """
    Samples batches with fixed number of classes and samples per class.
    Ensures each batch contains balanced representation from all classes.
    """
    
    def __init__(
        self,
        class_to_indices: dict,
        n_classes: int,
        n_samples: int,
        epoch_size: int = None
    ):
        self.class_to_indices = {k: list(v) for k, v in class_to_indices.items()}
        self.labels = list(self.class_to_indices.keys())
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        # Calculate epoch size if not provided
        if epoch_size is None:
            total_samples = sum(len(v) for v in self.class_to_indices.values())
            self.epoch_size = total_samples // (n_classes * n_samples)
        else:
            self.epoch_size = epoch_size
    
    def __iter__(self):
        for _ in range(self.epoch_size):
            # Randomly select classes for this batch
            chosen_classes = random.sample(self.labels, self.n_classes)
            batch_indices = []
            
            for cls in chosen_classes:
                indices = self.class_to_indices[cls]
                
                if len(indices) >= self.n_samples:
                    # Sample without replacement
                    sampled = random.sample(indices, self.n_samples)
                else:
                    # Oversample if not enough examples
                    sampled = random.choices(indices, k=self.n_samples)
                    
                batch_indices.extend(sampled)
            
            # Shuffle batch
            random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        return self.epoch_size
