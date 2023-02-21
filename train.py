import datasets
import os
from pathlib import Path

NUM_WORKERS = os.cpu_count()

### TEST FUNCTIONALITY
dataset_path = Path("datasets/dtd/dtd")
split_ratio = [0.6, 0.2, 0.2]
BATCH_SIZE = 32

train_dataloader, val_dataloader, test_dataloader = datasets.create_dataloaders(
    dataset_dir=dataset_path,
    split_ratio=split_ratio,
    transform=None,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

class_names = train_dataloader.dataset.dataset.class_names

print(f"Created DataLoaders with {len(train_dataloader)} training images," \
    f"{len(val_dataloader)} validation images and {len(test_dataloader)} testing images.")
print(f"Dataset contains {len(class_names)} classes.")




aa = 1