import os
import random
import shutil

input_dir = "processed_cats"
output_dir = "cats_split"

shutil.rmtree(output_dir, ignore_errors=True)

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

for split in ['train', 'val', 'test']:
    for cls in ['cute', 'ugly']:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

for cls in ['cute', 'ugly']:
    files = [f for f in os.listdir(os.path.join(input_dir, cls))
             if f.lower().endswith(".jpg")]
    random.shuffle(files)
    n = len(files)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    for i, f in enumerate(files):
        if i < train_end:
            dest = os.path.join(output_dir, 'train', cls, f)
        elif i < val_end:
            dest = os.path.join(output_dir, 'val', cls, f)
        else:
            dest = os.path.join(output_dir, 'test', cls, f)
        shutil.copy(os.path.join(input_dir, cls, f), dest)

print("Split done. Train/Val/Test counts:")
for split in ['train', 'val', 'test']:
    for cls in ['cute', 'ugly']:
        count = len(os.listdir(os.path.join(output_dir, split, cls)))
        print(f"{split} {cls}: {count}")
