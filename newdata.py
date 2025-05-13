import os
import shutil

source_dir = "/home/minh/Desktop/continual-compression-Min/cloc/datasets/coco/train2017"
target_dir = "/home/minh/Desktop/continual-compression-Min/cloc/datasets/coco/new"
index_file = "/home/minh/Desktop/continual-compression-Min/top_val_indices.txt"

os.makedirs(target_dir, exist_ok=True)

with open(index_file, 'r') as f:
    indices = [int(line.strip()) - 1 for line in f if line.strip().isdigit()]

all_files = sorted(os.listdir(source_dir))

max_index = max(indices)
if max_index > len(all_files):
    print(f"File chỉ mục vượt quá số ảnh ({max_index} >= {len(all_files)})")
    exit()

for i in indices:
    filename = all_files[i]
    src = os.path.join(source_dir, filename)
    dst = os.path.join(target_dir, filename)
    shutil.copy(src, dst)
