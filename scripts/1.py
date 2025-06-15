import os
import re
from collections import defaultdict

# Define paths to your folders
task1_path = r"D:\Fraud Detection\data\SROIE Dataset\SROIE2019\SROIE2019\task1_train"
task2_path = r"D:\Fraud Detection\data\SROIE Dataset\SROIE2019\SROIE2019\task2_train"

def get_base_filenames_with_duplicates(folder):
    base_filenames = defaultdict(int)
    all_files = []

    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            all_files.append(file)
            # Remove suffixes like (1), (2)
            base_name = re.sub(r'\(\d+\)', '', os.path.splitext(file)[0]).strip()
            base_filenames[base_name] += 1
    return base_filenames, all_files

# Get filenames and counts for task1
task1_base_counts, task1_all = get_base_filenames_with_duplicates(task1_path)

# Print basic stats
print(f"\nTotal image files in task1: {len(task1_all)}")

# Find and show duplicates
duplicates = {k: v for k, v in task1_base_counts.items() if v > 1}
print(f"\nDuplicated base filenames in task1: {len(duplicates)}")

for name, count in list(duplicates.items())[:5]:  # just show a few
    print(f"{name} -> {count} files")
