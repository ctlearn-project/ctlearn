import os
import random
import shutil
import sys

data_path = sys.argv[1]
split_path = sys.argv[2]

train_split = 0.8
val_split = 0.1
test_split = 0.1

# For each class, get the filenames of the data
class_directories = [c for c in os.listdir(data_path) if 
        os.path.isdir(data_path+'/'+c)]
class_filenames = []
for class_dir in class_directories:
    class_filenames.append([f for f in os.listdir(data_path + class_dir)])
    random.shuffle(class_filenames[-1])
    print("Read in", len(class_filenames[-1]), "filenames in class",
            class_dir+'.')

# Create the directories
for class_dir in class_directories:
    try:
        os.makedirs(split_path+"/train/"+class_dir)
    except OSError:
        if not os.path.isdir(split_path+"/train/"+class_dir):
            raise
    try:
        os.makedirs(split_path+"/val/"+class_dir)
    except OSError:
        if not os.path.isdir(split_path+"/val/"+class_dir):
            raise
    try:
        os.makedirs(split_path+"/test/"+class_dir)
    except OSError:
        if not os.path.isdir(split_path+"/test/"+class_dir):
            raise

for class_dir, class_files in zip(class_directories, class_filenames):
    # Split into training, validation, and test sets
    print("Splitting class", class_dir+'...')
    train_files = class_files[:int(len(class_files)*train_split)]
    print("Split", len(train_files), "training files.")
    val_files = class_files[int(len(class_files)*train_split):
            int(len(class_files)*(train_split + val_split))]
    print("Split", len(val_files), "validation files.")
    test_files = class_files[int(len(class_files)*(train_split + val_split)):]
    print("Split", len(test_files), "test files.")
    
    # Copy the data or symlinks
    print("Copying data or symlinks...")
    for f in train_files:
        shutil.copy(data_path+'/'+class_dir+'/'+f, 
                split_path+"/train/"+class_dir+'/'+f, follow_symlinks=False)
    for f in val_files:
        shutil.copy(data_path+'/'+class_dir+'/'+f,
                split_path+"/val/"+class_dir+'/'+f, follow_symlinks=False)
    for f in test_files:
        shutil.copy(data_path+'/'+class_dir+'/'+f,
                split_path+"/test/"+class_dir+'/'+f, follow_symlinks=False)
