import os
import shutil
from tqdm import tqdm

train_file = "./train.txt"
test_file = "./test.txt"
val_file = "./val.txt"

source_folder = "./training_data1_v2"

if not os.path.exists("./train"):
    os.makedirs("./train")  
if not os.path.exists("./test"):
    os.makedirs("./test")
if not os.path.exists("./val"):
    os.makedirs("./val")

with open(train_file, "r") as file:
        content = file.readlines()
        for sample in tqdm(content):
                shutil.move(os.path.join(source_folder, sample[:-1]), os.path.join("./train", sample[:-1]))
        file.close()

with open(test_file, "r") as file:
        content = file.readlines()
        for sample in tqdm(content):
                shutil.move(os.path.join(source_folder, sample[:-1]), os.path.join("./test", sample[:-1]))
        file.close()

with open(val_file, "r") as file:
        content = file.readlines()
        for sample in tqdm(content):
                shutil.move(os.path.join(source_folder, sample[:-1]), os.path.join("./val", sample[:-1]))
        file.close()
                