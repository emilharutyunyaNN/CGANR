import os
import numpy as np
import random
from scipy.io import loadmat
from datasets import Dataset, Features, Array3D, interleave_datasets

def extend(paths, epoch_num, train_flag=True):
    paths_all = []
    for _ in range(epoch_num):
        if train_flag:
            random.shuffle(paths)
        paths_all.extend(paths)
    return paths_all

def load_mat_file(file_path):
    data = loadmat(file_path)
    return {
        "input": data.get("input"),
        "target": data.get("target")
    }

def generate_patch(file):
    image = loadmat(file.replace("target", "input")).get('input').astype(np.float32)[:, :, 0:3]
    label = loadmat(file).get('target').astype(np.float32)
    crop_edge = 30
    s = 256
    image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
    label = label[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
    stride = s
    size = image.shape[0]

    cur_trial_count = 0
    x = 0
    while True:
        y = 0
        while True:
            rand_choice_stride = random.randint(0, 15)
            xx = min(x + rand_choice_stride * s // 16, size - s)
            yy = min(y + rand_choice_stride * s // 16, size - s)
            if yy != size - s and xx != size - s:
                img = image[xx:xx + s, yy:yy + s]
                lab = label[xx:xx + s, yy:yy + s]
                print("extracted: ", img, lab)
                yield {'input':img.astype(np.float32), 'target':lab.astype(np.float32)}

            if yy == size - s:
                break
            y += stride
        if xx == size - s:
            break
        x += stride

def generate_patch_wrapper(path):
    def generator():
        for item in generate_patch(path):
            yield item
    return generator

def dynamic_dataset_generator(file_paths, batch_size=100):
    features = Features({
        'input': Array3D(dtype='float32', shape=(256, 256, 3)),
        'target': Array3D(dtype='float32', shape=(256, 256, 3))
    })

    def file_path_batches():
        for i in range(0, len(file_paths), batch_size):
            yield file_paths[i:i + batch_size]

    for batch_paths in file_path_batches():
        datasets = [Dataset.from_generator(generate_patch_wrapper(path), features=features, streaming=True) for path in batch_paths]
        interleaved_dataset = interleave_datasets(datasets, probabilities=[1/len(datasets)]*len(datasets), stopping_strategy="all_exhausted")
        for example in interleaved_dataset:
            yield example

def load_mat_dataset(data_dir, epochs=1, batch_size=100):
    file_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if '.mat' in fname]
    file_paths = extend(file_paths, epochs)
    print(f"Total file paths: {len(file_paths)}")
    return dynamic_dataset_generator(file_paths, batch_size)

# Example usage
data_dir = "/home/hkhz/daihui/Training/target"
dataset_iter = load_mat_dataset(data_dir, epochs=500, batch_size=1)
import time
while True:
# Use the iterator
    data= next(dataset_iter)

    input, target = data['input'], data['target']

    import matplotlib.pyplot as plt

    plt.imsave("./input_sample.jpg", input)
    plt.imsave("./target_sample.jpg", target)
    
    time.sleep(5)