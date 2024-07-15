"""import argparse
import h5py
import numpy as np
import os
import tarfile
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def main(args):
    tar_path = args.tarfile
    hdf5_path = os.path.splitext(tar_path)[0] + '.hdf5'

    # Using libver='latest' for handling large datasets
    hf = h5py.File(hdf5_path, 'w', libver='latest')

    print('Reading {} ...'.format(tar_path))
    tf = tarfile.open(tar_path)
    img_count = 0

    groups = {}

    for tarinfo in tqdm(tf, total=len(tf.getmembers())):
        if not tarinfo.isreg():
            continue
        
        tn = tarinfo.name
        path_parts = tn.split('/')

        fn = path_parts[-1]
        class_name = path_parts[-2]
        dataset_name = "/".join([path_parts[-3], path_parts[-2]])

        if dataset_name not in groups:
            grp = hf.create_group(dataset_name)
            groups[dataset_name] = grp
        else:
            grp = groups[dataset_name]

        try:
            image_file = tf.extractfile(tarinfo)
            if image_file is None:
                print(f"Failed to extract {tn}. Skipping...")
                continue

            image_data = image_file.read()
            if not image_data:
                print(f"Empty image data for {tn}. Skipping...")
                continue

            image_np = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Failed to decode image {tn}. Skipping...")
                continue

            # Ensure image is in RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Plot the image for verification
            

            ds = grp.create_dataset(fn, data=image, dtype=np.uint8)
            ds.attrs['class'] = class_name

            img_count += 1
        except Exception as e:
            print(f"Error processing {tn}: {e}")

    tf.close()
    hf.close()

    print('Created {} with {} images.'.format(hdf5_path, img_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert tar to hdf5')
    parser.add_argument('tarfile')
    args = parser.parse_args()
    main(args)"""
    
    
import argparse
import h5py
import numpy as np
import os
import tarfile
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def main(args):
    tar_path = args.tarfile
    hdf5_path = os.path.splitext(tar_path)[0] + '.hdf5'

    # Check if the tar file exists
    if not os.path.exists(tar_path):
        print(f"Error: The tar file '{tar_path}' does not exist.")
        return

    # Using libver='latest' for handling large datasets
    hf = h5py.File(hdf5_path, 'w', libver='latest')

    print('Reading {} ...'.format(tar_path))
    
    try:
        tf = tarfile.open(tar_path)
    except Exception as e:
        print(f"Error opening tar file {tar_path}: {e}")
        return

    img_count = 0
    groups = {}

    for tarinfo in tqdm(tf, total=len(tf.getmembers())):
        if not tarinfo.isreg():
            continue
        
        tn = tarinfo.name
        path_parts = tn.split('/')
        print(f"Processing: {tn}")
        print(f"Path parts: {path_parts}")

        if len(path_parts) < 4:
            print(f"Skipping {tn}: not enough path parts.")
            continue

        fn = path_parts[-1]
        x_name = path_parts[-2]
        type_name = path_parts[-3]
        dataset_name = path_parts[-4]

        dataset_type = 'Training' if 'training' in type_name else 'Validation'
        dataset_type_path = os.path.join(dataset_type, type_name)

        if dataset_type_path not in groups:
            grp = hf.create_group(dataset_type_path)
            groups[dataset_type_path] = grp
        else:
            grp = groups[dataset_type_path]

        try:
            image_file = tf.extractfile(tarinfo)
            if image_file is None:
                print(f"Failed to extract {tn}. Skipping...")
                continue

            image_data = image_file.read()
            if not image_data:
                print(f"Empty image data for {tn}. Skipping...")
                continue

            image_np = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Failed to decode image {tn}. Skipping...")
                continue

            # Ensure image is in RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Plot the image for verification
            """plt.imshow(image)
            plt.title(f"Image: {fn}")
            plt.savefig('./fig_create.jpg')"""

            ds = grp.create_dataset(f'{x_name}/{fn}', data=image, dtype=np.uint8)
            ds.attrs['class'] = type_name

            img_count += 1
        except Exception as e:
            print(f"Error processing {tn}: {e}")

    tf.close()
    hf.close()

    print('Created {} with {} images.'.format(hdf5_path, img_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert tar to hdf5')
    parser.add_argument('tarfile')
    args = parser.parse_args()
    main(args)
