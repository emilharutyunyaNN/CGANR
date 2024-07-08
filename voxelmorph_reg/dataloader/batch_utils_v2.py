"""import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets, Dataset, Image, Features, Array3D
import datasets
import cv2
import os
import numpy as np
import random
import matplotlib
from scipy.io import loadmat
import glob
matplotlib.use('Agg')  # Use a non-interactive backend suitable for environments without a display
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
#from GAN_R_v6 import init_parameters
import torchvision.transforms as transforms
from configobj import ConfigObj


def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()

    tc.model_path = '/home/hkhz/emil/Project_ai/voxelmorph_reg/models' # set the path to save model
    tc.prev_checkpoint_path = None
    tc.save_every_epoch = True

    # pretrained checkpoints to start from
    tc.G_warmstart_checkpoint = None
    tc.D_warmstart_checkpoint = None
    tc.R_warmstart_checkpoint = None
    assert not (tc.prev_checkpoint_path
                and (tc.G_warmstart_checkpoint or tc.D_warmstart_checkpoint or tc.R_warmstart_checkpoint))

    tc.image_path = 'L:/Pneumonia_Dataset/Second_reg/Training/target/*.mat' # path for training data 
    vc.image_path = 'J:/Pneumonia_Dataset/Second_reg/Validation/target/*.mat' # path for validation data

    def convert_inp_path_from_target(inp_path: str):
        return inp_path.replace('target', 'input')

    tc.convert_inp_path_from_target = convert_inp_path_from_target
    vc.convert_inp_path_from_target = convert_inp_path_from_target

    tc.is_mat, vc.is_mat = True, True  # True for .mat, False for .npy
    tc.data_inpnorm, vc.data_inpnorm = 'norm_by_mean_std', 'norm_by_mean_std'
    tc.channel_start_index, vc.channel_start_index = 0, 0
    tc.channel_end_index, vc.channel_end_index = 3, 3  # exclusive

    # network and loss params
    tc.is_training, vc.is_training = True, False
    tc.image_size, vc.image_size = 256, 256
    tc.num_slices, vc.num_slices = 3, 3
    tc.label_channels, vc.label_channels = 3, 3
    assert tc.channel_end_index - tc.channel_start_index == tc.num_slices
    assert vc.channel_end_index - vc.channel_start_index == vc.num_slices
    tc.n_channels, vc.n_channels = 32, 32
    tc.lamda = 50.0  # adv loss

    tc.nf_enc, vc.nf_enc = [8, 16, 16, 32, 32], [8, 16, 16, 32, 32]  # for aligner
    tc.nf_dec, vc.nf_dec = [32, 32, 32, 32, 32, 16, 16], [32, 32, 32, 32, 32, 16, 16]  # for aligner
    tc.R_loss_type = 'ncc'
    tc.lambda_r_tv = 1.0  # .1    # tv of predicted flow
    tc.gauss_kernel_size = 79
    tc.dvf_clipping = True  # clip DVF to [mu-sigma*dvf_clipping_nsigma, mu+sigma*dvf_clipping_nsigma]
    tc.dvf_clipping_nsigma = 3
    tc.dvf_thresholding = True  # clip DVF to [-dvf_thresholding_distance, dvf_thresholding_distance]
    tc.dvf_thresholding_distance = 30

    # training params
    tc.batch_size, vc.batch_size = 4, 4
    tc.n_shuffle_epoch, vc.n_shuffle_epoch = 1, 5000  # for the batchloader
    tc.initial_alternate_steps = 6000  # train G/D for initial_alternate_steps steps before switching to R for the same # of steps
    tc.valid_steps = 100  # perform validation when D_steps % valid_steps == 0 or at the end of a loop of (train G/D, train R)
    tc.n_threads, vc.n_threads = 2, 2
    tc.q_limit, vc.q_limit = 100, 300
    tc.N_epoch = 20  # number of loops

    tc.tol = 0  # current early stopping patience
    tc.max_tol = 2  # the max-allowed early stopping patience
    tc.min_del = 0  # the lowest acceptable loss value reduction

    # case filtering
    tc.case_filtering = False
    tc.case_filtering_metric = 'ncc'  # 'ncc'
    # divide each patch into case_filtering_x_subdivision patches alone the x dimension for filtering (1 = no division)
    tc.case_filtering_x_subdivision = 2
    tc.case_filtering_y_subdivision = 2
    assert tc.case_filtering_x_subdivision >= 1 and tc.case_filtering_y_subdivision >= 1
    tc.case_filtering_starting_epoch = 2  # case filtering only when epoch >= case_filtering_starting_epoch
    tc.case_filtering_cur_mean, tc.case_filtering_cur_stdev = 0.3757, 0.0654  # for lung elastic (256x256 patch)
    tc.case_filtering_nsigma = 2
    tc.case_filtering_recalc_every_eval = True

    # case filtering for dataloader
    tc.filter_blank, vc.filter_blank = True, True
    tc.filter_threshold, vc.filter_threshold = 0.9515, 0.9515  # 0.9515 for elastic/MT

    # per-pixel loss mask to account for out of the field information brought in by R
    tc.loss_mask, vc.loss_mask = False, False  # True, False

    # training resume parameters
    tc.epoch_begin = 0
    # this overrides tc.epoch_  begin the training schedule; tc.epoch_begin is required for logging
    # set it to None when not used
    tc.iter_begin =  None

    return tc, vc


class BatchLoader(object):
    def __init__(self, images, config, input_channels, is_testing, n_parallel_calls, q_limit, n_epoch):
        self.num_epoch = n_epoch
        self.images = images
        self.PATHS = []
        self.config = config
        self.is_testing = is_testing
        self.input_channels = input_channels
        self.label_channels = config.label_channels
        self.raw_size = config.image_size
        self.image_size = config.image_size
        self.num_parallel_calls = n_parallel_calls
        self.q_limit = q_limit

        for i in range(self.num_epoch):
            if self.config.is_training:
                random.shuffle(self.images)
            self.PATHS.extend(self.images)
        print("Length of the image file list: " + str(len(self.PATHS)))

        self.dataset = self.create_dataset_from_generator()

        assert (self.is_testing and not self.config.is_training) or (not self.is_testing and self.config.is_training)
        if self.config.is_training:
            self.dataset = self.dataset.shuffle(seed=random.randint(0, 10000))
            self.dataset = self.dataset.map(self.augment)

        self.dataloader = self.create_dataloader(self.dataset)

    def create_dataset_from_generator(self):
        raise NotImplementedError

    def augment(self, img, lab):
        raise NotImplementedError

    def create_dataloader(self, dataset):
        if self.config.is_training:
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_threads)
        else:
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.n_threads)

    def shuffle_dataset(self):
        self.dataset = self.dataset.shuffle(seed=random.randint(0, 10000))

    def get_dataloader(self):
        self.shuffle_dataset()
        self.dataloader = self.create_dataloader(self.dataset)
        return self.dataloader
    

class ImageTransformationBatchLoader(BatchLoader):
    def __init__(self, train_images, tc, num_slices, **kwargs):
        super().__init__(train_images, tc, num_slices, **kwargs)

        if not hasattr(self.config, 'filter_blank'):
            self.config.filter_blank = False
        elif self.config.filter_blank:
            assert self.config.filter_threshold is not None

        self.cur_filter_count = 0
        self.case_trial_limit = 10

    def create_dataset_from_generator(self):
        #dataset = Dataset.from_list(self.PATHS)
        with ThreadPoolExecutor(max_workers=self.config.n_threads) as executor:
            futures = [executor.submit(self.create_dataset, [path]) for path in self.PATHS]
            dataset_list = [future.result() for future in futures]
        return interleave_datasets(dataset_list, probabilities=[1/len(dataset_list)]*len(dataset_list))
    def generator_function(self, paths):
        for path in paths:
            for img, lab in self.parse_and_generate(path):
                yield {'image': img, 'label': lab}

    def create_dataset(self, paths):
        print("here")
        features = Features({
            'image': Array3D(shape=(self.image_size,self.image_size,self.input_channels), dtype='float32'),
            'label': Array3D(shape=(self.image_size,self.image_size,self.input_channels), dtype='float32')

        })
        return Dataset.from_generator(self.generator_function, gen_kwargs={
            'paths': paths
    }, features=features,streaming = True)

    def parse_and_generate(self, path):
        print("inside generator")
        s = self.config.image_size
        stride = s

        start, end = self.config.channel_start_index, self.config.channel_end_index
        #path = path.decode('UTF-8')
        print("passed")

        if self.config.is_mat:
            print("mat")
            # image = loadmat(path.replace('target', 'input'))['input'].astype(np.float32)[:, :, start:end]
            try:
                print("trying ...")
                image = loadmat(path.replace("target", "input")).get('input').astype(np.float32)[:, :, start:end]
                label = loadmat(path).get('target').astype(np.float32) #/ 255.0
            except:
                print(self.config.convert_inp_path_from_target(path))
        else:
            # image = np.transpose(np.load(path.replace('target', 'input')).astype(np.float32)[start:end, :, :], axes=[1, 2, 0])
            image = np.transpose(
                np.load(self.config.convert_inp_path_from_target(path)).astype(np.float32)[:, :, start:end],
                axes=[1, 2, 0])
            label = np.transpose(np.load(path).astype(np.float32), axes=[1, 2, 0]) #/ 255.0

        if self.config.data_inpnorm == 'norm_by_specified_value':
            normalize_vector = [1500, 1500, 1500, 1000]
            normalize_vector = np.reshape(normalize_vector, [1, 1, 4])
            image = image / normalize_vector
        elif self.config.data_inpnorm == 'norm_by_mean_std':
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        crop_edge = 30
        image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        label = label[crop_edge:-crop_edge, crop_edge:-crop_edge, :]

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

                    if self.config.filter_blank and np.mean(lab) >= self.config.filter_threshold \
                            and cur_trial_count < self.case_trial_limit:
                        # print("debug: fitered out patch with mean:", np.mean(lab))
                        # self.cur_filter_count += 1
                        # plt.imsave('filtered_label_patch_{}.jpg'.format(self.cur_filter_count), lab)
                        cur_trial_count += 1
                        # if cur_trial_count == self.case_trial_limit:
                        #     print("Blank filtering max trial reached")
                        continue
                    else:
                        # if 0 < cur_trial_count < self.case_trial_limit:
                        #     print("Blank filtering helps +1")
                        yield (img.astype(np.float32), lab.astype(np.float32))

                if yy == size - s:
                    break
                y += stride
            if xx == size - s:
                break
            x += stride

    def augment(self, img, lab):
        img = torch.tensor(img)
        lab = torch.tensor(lab)

        # Convert to PIL Image
        img = transforms.ToPILImage()(img)
        lab = transforms.ToPILImage()(lab)

        # Apply random horizontal flip
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            lab = transforms.functional.hflip(lab)

        # Apply random rotation
        angle = random.choice([0, 90, 180, 270])
        img = transforms.functional.rotate(img, angle)
        lab = transforms.functional.rotate(lab, angle)

        # Convert back to tensor
        img = transforms.ToTensor()(img)
        lab = transforms.ToTensor()(lab)

        return img, lab


training_imgs = glob.glob("/home/hkhz/daihui/Training/target/*.mat")
validation_imgs = glob.glob("/home/hkhz/daihui/Validation/target/*.mat")
tc,vc = init_parameters()
train_bl = ImageTransformationBatchLoader(training_imgs, tc, tc.num_slices, is_testing=False,
                                              n_parallel_calls=tc.n_threads, q_limit=tc.q_limit,
                                              n_epoch=tc.n_shuffle_epoch)

iterator_train_bl = iter(train_bl.dataset)

print("DATA check")
print(next(iterator_train_bl))

def load_mat_file(file_path):
    data = loadmat(file_path)

    return{
        "input": data.get("input"),
        "target":data.get("target")
    }


example_file = training_imgs[0]
example_data = load_mat_file(example_file)
print("Example data: ", example_data)"""
"""training_data = [load_mat_file(file) for file in training_imgs]
validation_data = [load_mat_file(file) for file in validation_imgs]

dataset_training = Dataset.from_list(training_data)

dataset_validation = Dataset.from_list(validation_data)"""

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, interleave_datasets, Features, Array3D
import numpy as np
import random
import matplotlib
from scipy.io import loadmat
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchvision.transforms as transforms
from configobj import ConfigObj

matplotlib.use('Agg')  # Use a non-interactive backend suitable for environments without a display


def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()

    tc.model_path = '/home/hkhz/emil/Project_ai/voxelmorph_reg/models'  # set the path to save model
    tc.prev_checkpoint_path = None
    tc.save_every_epoch = True

    # pretrained checkpoints to start from
    tc.G_warmstart_checkpoint = None
    tc.D_warmstart_checkpoint = None
    tc.R_warmstart_checkpoint = None
    assert not (tc.prev_checkpoint_path and (tc.G_warmstart_checkpoint or tc.D_warmstart_checkpoint or tc.R_warmstart_checkpoint))

    tc.image_path = 'L:/Pneumonia_Dataset/Second_reg/Training/target/*.mat'  # path for training data
    vc.image_path = 'J:/Pneumonia_Dataset/Second_reg/Validation/target/*.mat'  # path for validation data

    def convert_inp_path_from_target(inp_path: str):
        return inp_path.replace('target', 'input')

    tc.convert_inp_path_from_target = convert_inp_path_from_target
    vc.convert_inp_path_from_target = convert_inp_path_from_target

    tc.is_mat, vc.is_mat = True, True  # True for .mat, False for .npy
    tc.data_inpnorm, vc.data_inpnorm = 'norm_by_mean_std', 'norm_by_mean_std'
    tc.channel_start_index, vc.channel_start_index = 0, 0
    tc.channel_end_index, vc.channel_end_index = 3, 3  # exclusive

    # network and loss params
    tc.is_training, vc.is_training = True, False
    tc.image_size, vc.image_size = 256, 256
    tc.num_slices, vc.num_slices = 3, 3
    tc.label_channels, vc.label_channels = 3, 3
    assert tc.channel_end_index - tc.channel_start_index == tc.num_slices
    assert vc.channel_end_index - vc.channel_start_index == vc.num_slices
    tc.n_channels, vc.n_channels = 32, 32
    tc.lamda = 50.0  # adv loss

    tc.nf_enc, vc.nf_enc = [8, 16, 16, 32, 32], [8, 16, 16, 32, 32]  # for aligner
    tc.nf_dec, vc.nf_dec = [32, 32, 32, 32, 32, 16, 16], [32, 32, 32, 32, 32, 16, 16]  # for aligner
    tc.R_loss_type = 'ncc'
    tc.lambda_r_tv = 1.0  # .1    # tv of predicted flow
    tc.gauss_kernel_size = 79
    tc.dvf_clipping = True  # clip DVF to [mu-sigma*dvf_clipping_nsigma, mu+sigma*dvf_clipping_nsigma]
    tc.dvf_clipping_nsigma = 3
    tc.dvf_thresholding = True  # clip DVF to [-dvf_thresholding_distance, dvf_thresholding_distance]
    tc.dvf_thresholding_distance = 30

    # training params
    tc.batch_size, vc.batch_size = 4, 4
    tc.n_shuffle_epoch, vc.n_shuffle_epoch = 1, 5000  # for the batchloader
    tc.initial_alternate_steps = 6000  # train G/D for initial_alternate_steps steps before switching to R for the same # of steps
    tc.valid_steps = 100  # perform validation when D_steps % valid_steps == 0 or at the end of a loop of (train G/D, train R)
    tc.n_threads, vc.n_threads = 2, 2
    tc.q_limit, vc.q_limit = 100, 300
    tc.N_epoch = 20  # number of loops

    tc.tol = 0  # current early stopping patience
    tc.max_tol = 2  # the max-allowed early stopping patience
    tc.min_del = 0  # the lowest acceptable loss value reduction

    # case filtering
    tc.case_filtering = False
    tc.case_filtering_metric = 'ncc'  # 'ncc'
    # divide each patch into case_filtering_x_subdivision patches alone the x dimension for filtering (1 = no division)
    tc.case_filtering_x_subdivision = 2
    tc.case_filtering_y_subdivision = 2
    assert tc.case_filtering_x_subdivision >= 1 and tc.case_filtering_y_subdivision >= 1
    tc.case_filtering_starting_epoch = 2  # case filtering only when epoch >= case_filtering_starting_epoch
    tc.case_filtering_cur_mean, tc.case_filtering_cur_stdev = 0.3757, 0.0654  # for lung elastic (256x256 patch)
    tc.case_filtering_nsigma = 2
    tc.case_filtering_recalc_every_eval = True

    # case filtering for dataloader
    tc.filter_blank, vc.filter_blank = True, True
    tc.filter_threshold, vc.filter_threshold = 0.9515, 0.9515  # 0.9515 for elastic/MT

    # per-pixel loss mask to account for out of the field information brought in by R
    tc.loss_mask, vc.loss_mask = False, False  # True, False

    # training resume parameters
    tc.epoch_begin = 0
    # this overrides tc.epoch_  begin the training schedule; tc.epoch_begin is required for logging
    # set it to None when not used
    tc.iter_begin = None

    return tc, vc


class ImageTransformationBatchLoader:
    def __init__(self, images, config, num_slices, is_testing, n_parallel_calls, q_limit, n_epoch):
        self.num_epoch = n_epoch
        self.images = images
        self.PATHS = []
        self.config = config
        self.is_testing = is_testing
        self.num_slices = num_slices
        self.label_channels = config.label_channels
        self.raw_size = config.image_size
        self.image_size = config.image_size
        self.num_parallel_calls = n_parallel_calls
        self.q_limit = q_limit

        for i in range(self.num_epoch):
            if self.config.is_training:
                random.shuffle(self.images)
            self.PATHS.extend(self.images[:10])
        print("Length of the image file list: " + str(len(self.PATHS)))

        self.dataset = self.create_dataset_from_generator()

        if self.config.is_training:
            
            augmented_data = [self.augment_example(example) for example in self.dataset]
            self.dataset = Dataset.from_dict({'image': [d['image'] for d in augmented_data],
                                              'label': [d['label'] for d in augmented_data]})

        self.dataloader = self.create_dataloader(self.dataset)

    def create_dataset_from_generator(self):
        #with ThreadPoolExecutor(max_workers=self.config.n_threads) as executor:
         #   futures = [executor.submit(self.create_dataset, [path]) for path in self.PATHS]
        dataset_list = [self.create_dataset([path]) for path in self.PATHS]
        return interleave_datasets(dataset_list, probabilities=[1 / len(dataset_list)] * len(dataset_list))

    def generator_function(self, paths):
        for path in paths:
            for img, lab in self.parse_and_generate(path):
                yield {'image': img, 'label': lab}

    def create_dataset(self, paths):
        try:
            features = Features({
                'image': Array3D(shape=(self.image_size, self.image_size, self.num_slices), dtype='float32'),
                'label': Array3D(shape=(self.image_size, self.image_size, self.label_channels), dtype='float32')
            })
            dataset = Dataset.from_generator(self.generator_function, gen_kwargs={'paths': paths}, features=features, streaming=True)
            
            #if len(dataset) == 0:
            #    print(f"No data in dataset for paths: {paths}")
            #    return None
            return dataset
        except Exception as e:
            print(f"Error creating dataset for paths {paths}: {e}")
            return None

    def parse_and_generate(self, path):
        s = self.config.image_size
        stride = s

        start, end = self.config.channel_start_index, self.config.channel_end_index

        if self.config.is_mat:
            try:
                image = loadmat(path.replace("target", "input")).get('input').astype(np.float32)[:, :, start:end]
                label = loadmat(path).get('target').astype(np.float32)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return
        else:
            image = np.transpose(
                np.load(self.config.convert_inp_path_from_target(path)).astype(np.float32)[:, :, start:end],
                axes=[1, 2, 0])
            label = np.transpose(np.load(path).astype(np.float32), axes=[1, 2, 0])

        if self.config.data_inpnorm == 'norm_by_specified_value':
            normalize_vector = [1500, 1500, 1500, 1000]
            normalize_vector = np.reshape(normalize_vector, [1, 1, 4])
            image = image / normalize_vector
        elif self.config.data_inpnorm == 'norm_by_mean_std':
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        crop_edge = 30
        image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        label = label[crop_edge:-crop_edge, crop_edge:-crop_edge, :]

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

                    if self.config.filter_blank and np.mean(lab) >= self.config.filter_threshold \
                            and cur_trial_count < self.case_trial_limit:
                        cur_trial_count += 1
                        continue
                    else:
                        yield img.astype(np.float32), lab.astype(np.float32)

                if yy == size - s:
                    break
                y += stride
            if xx == size - s:
                break
            x += stride

    def augment_example(self, example):
        img = torch.tensor(example['image'])
        lab = torch.tensor(example['label'])
        
        #img = img.permute(1,2,0)
        #lab = lab.permute(1,2,0)
        print(img.shape)
        print(lab.shape)
        # Convert to PIL Image
        try:
            img = transforms.ToPILImage()(img)
            lab = transforms.ToPILImage()(lab)
        except:
            print(img.shape, lab.shape)    
        # Apply random horizontal flip
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            lab = transforms.functional.hflip(lab)

        # Apply random rotation
        angle = random.choice([0, 90, 180, 270])
        img = transforms.functional.rotate(img, angle)
        lab = transforms.functional.rotate(lab, angle)
        img = img.permute(2,0,1)
        lab = lab.permute(2,0,1)
        # Convert back to tensor
        example['image'] = np.array(img)
        example['label'] = np.array(lab)

        return example

    def create_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=self.config.is_training, num_workers=self.config.n_threads)


# Initialize parameters
tc, vc = init_parameters()

# Load training and validation images
training_imgs = glob.glob("/home/hkhz/daihui/Training/target/*.mat")
validation_imgs = glob.glob("/home/hkhz/daihui/Validation/target/*.mat")

# Create data loaders
"""train_bl = ImageTransformationBatchLoader(training_imgs, tc, tc.num_slices, is_testing=False,
                                          n_parallel_calls=tc.n_threads, q_limit=tc.q_limit,
                                          n_epoch=tc.n_shuffle_epoch)
"""
# Check data
#iterator_train_bl = iter(train_bl.dataloader)
print("DATA check")
#print(next(iterator_train_bl))

import os
import numpy as np
from scipy.io import loadmat
from datasets import Dataset, Features, Array3D

#paths_all = []
def extend(paths, epoch_num, train_flag = True):
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

def dynamic_dataset_generator(file_paths):
    features = Features({
        'input': Array3D(dtype='float32', shape=(256, 256, 3)),
        'target': Array3D(dtype='float32', shape=(256, 256, 3))
    })
    
    for path in file_paths:
        def generator(path=path):  # Capture path correctly
            return generate_patch(path)
        dataset = Dataset.from_generator(generator, features=features, streaming=True)
        print(f"Created dataset for: {path}")
        yield dataset
        
def load_mat_dataset(data_dir, epochs=1):
    file_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if '.mat' in fname]
    file_paths = extend(file_paths, epochs)
    print(len(file_paths))
    datasets = list(dynamic_dataset_generator(file_paths))
    dataset = interleave_datasets(datasets, probabilities=[1/len(datasets)]*len(datasets), stopping_strategy="all_exhausted")
    return dataset
"""def generate_examples(file_paths):
    for file_path in file_paths:
        example = load_mat_file(file_path)
        yield example

def load_mat_dataset(data_dir):
    file_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if '.mat' in fname]
    file_paths = extend(file_paths, 500)
    print("Length of files: ", file_paths)
    features = Features({
        'input': Array3D(dtype='float32', shape=(256, 256, 3)),  # Adjust the shape according to your data
        'target': Array3D(dtype='float32', shape=(256, 256, 3))  # Adjust the shape according to your data
    })
    datasets = [Dataset.from_generator(lambda: generate_patch(path), features=features, streaming=True) for path in file_paths]
    #dataset = Dataset.from_generator(lambda: generate_examples(file_paths), features=features, streaming=True)
    dataset = interleave_datasets(datasets, probabilities= [1/len(datasets)]*len(datasets), stopping_strategy="all_exhausted")
    return dataset"""

from datasets import load_dataset
data_dir = "/home/hkhz/daihui/Training/target"
dataset = load_mat_dataset(data_dir, 500)
iter_d = iter(dataset)
print(next(iter_d))
"""def load_mat_file(file_path):
    data = loadmat(file_path)
    return {
        "input": data.get("input"),
        "target": data.get("target")
    }
"""
"""example_file = training_imgs[0]
example_data = load_mat_file(example_file)
print("Example data: ", example_data)"""
