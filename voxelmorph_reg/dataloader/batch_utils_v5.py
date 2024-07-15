
import os
import cv2
import numpy as np
import random
from scipy.io import loadmat
from datasets import Dataset, Features, Array3D, interleave_datasets
from configobj import ConfigObj
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from torchvision.transforms import functional

def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()

    tc.model_path = '/home/hkhz/emil/Project_ai/voxelmorph_reg/models'
    tc.prev_checkpoint_path = None
    tc.save_every_epoch = True

    tc.G_warmstart_checkpoint = None
    tc.D_warmstart_checkpoint = None
    tc.R_warmstart_checkpoint = None
    assert not (tc.prev_checkpoint_path and (tc.G_warmstart_checkpoint or tc.D_warmstart_checkpoint or tc.R_warmstart_checkpoint))

    tc.image_path = 'L:/Pneumonia_Dataset/Second_reg/Training/target/*.mat'
    vc.image_path = 'J:/Pneumonia_Dataset/Second_reg/Validation/target/*.mat'

    def convert_inp_path_from_target(inp_path: str):
        return inp_path.replace('target', 'input')

    tc.convert_inp_path_from_target = convert_inp_path_from_target
    vc.convert_inp_path_from_target = convert_inp_path_from_target

    tc.is_mat, vc.is_mat = True, True
    tc.data_inpnorm, vc.data_inpnorm = 'norm_by_mean_std', 'norm_by_mean_std'
    tc.channel_start_index, vc.channel_start_index = 0, 0
    tc.channel_end_index, vc.channel_end_index = 3, 3

    tc.is_training, vc.is_training = True, False
    tc.image_size, vc.image_size = 256, 256
    tc.num_slices, vc.num_slices = 3, 3
    tc.label_channels, vc.label_channels = 3, 3
    assert tc.channel_end_index - tc.channel_start_index == tc.num_slices
    assert vc.channel_end_index - vc.channel_start_index == vc.num_slices
    tc.n_channels, vc.n_channels = 32, 32
    tc.lamda = 50.0

    tc.nf_enc, vc.nf_enc = [8, 16, 16, 32, 32], [8, 16, 16, 32, 32]
    tc.nf_dec, vc.nf_dec = [32, 32, 32, 32, 32, 16, 16], [32, 32, 32, 32, 32, 16, 16]
    tc.R_loss_type = 'ncc'
    tc.lambda_r_tv = 1.0
    tc.gauss_kernel_size = 79
    tc.dvf_clipping = True
    tc.dvf_clipping_nsigma = 3
    tc.dvf_thresholding = True
    tc.dvf_thresholding_distance = 30

    tc.batch_size, vc.batch_size = 4, 4
    tc.n_shuffle_epoch, vc.n_shuffle_epoch = 1, 5000
    tc.initial_alternate_steps = 6000
    tc.valid_steps = 100
    tc.n_threads, vc.n_threads = 2, 2
    tc.q_limit, vc.q_limit = 100, 300
    tc.N_epoch = 20

    tc.tol = 0
    tc.max_tol = 2
    tc.min_del = 0

    tc.case_filtering = False
    tc.case_filtering_metric = 'ncc'
    tc.case_filtering_x_subdivision = 2
    tc.case_filtering_y_subdivision = 2
    assert tc.case_filtering_x_subdivision >= 1 and tc.case_filtering_y_subdivision >= 1
    tc.case_filtering_starting_epoch = 2
    tc.case_filtering_cur_mean, tc.case_filtering_cur_stdev = 0.3757, 0.0654
    tc.case_filtering_nsigma = 2
    tc.case_filtering_recalc_every_eval = True

    tc.filter_blank, vc.filter_blank = True, True
    tc.filter_threshold, vc.filter_threshold = 0.9515, 0.9515

    tc.loss_mask, vc.loss_mask = False, False

    tc.epoch_begin = 0
    tc.iter_begin = None

    return tc, vc

def generate_patch(file, image_size, is_mat = True, inpnorm = 'norm_by_mean_std', start = 0, end = 3):
        s = image_size
        stride = s
        #start, end = self.config.channel_start_index, self.config.channel_end_index
        if is_mat:
            try:
                image = loadmat(file.replace("target", "input")).get('input').astype(np.float32)[:, :, start:end]
                label = loadmat(file).get('target').astype(np.float32)
                #print("patch gen: ", image.shape, label.shape)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                return
        else:
            # handle other file types if needed
            try:
                image = cv2.cvtColor(cv2.imread(file.replace("target", "input"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                label = cv2.cvtColor(cv2.imread(file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                #print(image.shape)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                return
                
            pass

        if inpnorm == 'norm_by_specified_value':
            normalize_vector = [1500, 1500, 1500, 1000]
            normalize_vector = np.reshape(normalize_vector, [1, 1, 4])
            image = image / normalize_vector
        elif inpnorm == 'norm_by_mean_std':
            #print("?")
            #print(np.mean(image), np.std(image))
            #print("done")
           # print(np.min(image))
            #print("image: ", np.min(image), np.mean(image), np.max(image), np.std(image))
            #image = (image - np.mean(image)) / (np.std(image) + 1e-5)
            #print("image post std: ", np.min(image), np.mean(image), np.max(image))
           # print(np.min(image))
            pass
        crop_edge = 30
        #s = 256
        image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        label = label[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        size = min(image.shape[0],image.shape[1])
        #print("****", image.shape)
        patch_count = 0
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
                    patch_count+=1
                    #("sending: ", img.shape, lab.shape, xx, yy, s, image.shape, label.shape)
                    yield {'input': img.astype(np.float32), 'target': lab.astype(np.float32)}
                if yy == size - s:
                    break
                y += stride
            if xx == size - s:
                break
            x += stride
        #print(f"Generated {patch_count} patches from image {file}")

def generate_patch_wrapper(path, image_size, is_mat):
            def generator():
                yield from generate_patch(path, image_size, is_mat)
            return generator
        
def regular_generator(ex):
    def generator():
        for x in ex:
            yield x
    return generator
class TransformImageBatchLoader:
    def __init__(self, images, config, input_channels, is_testing, n_parallel_calls, q_limit, n_epoch, gen_batch=10):
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
        self.inp_norm = self.config.data_inpnorm
        self.start, self.end = self.config.channel_start_index, self.config.channel_end_index
        self.is_training = self.config.is_training
        self.is_mat = self.config.is_mat
        self.batch_size = self.config.batch_size
        assert (self.is_testing and not self.config.is_training) or (not self.is_testing and self.config.is_training)
        
        self.extend()
        self.dataset_generator = self.load_mat_dataset(gen_batch)

    
    
    def dynamic_dataset_generator(self, batch_size=10, reshuffle = False):
        features = Features({
            'input': Array3D(dtype='float32', shape=(256, 256, 3)),
            'target': Array3D(dtype='float32', shape=(256, 256, 3))
        })

        
        all_datasets = []
        def file_path_batches():
            for i in range(0, len(self.PATHS), batch_size):
                yield self.PATHS[i:i + batch_size]

        grad_list = []
        for batch_paths in file_path_batches():
            datasets = [Dataset.from_generator(generate_patch_wrapper(path, self.image_size, self.is_mat), features=features, streaming=True) for path in batch_paths]
            grad_list.extend(datasets)
            if reshuffle:
                random.shuffle(grad_list)
            interleaved_dataset = interleave_datasets(grad_list, probabilities=[1/len(grad_list)]*len(grad_list), stopping_strategy="all_exhausted")
            #for batch in interleave_datasets.iter(batch_size = 4)

            all_examples = []
            for batch in interleaved_dataset.iter(batch_size =self.batch_size):
                if len(batch['input']) !=self.batch_size:
                    #all_examples.append(batch)
                    break
                #print(batch['input'][0].shape)
                #all_examples.append(batch)
                #if len(all_examples) == self.batch_size:
                yield self.prepare_batch(batch)
                #all_examples = []
            
            
            #grad_list = [ds for ds in grad_list if not ds.is_exhausted()]
            """for example in self.batch_generator(interleaved_dataset, self.batch_size):
                if self.is_training:
                    yield self.augment_batch(example)
                else:
                    yield self.stack_batch(example)"""
    def prepare_batch(self, examples):
        if self.is_training:
            return self.augment(examples)
        else:
            return self.stack_batch(examples)                
    #
    def stack_batch(self, example):
        #print("----", batch[0]['input'].shape, batch[0]['target'].shape)
        img = torch.stack([torch.tensor(example['input'][i], dtype=torch.float32) for i in range(len(example['input']))])
        lab = torch.stack([torch.tensor(example['target'][i], dtype=torch.float32) for i in range(len(example['target']))])
        example['input'] = img
        example['target'] = lab

        
        return example
    def augment_batch(self, batch):
        #print("----", batch[0]['input'].shape, batch[0]['target'].shape)
        augmented = [self.augment(ex) for ex in batch]
        augmented_batch_input = torch.stack([aug['input'] for aug in augmented])
        augmented_batch_target = torch.stack([aug['target'] for aug in augmented])
        return {'input': augmented_batch_input, 'target': augmented_batch_target}      
    def batch_generator(self, dataset, batch_size):
        batch = []
        for example in dataset:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    def load_mat_dataset(self, gen_batch = 1, reshuffle = False):
        #file_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if '.mat' in fname]
        #file_paths = extend(file_paths, epochs)
        #print(f"Total file paths: {len(file_paths)}")
        return self.dynamic_dataset_generator(gen_batch, reshuffle)
    def extend(self):
        for _ in range(self.num_epoch):
            if self.config.is_training:
                random.shuffle(self.images)
            self.PATHS.extend(self.images)

    def augment(self, example):
        #print(example['input'][0])
        img = torch.stack([torch.tensor(example['input'][i], dtype=torch.float32) for i in range(len(example['input']))])
        lab = torch.stack([torch.tensor(example['target'][i], dtype=torch.float32) for i in range(len(example['target']))])
        print(img.shape, lab.shape)
        #img = torch.tensor(example['input']).contiguous()
        #lab = torch.tensor(example['target']).contiguous()
        #print("IMG, TGT: ", img.shape, lab.shape)
        print(img.shape)
        img = img.permute(0,3, 1, 2).contiguous()
        lab = lab.permute(0,3, 1, 2).contiguous()
        imglab = torch.cat([img, lab], dim=1).contiguous()
        flip = transforms.RandomHorizontalFlip(p=0.5)
        imglab = flip(imglab)
        
        num_channels = img.shape[1]
        img, lab = imglab[:,:num_channels].contiguous(), imglab[:,num_channels:].contiguous()
        print(img.shape, lab.shape)
        rotations = [0, 90, 180, 270]
        angle = random.choice(rotations)
        img = functional.rotate(img, angle)
        lab = functional.rotate(lab, angle)

        example['input'] = img
        example['target'] = lab

        
        return example
    
    def reset_paths(self):
        self.PATHS = []
        self.extend()
        self.dataset_generator = self.load_mat_dataset()
        return self
        
    
    def __iter__(self):
        self.PATHS = []
        self.extend()
        if self.is_training:
            self.dataset_generator = self.load_mat_dataset(reshuffle=True)
        else:
            self.dataset_generator = self.load_mat_dataset()
        return self
    
    def __next__(self):
        try:
            return next(self.dataset_generator)
        except StopIteration:
            self.epoch += 1
            if self.epoch > self.num_epoch:
                raise StopIteration
            self.extend()
            self.dataset_generator = self.load_mat_dataset()
            return next(self.dataset_generator)

"""# Initialize parameters
tc, vc = init_parameters()
import glob
# Load training images
training_imgs = glob.glob("/home/hkhz/daihui/Training/target/*.mat")
valid_imgs = glob.glob("/home/hkhz/daihui/Validation/target/*.mat")

training_imgs = glob.glob("/home/hkhz/remote_mnt/data/Training/target/*.jpg")
valid_imgs = glob.glob("/home/hkhz/remote_mnt/data/Validation/target/*.jpg")

tc.is_mat = False
vc.is_mat = False


train_dataset_iter = TransformImageBatchLoader(training_imgs, tc, tc.num_slices, is_testing=False,
                                               n_parallel_calls=tc.n_threads, q_limit=tc.q_limit,
                                               n_epoch=2)
valid_dataset_iter = TransformImageBatchLoader(valid_imgs, vc, vc.num_slices, is_testing=True,
                                              n_parallel_calls=vc.n_threads, q_limit=vc.q_limit,
                                              n_epoch=2)
print(train_dataset_iter.PATHS[:10])
print(valid_dataset_iter.PATHS[:10])
iterable_t = iter(train_dataset_iter)
iterable_v = iter(valid_dataset_iter)
# Use the iterator
import matplotlib.pyplot as plt

while True:
    i = 0
    k = 0
    try:
        i+=1
        data_t = next(iterable_t)
        #print(data_t)
        input, target = data_t['input'], data_t['target']
        
        j = 0
        print(input.shape, target.shape)
        #for inp, tgt in zip(input, target):
          #  inp = inp.squeeze(0).permute(1,2,0).numpy()
          #  tgt = tgt.squeeze(0).permute(1,2,0).numpy()
           # j+=1
           # plt.imsave(f"./verify/trainfig{i}_{j}_input.jpg", inp)
           # plt.imsave(f"./verify/trainfig{i}_{j}_target.jpg", tgt)
        
        k+=1
        j=0
        data_v = next(iterable_v)
        input, target = data_v['input'], data_v['target']
        #for inp, tgt in zip(input, target):
            #j+=1
           # inp = inp.squeeze(0).permute(1,2,0).numpy()
           # tgt = tgt.squeeze(0).permute(1,2,0).numpy()
           ## plt.imsave(f"./verify/validfig{k}_{j}_input.jpg", inp)
           # plt.imsave(f"./verify/validfig{k}_{j}_target.jpg", tgt)
        
        
    except StopIteration:
        print("Iteration stopped.")
        break"""