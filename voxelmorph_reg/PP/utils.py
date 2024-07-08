import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

"""
Common informatiom: patches are of size 256x256 ,
so reconstruction will be set with size 256 automatically
"""

#HELPER FUNCTIONS ------- decomstruct an image -------------- 

# Used in the dataset to get the input for the image
def deconstruct(img):
        
    """
    ----------- Suppose these steps are already completed
    
        image_path = 'path_to_your_image.jpg'
        image = cv2.imread(image_path)

        # Convert the image from BGR to RGB 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    """
    height, width, channels = img.shape
    flattened_img = img.reshape(-1)
    flattened_tensor = torch.tensor(flattened_img, dtype=torch.uint8)
    
    # to reconstruct:
    # 
    
    return flattened_tensor

#Building the image back from a flattened vector: will use after PP makes the prediction

def reconstruct(flattened_vector, h=256, w=256, c=3):
    reconstructed_image = flattened_vector.numpy().reshape((h, w, c)) # ATTENTION: the reconstructed dimension will be up for change
    return reconstructed_image

#Used in the alternating training process to freeze one model and train the other

def swap_training(network_to_train, network_to_not_train):
    for param in network_to_not_train.parameters():
        param.requires_grad = False

    for param in network_to_train.parameters():
        param.requires_grad = True

    network_to_not_train.eval()
    network_to_train.train()



from torch.utils.data import Dataset
import os
import cv2


#Dataset for stained and not stained data : #TODO: put all the filtered split patches in one directory and mount from 31.10
# Process each data individually
class PP_data(Dataset):
    def __init__(self, stained_img_dir,not_stained_img_dir, registration = None):
        self.img_dir_s = stained_img_dir
        self.img_dir_ns = not_stained_img_dir
        self.transform = registration
        self.imgs_stained = os.listdir(stained_img_dir)
        self.imgs_not_tained = os.listdir(not_stained_img_dir)
        
        
    def __len__(self):
        return len(self.imgs_stained)
    
    def __getitem__(self,idx):
        stained_img_path = os.path.join(self.img_dir_s, self.imgs_stained[idx])
        not_stained_img_path = os.path.join(self.img_dir_ns, "not_"+self.imgs_stained[idx])
        
        stained_img = cv2.imread(stained_img_path)
        stained_img = cv2.cvtColor(stained_img, cv2.COLOR_BGR2RGB)
        
        not_stained_img = cv2.imread(not_stained_img_path)
        not_stained_img = cv2.cvtColor(not_stained_img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            not_stained_img, stained_img = self.transform(not_stained_img, stained_img)
        
        stained_img = stained_img/255.0
        not_stained_img = not_stained_img/255.0
        
        stained_flattened = deconstruct(stained_img)
        not_stained_flattened = deconstruct(not_stained_img)
        
        data_list = [[not_stained_flattened[i], stained_flattened[i]] for i in range(len(stained_flattened))]
        
        data = np.array(data_list)
        
        #OR 
        """
            {'pixel_values': not_stained_flattened, 'y': stainede_flattened}
        """
        
        return data
        
    
    