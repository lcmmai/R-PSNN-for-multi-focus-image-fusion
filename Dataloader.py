import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image

def load_train_list(load_img1_path, load_img2_path, load_groundtruth_path): #get train list
    filename_list = os.listdir(load_img1_path)  #get file name of every image
                                                #the file name in each folder should be same
    dataset = []
    for filename in filename_list:
        dataset.append((load_img1_path+filename, load_img2_path+filename, load_groundtruth_path+filename))
    return dataset

def load_test_list(load_img1_path, load_img2_path): #get test list
    filename_list = os.listdir(load_img1_path)   #get file name of every image
                                                 #the file name in each folder should be same
    dataset = []
    for filename in filename_list:
        dataset.append((load_img1_path + filename, load_img2_path + filename))
    return dataset

class Train_data_loader(data.Dataset):
    def __init__(self, img1, img2, groundtruth, \
                 mode ='train', img_height = 350, img_width = 350): #image = 350*350,default mode = train
        self.mode = mode
        self.img_height = img_height
        self.img_width = img_width
        if mode == 'train':
            self.train_dataset = load_train_list(img1, img2, groundtruth)
            self.data_list = self.train_dataset
            print("Total training examples:", len(self.data_list))

    def __getitem__(self, index):
        i1_path, i2_path, gr_path= self.data_list[index]
        i1 = Image.open(i1_path)
        i2 = Image.open(i2_path)
        gr = Image.open(gr_path)


        i1 = i1.resize((self.img_height, self.img_width), Image.ANTIALIAS)  #resize image to target size which U-Net can process
        i2 = i2.resize((self.img_height, self.img_width), Image.ANTIALIAS)
        gr = gr.resize((self.img_height, self.img_width), Image.ANTIALIAS)

        i1 = torch.from_numpy(np.asarray(i1) / 255).float() #transform numpy to tensor
        i2 = torch.from_numpy(np.asarray(i2) / 255).float()
        gr = torch.from_numpy(np.asarray(gr) / 255).float()

        return i1.permute(2, 0, 1), i2.permute(2, 0, 1), gr.permute(2, 0, 1)    #change dimension contents
            
    def __len__(self):
        return len(self.data_list)

class Test_data_loader(data.Dataset):
    def __init__(self, img1, img2, \
                 mode='test', img_height=350, img_width=350):  # image = 224*224,default mode = train
        self.mode = mode
        self.img_height = img_height
        self.img_width = img_width

        if mode == 'test':
            self.test_dataset = load_test_list(img1, img2)
            self.data_list = self.test_dataset
            print("Total testing examples:", len(self.data_list))

    def __getitem__(self, index):
        i1_path, i2_path = self.data_list[index]
        i1 = Image.open(i1_path)
        i2 = Image.open(i2_path)

        # i1 = i1.resize((self.img_height, self.img_width), Image.ANTIALIAS)  # resize image to target size which U-Net can process
        # i2 = i2.resize((self.img_height, self.img_width), Image.ANTIALIAS)

        i1 = torch.from_numpy(np.asarray(i1) / 255).float()  # transform numpy to tensor
        i2 = torch.from_numpy(np.asarray(i2) / 255).float()

        return i1.permute(2, 0, 1), i2.permute(2, 0, 1)# change dimension contents

    def __len__(self):
        return len(self.data_list)