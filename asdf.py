import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

print(torch.version)
torch.cuda.is_available()
########################################################

table = pd.read_csv('/project_space/ADNI_ROY/scripts/adni_table_all.csv')
print(table.head())
print('-'15, 'the total images are', len(table),'-'15)
print(table['Path'][0])
###########################################################
class NeuroImagingDataset(Dataset):
    def init(self, data_df, transform = None):
        self.data_df = data_df
        self.transform = transform

    def len(self):
        return len(self.data_df)

    def getitem(self, idx):

        img_path = self.data_df.iloc[idx]['Path']
        label = self.data_df.iloc[idx]['DX_original']
        if label == 'CN':
            label=torch.tensor([1, 0, 0])
        elif label == 'MCI':
            label=torch.tensor([0, 0, 1])
        elif label == 'Dementia':
            label=torch.tensor([0, 1, 0])

#         print(label)
#         print(type(label))
        img = nib.load(img_path) 
        #print(img.shape)
        # get the image data as a numpy array
        img = img.get_fdata()

        # convert the image to a tensor and add channel dimension

        #print(img)
        img =torch.from_numpy(img).unsqueeze(0)


        if self.transform is not None:
            img = self.transform(img)
        def image_shape(self):
            return utils.load_nifti(self.img_path[0]).shape

        return img, label
    
#######################################################    
    #patient-wise train-test-split
pt_CN = table[table['DX_original'] == 'CN']['PTID'].unique()
pt_MCI = table[table['DX_original'] == 'MCI']['PTID'].unique()
pt_Dementia = table[table['DX_original'] == 'Dementia']['PTID'].unique()

print(f" the total number of patients is {len(table['PTID'].unique())}")
print(f" the total number of patients' diagnoses is {len(pt_CN) + len(pt_MCI) +  len(pt_Dementia)} because one patient has multiple diagnoses")
print(f' - CN: {len(pt_CN)}')
print(f' - MCI: {len(pt_MCI)}')
print(f' - Dementia: {len(pt_Dementia)}')

# Use the train_test_split function twice to get 8(train):1(validation):1(test) ratio ##?? should we do filter
#1
pt_CN_train_val, pt_CN_test = train_test_split(pt_CN, test_size = 0.1, random_state = 0, shuffle = False)
pt_MCI_train_val, pt_MCI_test = train_test_split(pt_MCI, test_size = 0.1, random_state = 0, shuffle = False)
pt_Dementia_train_val, pt_Dementia_test = train_test_split(pt_Dementia, test_size = 0.1, random_state = 0, shuffle = False)

#2
pt_CN_train, pt_CN_val = train_test_split(pt_CN_train_val, test_size = 1/9, random_state = 0, shuffle = False)
pt_MCI_train, pt_MCI_val = train_test_split(pt_MCI_train_val, test_size = 1/9, random_state = 0, shuffle = False)
pt_Dementia_train, pt_Dementia_val = train_test_split(pt_Dementia_train_val, test_size = 1/9,random_state = 0, shuffle = False)

#################################################################
# combine train, validation and test sets
pt_train = np.concatenate([pt_CN_train, pt_MCI_train, pt_Dementia_train])
pt_validation = np.concatenate([pt_CN_val, pt_MCI_val, pt_Dementia_val])
pt_test = np.concatenate([pt_CN_test, pt_MCI_test, pt_Dementia_test])
#print(pt_train)
#print(pt_test)

# Because one patient has multiple images
train_df = table[table['PTID'].isin(pt_train)]
val_df = table[table['PTID'].isin(pt_validation)]
test_df = table[table['PTID'].isin(pt_test)]

def has_common_element(list1, list2):
    for element in list1:
        if element in list2:
            print(element)

has_common_element(pt_train, pt_validation)
print('-'50)
has_common_element(pt_train, pt_test)
print('-'50)
has_common_element(pt_validation, pt_test)
##################################################
# combine train, validation and test sets
pt_train = np.concatenate([pt_CN_train, pt_MCI_train, pt_Dementia_train])
pt_validation = np.concatenate([pt_CN_val, pt_MCI_val, pt_Dementia_val])
pt_test = np.concatenate([pt_CN_test, pt_MCI_test, pt_Dementia_test])
#print(pt_train)
#print(pt_test)

# Because one patient has multiple images
train_df = table[table['PTID'].isin(pt_train)]
val_df = table[table['PTID'].isin(pt_validation)]
test_df = table[table['PTID'].isin(pt_test)]

def has_common_element(list1, list2):
    for element in list1:
        if element in list2:
            print(element)

has_common_element(pt_train, pt_validation)
print('-'50)
has_common_element(pt_train, pt_test)
print('-'50)
has_common_element(pt_validation, pt_test)
train_dataset = NeuroImagingDataset(data_df = train_df)
validation_dataset = NeuroImagingDataset(data_df = val_df)
test_dataset = NeuroImagingDataset(data_df = test_df)
train_dataset
##################################
# Define the data loaders for each set
train_loader = DataLoader(train_dataset, batch_size = 32, drop_last = True) # 4832
val_loader = DataLoader(validation_dataset, batch_size = 32, drop_last = True) # 632
test_loader = DataLoader(test_dataset, batch_size  = 32, drop_last = True) # 6*32 # drop the last batch - drop_last =true
print("Train loader size:", len(train_loader))
print("Val loader size:", len(val_loader))
print("Test loader size:", len(test_loader))
######################################################
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MyCNN(nn.Module):
    def init(self):
        super(MyCNN, self).init()
        self.conv1 = nn.Conv3d(1,32,1) #kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv3d(32,64,1)#kernel_size = 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(64242424,128)
        self.fc2 = nn.Linear(128,3)
        #output vector sized

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64242424)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        #softmaxit will reutrn normalized every class oupt the probability of every deimesion added upto 1 
        return x
###########################################
num_epochs = 3
loss_values = [] #initialize an emptly list to store loss values

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(train_loader,0):

        inputs, labels = data
        optimizer.zero_grad()

        inputs_x = inputs.float()
        inputs_x = torch.mean(inputs_x, dim=1, keepdim=True)
        b, c, d, h, w = inputs_x.shape
        inputs_x = inputs_x.permute(0,1,2,3,4)
        outputs = model(inputs_x)
        print('outputs', outputs,outputs.shape)
        labels = labels.float()
        b,c,d,h,w = inputs_x.shape
        print('labels.shape:',labels.shape)
        print('labels:',labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch +1, running_loss / len(train_loader)))

    loss_values.append(running_loss / len(train_loader)) # append the loss value to the list














