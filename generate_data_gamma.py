
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.io,scipy.misc
import os
import skimage.transform
import random
import cv2
from scipy import signal
# from PIL import Image
#img = Image.open('image.png').convert('LA')
#img.save('greyscale.png')


# In[2]:


#import os
#import os.path
File_names = []
for dirpath, dirnames, filenames in os.walk("Compression_CC/"):
    for filename in [f for f in filenames if f.endswith(".png")]:
        File_names.append(os.path.join(dirpath, filename))
    


# In[3]:


image_size1 = 256
image_size2 = 256
Num_of_data = 20000
Focus_step = 500


# In[4]:


len(File_names)


# In[5]:


DATA = np.zeros([Num_of_data,image_size1,image_size2,2],np.float32)
Label = np.zeros([Num_of_data,3],np.float32)


# In[6]:


Ground_truth_space = [1150,1250,1350,1400,1500,1650,1750,1800,1950,2000]
Data_space = np.arange(500,2101,50)
#print Data_space


# In[7]:


def Data_aug(im):
    k = 400
    caseidx = random.randint(1,4)
    if caseidx == 1:
        return im[im.shape[0]/2-k:im.shape[0]/2+k,im.shape[1]/2-k:im.shape[1]/2+k]
#     elif caseidx == 2:
#         return np.rot90(im)
    elif caseidx == 2:
        return np.rot90(np.rot90(im[im.shape[0]/2-k:im.shape[0]/2+k,im.shape[1]/2-k:im.shape[1]/2+k]))
#     elif caseidx == 4:
#         return np.rot90(np.rot90(np.rot90(im)))
    elif caseidx == 3:
        return np.flip(im[im.shape[0]/2-k:im.shape[0]/2+k,im.shape[1]/2-k:im.shape[1]/2+k],1)
#     elif caseidx == 6:
#         return np.rot90(np.flip(im,1))
    elif caseidx == 4:
        return np.rot90(np.rot90(np.flip(im[im.shape[0]/2-k:im.shape[0]/2+k,im.shape[1]/2-k:im.shape[1]/2+k],1)))
#     elif caseidx == 8:
#         return np.rot90(np.rot90(np.rot90(np.flip(im,1))))


# In[8]:


def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 0.005
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + abs(gauss)
        return noisy
    elif noise_typ == "poisson":
        image = np.maximum(image,0)
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image*255) / float(255) + image
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    elif noise_typ =="None":
        return image


# In[11]:


def Defocus(input_image, filter_size, scale):
    #dr = random.randint(-30,30)
    #dr2 = random.randint(-30,30)
    #print('input_image_min',input_image.min())
    input_image = np.power(input_image,2.4)
    #print filter_size
    #print 'filters/{}.mat'.format(int(filter_size))
    #print('filters/{}.mat'.format(int(filter_size)))
    a = scipy.io.loadmat('filters/{}.mat'.format(int(filter_size)))
    h = a['h']
    pip = signal.convolve(input_image, h, mode='full', method='auto');
    pip2 = skimage.transform.rescale(pip, scale)
    M = pip2.shape[0]/2
    N = pip2.shape[1]/2
    #plt.imshow(pip2)
    #plt.show()
    crop = np.maximum(0,pip2[M-image_size1/2+dr:M+image_size1/2+dr,N-image_size2/2+dr2:N+image_size2/2+dr2])
    crop_noise = np.power(crop,1/2.4)#noisy("gauss",crop)
    #plt.imshow(crop)
    #plt.show()
    #print(crop_noise.min(),crop_noise.max())
    crop_noise = (crop_noise - crop_noise.min()) / (crop_noise.max()-crop_noise.min())
    return crop_noise


# In[15]:


for i in range(Num_of_data):
    print(i)
    ## Load an image and convert it to grayscale
    idx = random.randint(0,len(File_names)-1)
    img = cv2.imread(File_names[idx],0)
    dr = random.randint(-30,30)
    dr2 = random.randint(-30,30)
    while True:
        if min(img.shape[0]-800,img.shape[1]-800) > 0:
            break
        idx = random.randint(0,len(File_names)-1)
        img = cv2.imread(File_names[idx],0)
    #print(File_names[idx])
    Focused_image = Data_aug(img)
    if Focused_image.max()>1:
        Focused_image = Focused_image/255.0
    #print(Focused_image.min())
    #plt.imshow(Focused_image)
    #plt.show()
    ## Generate the statistics
    
    current_position = random.choice(Ground_truth_space)
    #Data_space = np.arange(current_position-300,min(2101,current_position+301),50)
    current_data_space = list(set(Data_space).difference(set([current_position,current_position-Focus_step,1800,1750,1700,1650,1900,1850,1950,2000,2050,2100])))
    defocus1 = random.choice(current_data_space)
    defocus2 = defocus1 + Focus_step
#     print(defocus2)
    model_mat = scipy.io.loadmat('Calibration_gamma/Model_parameters_{}.mat'.format(current_position))
    Model_parameters = model_mat['Model_parameters'] 
    ## image #1
    DATA[i,:,:,0] = Defocus(Focused_image,Model_parameters[defocus1/50-10,2],Model_parameters[defocus1/50-10,3])
    #print(defocus2/50-10)
    DATA[i,:,:,1] = Defocus(Focused_image,Model_parameters[defocus2/50-10,2],Model_parameters[defocus2/50-10,3])
    
    Label[i,0] = current_position
    Label[i,1] = defocus1
    Label[i,2] = defocus2
#     print(Label[i,:])
    

# In[14]:


np.save('train_data_256_500_norm',DATA)
np.save('train_label_256_500_norm',Label)


