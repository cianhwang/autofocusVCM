
# coding: utf-8

# In[1]:

import numpy as np
#import h5py
#import tensorflow as tf
#import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
#from scipy import signal
from tensorflow.python.keras.layers import Input,Conv2D,Concatenate,Flatten,Dense,LeakyReLU,Dropout
from tensorflow.python.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.python.keras.models import Model


from tensorflow.python.keras import backend as K

from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.optimizers import Adam

#from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.python.keras.models import Sequential


# In[2]:


fn = 2
d_min = 1
d_max = 10
blur_filter_size = 11
blur_range = 10
image_size = 128
epochs = 300
batch_size = 28


# In[3]:


train_data_path = 'train_data_one_im_norm_2set_512.npy'
#test_data_path = 'test_data.npy'
train_label_path = 'train_label_one_im_norm_2set_512.npy'
#test_label_path = 'test_label.npy'


# In[4]:


train_data = np.load(train_data_path)
#test_data = np.load(test_data_path)
train_label = np.load(train_label_path)
#test_label = np.load(test_label_path) #,mmap_mode='r'

test_data = np.load('real_data_1_im_norm_512.npy')
test_label = np.load('real_data_label_1_im_norm_512.npy')

f_code=h5py.File('code_884_1x16_mix_4bit.h5','r')
code=f_code['data'][:]
code=code.astype(np.float32)
# In[4]:
'''

train_data = np.load(train_data_path)
#test_data = np.load(test_data_path)
train_label = np.load(train_label_path)
#test_label = np.load(test_label_path) #,mmap_mode='r'


a = np.load('train_data_500_2100_100.npy',mmap_mode='r')
b = np.load('train_data_1_500_2100_100.npy',mmap_mode='r')
train_data = np.concatenate((a,b[0:2000,:,:,:]),axis=0)

c = np.load('train_label_500_2100_100.npy',mmap_mode='r')
d = np.load('train_label_1_500_2100_100.npy',mmap_mode='r')
train_label = np.concatenate((c,d[0:2000,:]),axis=0)


test_data = np.load('../synthetic_data/real_data_norm_500.npy')
test_label = np.load('../synthetic_data/real_data_label_norm_500.npy')
'''

# In[5]:
def my_init(shape, dtype=tf.float32,partition_info=None):
    return K.variable(value=code, dtype=dtype)

# In[5]:


print(test_data.shape)


# In[6]:
'''

input_image1 = Input(shape=(256,256,1))
layer1_1 = (Conv2D(16, (8, 8), 8,padding='valid',activation=relu))(input_image1)
layer2_1 = (Conv2D(32, (4, 4), 4,padding='valid',activation=relu))(layer1_1)

input_image2 = Input(shape=(256,256,1))
layer1_2 = (Conv2D(16, (8, 8), 8,padding='valid',activation=relu))(input_image2)
layer2_2 = (Conv2D(32, (4, 4), 4,padding='valid',activation=relu))(layer1_2)

layer2 = Concatenate(3)([layer2_1,layer2_2])

flattened = Flatten()(layer2)
dense1 = Dense(128,activation=relu)(flattened)
dense2 = Dense(512,activation=relu)(dense1)
dense3 = Dense(10,activation=relu)(dense2)
output_position = Dense(1)(dense3)

model = Model(inputs=[input_image1,input_image2], outputs=output_position)
tcbc = TensorBoard(log_dir='1')
'''

# In[7]:
input_image1 = Input(shape=(512,512,1))
#layer1_1 = Conv2D(4, (8, 8), 8,padding='valid',activation=None,use_bias=False,kernel_initializer = my_init,trainable=True,name='layer1')(input_image1)
layer1_1 = Conv2D(4, (8, 8), 8,padding='valid',activation=relu)(input_image1)
#layer1_1 = LeakyReLU(0.1)(layer1_1)

layer2_1 = Conv2D(8, (4, 4), 4,padding='valid',activation=relu)(layer1_1)
#layer2_1 = LeakyReLU(0.1)(layer2_1)

layer3_1 = Conv2D(8, (4, 4), 4,padding='valid',activation=relu)(layer2_1)
#layer3_1 = LeakyReLU(0.1)(layer3_1)


# input_image2 = Input(shape=(256,256,1))
# layer1_2 = (Conv2D(16, (8, 8), 8,padding='valid',activation=relu))(input_image2)
# layer2_2 = (Conv2D(32, (4, 4), 4,padding='valid',activation=relu))(layer1_2)

# layer2 = Concatenate(3)([layer2_1,layer2_2])

flattened = Flatten()(layer3_1)
dense1 = Dense(1024)(flattened)
dense1 = LeakyReLU(0.1)(dense1)
#dense1 = Dropout(0.8)(dense1)

dense2 = Dense(512)(dense1)
dense2 = LeakyReLU(0.1)(dense2)
#dense2 = Dropout(0.5)(dense2)

dense3 = Dense(10)(dense2)
dense3 = LeakyReLU(0.1)(dense3)

output_position = Dense(1)(dense3)
print(output_position)

model = Model(inputs=input_image1, outputs=output_position)
tcbc = TensorBoard(log_dir='1')

filepath="one_im_models1/weights-improvement-{epoch:03d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max',period=2)


# In[8]:


print(model.output_shape)


# In[9]:


# In[7]:

'''
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
model.compile(loss='mse', optimizer=opt)

#model.fit(train_data[0:18000,:,:,:], (train_label[0:18000,1:2]-train_label[0:18000,0:1])/100, epochs = epochs,batch_size = batch_size, validation_data=(train_data[18000:20000,:,:,:], (train_label[18000:20000,1:2]-train_label[18000:20000,0:1])/100),verbose=1)
model.fit(train_data, (train_label[:,1:2]-train_label[:,0:1])/100, epochs = epochs,batch_size = batch_size, 
          validation_data=(test_data[:,:,:,0:1], (test_label[:,1:2]-test_label[:,0:1])/100.0),verbose=1, callbacks=[tcbc])
'''

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001)
model.compile(loss='mse', optimizer=opt)

#model.fit(train_data[0:18000,:,:,:], (train_label[0:18000,1:2]-train_label[0:18000,0:1])/100, epochs = epochs,batch_size = batch_size, validation_data=(train_data[18000:20000,:,:,:], (train_label[18000:20000,1:2]-train_label[18000:20000,0:1])/100),verbose=1)
model.fit(train_data[:,:,:,0:1], abs(train_label[:,1:2]-train_label[:,0:1])/100, epochs = epochs,batch_size = batch_size, 
          validation_data=(test_data[:,:,:,0:1], abs(test_label[:,1:2]-test_label[:,0:1])/100),verbose=1, callbacks=[tcbc,checkpoint])

##############


# In[ ]:


##############
#model.fit(train_data[0:22000,:,:,:], train_label[0:22000,:], epochs = epochs,batch_size = batch_size,
#          validation_data=(train_data[22000:24000,:,:,:], train_label[22000:24000,:]),verbose=1)
#model.evaluate(test_data, test_label)

#model.fit(test_data[0:290,:,:,:], (test_label[0:290,1:2]-test_label[0:290,0:1])/100, epochs = epochs,batch_size = batch_size, validation_data=(test_data[290:300,:,:,:], (test_label[290:300,1:2]-test_label[290:300,0:1])/100),verbose=1)




model.save('my_model_step1.h5')


# In[5]:
'''

# In[7]:


model = Sequential()
model.add(Conv2D(16,(8, 8), 8,padding='valid',input_shape=(256,256,1)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)


model.add(Conv2D(32, (4, 4), 4,padding='valid'))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)

# # model.add(Conv2D(256, (7, 7), 1,padding='valid'))
# # model.add(Activation('relu'))
# # print(model.output_shape)

# model.add(Conv2D(64, (2, 2), 2,padding='valid'))
# model.add(Activation('relu'))
# print(model.output_shape)

# model.add(Conv2D(16, (1, 1), 1,padding='valid'))
# model.add(Activation('relu'))
# print(model.output_shape)


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))

#model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
tcbc = TensorBoard(log_dir='1')
'''
