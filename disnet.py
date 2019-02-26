
# coding: utf-8

# In[1]:


# coding: utf-8

# In[1]:

import numpy as np
#import h5py
#import tensorflow as tf
#import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
#from scipy import signal
from tensorflow.python.keras.layers import Input,Conv2D,Concatenate,Flatten,Dense,LeakyReLU,Dropout, ReLU,Softmax
from tensorflow.python.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.python.keras.models import Model

from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras import backend as K

from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# In[2]:


fn = 2
d_min = 1
d_max = 10
blur_filter_size = 11
blur_range = 10
image_size = 512
epochs = 300
batch_size = 28


# In[3]:


train_data_path = 'vcm_data/train_data_one_im_norm_2set_512_vcm_discrinimator.npy'
#test_data_path = 'test_data.npy'
train_label_path = 'vcm_data/train_label_one_im_norm_2set_512_vcm_discrinimator.npy'
#test_label_path = 'test_label.npy'


# In[4]:


data = np.load(train_data_path)[:10000, :, :, :]
#test_data = np.load(test_data_path)
label = np.load(train_label_path)[:10000, :]


# In[5]:


print(data.shape)
print(label.shape)


# In[10]:


train_data = data
train_label = label


# In[7]:


#to_categorical(train_label[:, 2]).shape


# In[24]:


# In[7]:
input_image1 = Input(shape=(512,512,1), name = "input")
#layer1_1 = Conv2D(4, (8, 8), 8,padding='valid',activation=None,use_bias=False,kernel_initializer = my_init,trainable=True,name='layer1')(input_image1)
layer1_1 = Conv2D(8, (9, 9), 4,padding='same',activation=relu, name="Conv1_1")(input_image1)

layer2_1 = Conv2D(16, (5, 5), 2,padding='same',activation=relu, name="Conv2_1")(layer1_1)

layer3_1 = Conv2D(32, (3, 3), 2,padding='same',activation=relu, name="Conv3_1")(layer2_1)

layer4_1 = Conv2D(32, (3, 3), 2,padding='same',activation=relu, name="Conv4_1")(layer3_1)

flattened = Flatten(name="flat")(layer4_1)
#Dp1 = Dropout(0.5)(flattened)
dense1 = Dense(10, name="d1")(flattened)
#ReLU1 = ReLU(name="lr1")(dense1)

output_position = Dense(1, activation='sigmoid', name="out")(dense1)

print(output_position)

model = Model(inputs=input_image1, outputs=output_position)
model.summary()


print(model.output_shape)


# In[33]:


opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt)

model.fit(train_data.astype('float16')/255.0, train_label[:, 2:], epochs = epochs,batch_size = batch_size,verbose=1, validation_split = 0.2)

model.save('my_model_dis_512.h5')


# In[ ]:


model.save_weights('my_dis_weights_512.h5')


# In[ ]:


# print(to_categorical(train_label[:10, 2:]))


# In[27]:


# pred  = model.predict(train_data[:, :, :, :10])


# In[32]:


# idx = 5
# plt.imshow(train_data[idx, :, :, 0])

# print(train_label[idx, :])
# print(pred[idx, :])

