
# coding: utf-8

# In[8]:


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
#from tensorflow.python.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.python.keras.models import Model

from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras import backend as K

from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


fn = 2
d_min = 1
d_max = 10
blur_filter_size = 11
blur_range = 10
image_size = 256
epochs = 200
batch_size = 28


# In[3]:


train_data_path = 'vcm_data/train_data_one_im_norm_2set_256_vcm_discrinimator.npy'
#test_data_path = 'test_data.npy'
train_label_path = 'vcm_data/train_label_one_im_norm_2set_256_vcm_discrinimator.npy'
#test_label_path = 'test_label.npy'

print("data loading...")
# In[4]:


train_data = np.load(train_data_path)
#test_data = np.load(test_data_path)
train_label = np.load(train_label_path)


# In[5]:


print(train_data.shape)
print(train_label.shape)


# In[17]:


# In[7]:
input_image1 = Input(shape=(256,256,1), name = "input")
#layer1_1 = Conv2D(4, (8, 8), 8,padding='valid',activation=None,use_bias=False,kernel_initializer = my_init,trainable=True,name='layer1')(input_image1)
layer1_1 = Conv2D(1, (8, 8), 4,padding='valid',activation=relu, name="Conv1_1")(input_image1)

layer2_1 = Conv2D(1, (8, 8), 4,padding='valid',activation=relu, name="Conv2_1")(layer1_1)

flattened = Flatten(name="flat")(layer2_1)
dense1 = Dense(10, name="d1")(flattened)
ReLU1 = ReLU(name="lr1")(dense1)
Dp1 = Dropout(0.5, name = "dp1")(ReLU1)

output_position = Dense(2, activation='softmax', name="out")(Dp1)

print(output_position)

model = Model(inputs=input_image1, outputs=output_position)
model.summary()


print(model.output_shape)


# In[18]:


opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001)
model.compile(loss='mse', optimizer='Adam')

model.fit(train_data[:,:,:,:1], to_categorical(train_label[:, 2:]), epochs = epochs,batch_size = batch_size,verbose=1)

model.save('my_model_dis.h5')
model.save_weights('my_model_weights.h5')


# In[19]:


# print(to_categorical(train_label[:10, 2:]))


# In[20]:


# pred  = model.predict(train_data[:, :, :, :10])


# In[31]:


# idx = 2
# plt.imshow(train_data[idx, :, :, 0])

# print(train_label[idx, :])
# print(pred[idx, :])
