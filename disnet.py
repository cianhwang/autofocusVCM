
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

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


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


train_data_path = 'train_data_uint8_512_vcm_discrinimator.npy'
test_data_path = 'real_data_uint8_512_vcm_10000.npy'
train_label_path = 'train_label_uint8_512_vcm_discrinimator.npy'
test_label_path = 'real_label_uint8_512_vcm_10000.npy'


# In[4]:


train_data = np.load(train_data_path)
test_data = np.load(test_data_path)
train_label = np.load(train_label_path)
test_label = np.load(test_label_path)
test_label = (abs(test_label[:, 0] - test_label[:, 1]) <= 33)


# In[5]:


# train_data = data[:10000, :, :, :]
# train_label = label[:10000, :]


# In[7]:


# #to_categorical(train_label[:, 2]).shape
# import matplotlib.pyplot as plt
# %matplotlib inline
# idx = 7
# plt.imshow(train_data[idx, :, :, 0])

# print(train_label[idx, :])


# In[8]:


# plt.imshow(test_data[idx, :, :, 0])

# print(test_label[idx])


# In[9]:


# In[7]:
input_image1 = Input(shape=(512,512,1), name = "input")
#layer1_1 = Conv2D(4, (8, 8), 8,padding='valid',activation=None,use_bias=False,kernel_initializer = my_init,trainable=True,name='layer1')(input_image1)
layer1_1 = Conv2D(1, (8, 8), 8,padding='same',activation=relu, name="Conv1_1")(input_image1)

layer2_1 = Conv2D(1, (8, 8), 8,padding='same',activation=relu, name="Conv2_1")(layer1_1)

# layer3_1 = Conv2D(32, (3, 3), 2,padding='same',activation=relu, name="Conv3_1")(layer2_1)

# layer4_1 = Conv2D(32, (3, 3), 2,padding='same',activation=relu, name="Conv4_1")(layer3_1)

flattened = Flatten(name="flat")(layer2_1)

dense1 = Dense(10, name="d1")(flattened)
Dp1 = Dropout(0.5)(dense1)
ReLU1 = ReLU(name="lr1")(Dp1)

output_position = Dense(2, activation='softmax', name="out")(ReLU1)

print(output_position)

model = Model(inputs=input_image1, outputs=output_position)
model.summary()

tcbc = TensorBoard(log_dir='1')

filepath="models/dis_train_on_generate/{epoch:03d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max',period=2)


print(model.output_shape)


# In[10]:


def data_gen(features, labels, batch_size):
    while True:
        for i in np.arange(0, features.shape[0] - batch_size, batch_size):
            # choose random index in features
            batch_features = features[i:i+batch_size, :, :, :].astype('float16')/255.0
            batch_labels = labels[i:i+batch_size, :]
            yield (batch_features, batch_labels)


# In[11]:


opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt)

# model.fit(train_data[:,:,:,0:1].astype('float16')/255.0, to_categorical(train_label[:, 2:]), 
#           epochs = 100,batch_size = batch_size,
#           validation_data=(test_data[:,:,:,0:1].astype('float16')/255.0, to_categorical(test_label)), 
#           verbose=1, callbacks = [tcbc, checkpoint])
model.fit_generator(data_gen(train_data, to_categorical(train_label[:, 2:]), batch_size), 
                    steps_per_epoch = train_data.shape[0]/batch_size, epochs = epochs,
                    validation_data=data_gen(test_data, to_categorical(test_label), batch_size), 
                    validation_steps = batch_size,
                    verbose=1, callbacks = [tcbc, checkpoint])


# In[ ]:


model.save('my_model_dis_512.h5')

model.save_weights('my_dis_weights_512.h5')
# # model.load_weights('my_dis_weights_512.h5')


# In[ ]:


# print(to_categorical(train_label[:10, 2:]))


# In[ ]:


# pred  = model.predict(train_data[:, :, :, :50])


# In[ ]:


# idx = 25
# plt.imshow(train_data[idx, :, :, 0])

# print(train_label[idx, :])
# print(pred[idx, :])

