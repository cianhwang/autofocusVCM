
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
from tensorflow.python.keras.layers import Input,Conv2D,Concatenate,Flatten,Dense,LeakyReLU,Dropout, ReLU
from tensorflow.python.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.python.keras.models import Model


from tensorflow.python.keras import backend as K

from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.optimizers import Adam
# from tensorflow.python.keras.callbacks import TensorBoard,ModelCheckpoint

from sklearn.model_selection import train_test_split
import cv2

#from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.python.keras.models import Sequential
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))


# In[2]:


fn = 2
d_min = 1
d_max = 10
blur_filter_size = 11
blur_range = 10
image_size = 512
epochs = 300
batch_size = 32


# In[3]:


# train_data_path = 'train_data_uint8_512_10000_vcm_alpha.npy'
# test_data_path = 'real_data_uint8_512_vcm_10000_alpha.npy'
# train_label_path = 'train_label_uint8_512_10000_vcm_alpha.npy'
# test_label_path = 'real_label_uint8_512_vcm_10000_alpha.npy'

train_data_path = 'train_data_20000_raw.npy'
# test_data_path = 'real_data_uint8_512_vcm_10000_alpha.npy'
train_label_path = 'train_label_20000_raw.npy'
# test_label_path = 'real_label_uint8_512_vcm_10000_alpha.npy'


# In[4]:


train_data = np.load(train_data_path)
# test_data = np.load(test_data_path)
train_label = np.load(train_label_path)
# test_label = np.load(test_label_path)


# In[5]:


print(train_data.shape)
print(train_label.shape)
# test_data = test_data[1::2, :, :, :]
# test_label = test_label[1::2, :]


# In[6]:


# def pre_normalize(im):
#     # https://www.learnopencv.com/image-quality-assessment-brisque/
#     blurred = cv2.GaussianBlur(im, (7, 7), 1.166) # apply gaussian blur to the image
#     blurred_sq = blurred * blurred
#     sigma = cv2.GaussianBlur(im * im, (7, 7), 1.166) 
#     sigma = (sigma - blurred_sq) ** 0.5
#     sigma = sigma + 1.0/255 # to make sure the denominator doesn't give DivideByZero Exception
#     structdis = (im - blurred)/sigma # final MSCN(i, j) image
#     t = im - blurred
#     t = (t-t.min())/(t.max() - t.min())
#     return t

# for iii in range(train_data.shape[0]):
#     train_data[iii, :, :, 0] = pre_normalize(train_data[iii, :, :, 0].astype(np.float32))


# In[6]:


# import matplotlib.pyplot as plt
# %matplotlib inline
# for idx in range(10):
#     plt.imshow(train_data[idx, :, :, 0].astype(np.float32))
#     plt.show()
#     print(train_label[idx, :])
# print(train_data.dtype)
# # batch_features = np.zeros((image_size//2, image_size//2, 4))
# # batch_features[:, :, 0] = train_data[idx, 1::2, ::2, 0]
# # batch_features[:, :, 1] = train_data[idx, ::2, ::2, 0]
# # batch_features[:, :, 2] = train_data[idx, 1::2, 1::2, 0]
# # batch_features[:, :, 3] = train_data[idx, ::2, 1::2, 0]
# # plt.imshow(batch_features[:, :, 3]*16383)


# In[8]:


# plt.imshow(test_data[idx, :, :, 0])
# print(test_label[idx, :])


# In[7]:


# In[7]:
input_image1 = Input(shape=(512, 512, 1), name = "input")

layer1_1 = Conv2D(4, (8, 8), 8,padding='valid',activation=relu, name="Conv1_1")(input_image1)
#layer1_1 = LeakyReLU(0.1)(layer1_1)

layer2_1 = Conv2D(8, (4, 4), 4,padding='valid',activation=relu, name="Conv2_1")(layer1_1)
#layer2_1 = LeakyReLU(0.1)(layer2_1)

layer3_1 = Conv2D(16, (4, 4), 4,padding='valid',activation=relu, name="Conv3_1")(layer2_1)

flattened = Flatten(name="flat")(layer3_1)
# dense1 = Dense(1024, name="d1")(flattened)
# ReLU1 = LeakyReLU(0.1, name="lr1")(dense1)
# dp1 = Dropout(0.5)(ReLU1)

dense2 = Dense(256, name="d2")(flattened)
ReLU2 = LeakyReLU(0.1, name="lr2")(dense2)
dp2 = Dropout(0.5)(ReLU2)

dense3 = Dense(64, name="d3")(dp2)
ReLU3 = LeakyReLU(0.1, name="lr3")(dense3)
dp3 = Dropout(0.5)(ReLU3)

output_position = Dense(1, name="out")(dp3)
print(output_position)

model = Model(inputs=input_image1, outputs=output_position)
model.summary()

tcbc = TensorBoard(log_dir='1')

filepath="raw_models/gen/n_{epoch:03d}-{val_loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max',period=2)


# In[8]:


def data_gen(features, labels, batch_size):
    while True:
        for i in np.arange(0, features.shape[0] - batch_size, batch_size):
            # choose random index in features
            ################!!!
#             batch_features = np.zeros((batch_size, image_size//2, image_size//2, 4))
#             batch_features[:, :, :, 0] = features[i:i+batch_size, 1::2, ::2, 0]
#             batch_features[:, :, :, 1] = features[i:i+batch_size, ::2, ::2, 0]
#             batch_features[:, :, :, 2] = features[i:i+batch_size, 1::2, 1::2, 0]
#             batch_features[:, :, :, 3] = features[i:i+batch_size, ::2, 1::2, 0]
            batch_features = np.zeros((batch_size, image_size, image_size, 1))
            batch_features = features[i:i+batch_size, :, :, :]
            batch_features = batch_features.astype('float16')#/255###########################
            batch_labels = abs(labels[i:i+batch_size, 1:2] - labels[i:i+batch_size, :1])*100
            yield (batch_features, batch_labels)


# In[ ]:


opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001)
model.compile(loss='mse', optimizer='Adam')

X_train, X_test, y_train, y_test = train_test_split(
    train_data, train_label, test_size=0.2, random_state=233)

# model.fit(train_data[:,:,:,0:1].astype('float16')/255.0, abs(train_label[:,1:2]-train_label[:,0:1])/100, 
#           epochs = 100,batch_size = batch_size, #validation_split=0.2,
#           validation_data=(test_data[:,:,:,0:1].astype('float16')/255.0, abs(test_label[:,1:2]-test_label[:,0:1])/100), 
#           verbose=1, callbacks = [tcbc, checkpoint])
model.fit_generator(data_gen(X_train, y_train, batch_size), 
                    steps_per_epoch = X_train.shape[0]/batch_size, epochs = epochs,
                    validation_data=data_gen(X_test, y_test, batch_size), 
                    validation_steps = batch_size,verbose=1)#, callbacks = [tcbc, checkpoint])


# In[12]:


# model.save('my_model_512.h5')


# In[13]:


# print(train_label[:, :10])


# In[14]:


# from tensorflow.python.keras.models import load_model
# model = load_model('models/050-0.65.hdf5')


# In[15]:


# import matplotlib.pyplot as plt
# %matplotlib inline
# idxx = 89
# plt.imshow(test_data[idxx, :, :, 0])
# print(test_label[idxx, :])


# In[16]:


# model.predict(test_data[idxx:idxx+1, :, :, :1]/255.0)


# In[17]:


# import matplotlib.pyplot as plt
# %matplotlib inline
# for i in range(2000//64-1): #
#     batch_features = np.zeros((64, image_size//2, image_size//2, 4))
#     batch_features[:, :, :, 0] = X_test[64*i:64*i+64, 1::2, ::2, 0]
#     batch_features[:, :, :, 1] = X_test[64*i:64*i+64, ::2, ::2, 0]
#     batch_features[:, :, :, 2] = X_test[64*i:64*i+64, 1::2, 1::2, 0]
#     batch_features[:, :, :, 3] = X_test[64*i:64*i+64, ::2, 1::2, 0]
#     test_pred = model.predict(batch_features)
#     test_pred[test_pred<0]= 0
#     plt.scatter(abs(y_test[64*i:64*i+64, 0] - y_test[64*i:64*i+64, 1]), test_pred[:, 0]/100 - abs(y_test[64*i:64*i+64, 0] - y_test[64*i:64*i+64, 1]), color='black', s = 0.1)
# plt.show()
# # for i in range(8000//64-1): #
# #     train_pred = model.predict(train_data[64*i:64*i+64, :, :, :1]/255.0)
# #     train_pred[train_pred<0]= 0
# #     plt.scatter(abs(train_label[64*i:64*i+64, 0] - train_label[64*i:64*i+64, 1]), train_pred[:, 0]/100 - abs(train_label[64*i:64*i+64, 0] - train_label[64*i:64*i+64, 1]), color='black', s = 0.1)
# # plt.show()


# In[ ]:


# test_pred[test_pred<0]= 0
# print(test_pred[:, 0])


# In[ ]:


# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.imshow(test_data[0, :, :, 0])
# print(test_label)


# In[ ]:


# plt.scatter(abs(test_label[:, 0] - test_label[:, 1]), test_pred[:, 0])

