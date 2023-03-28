from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import random
import keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D ,Dropout, MaxPooling1D ,InputLayer
from keras.layers.pooling import GlobalMaxPooling1D
from keras import regularizers
import glob
import math
#sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import generator as gen
from pxrd import calc_QC_peaks
from pxrd import calc_virtualiQC
from pxrd import calc_multiQC
from pxrd import calc_others

dic_wvl = {}
dic_wvl['Cu_Ka'] = 1.54059
dic_wvl['Cu_Kb'] = 1.3810
wvl = dic_wvl['Cu_Ka']

aico_min = 4.000
aico_max = 6.000
aico_delta = 0.025
hklmno_range = 6
tth_min = 20.0
tth_max = 80.0
tth_step = 0.01
data_num_train_qc     = 30000
data_num_train_non_qc = 30000
data_num_test_qc      = 10000
data_num_test_non_qc  = 10000

path_dataset = '.'
path_model = './models/20230322'

if os.path.isdir('%s'%(path_model))==False:
    os.mkdir('%s'%(path_model))

num_all = int((aico_max-aico_min)/aico_delta)

# Generate reflection list
ref_list = gen.reflection_list(hklmno_range, wvl, aico_max, tth_max)

for i in range(num_all):
    
    aico = aico_min + aico_delta*i
    
    #ref_list = gen.calc_QC_peaks(hklmno_range, aico, aico_delta, wvl, tth_min, tth_max)
    
    # load relfection list and adjust tth.
    ref_list = gen.independent_reflection_list_in_tth_range(ref_list, wvl, aico + aico_delta, tth_min, tth_max)
    
    # load datasets for training and test        
    #x_train=np.load('%s/x_train.npy'%(path_dataset))
    #x_test=np.load('%s/x_test.npy'%(path_dataset))
    #y_train=np.load('%s/y_train.npy'%(path_dataset))
    #y_test=np.load('%s/y_test.npy'%(path_dataset))
    
    # Training data
    x_train, y_train = gen.dataset(path_dataset, wvl, ref_list, \
                                aico, aico_delta, \
                                hklmno_range, tth_min, tth_max, tth_step, \
                                data_num_train_qc, data_num_train_non_qc)
    # Test data
    x_test, y_test = gen.dataset(path_dataset, wvl, ref_list, \
                                aico, aico_delta, \
                                hklmno_range, tth_min, tth_max, tth_step, \
                                data_num_test_qc,  data_num_test_non_qc)
    
    input_=x_train.shape[1]
    tf.keras.backend.set_floatx('float64')
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()

            self.conv1 = Conv1D(128,20, strides=1, activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01))
            self.mp1 = MaxPooling1D(3,  strides=3,padding='valid')
            self.conv2 = Conv1D(128,20, strides=1, activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01))
            self.mp2 = MaxPooling1D(2,  strides=3,padding='valid')
            self.conv3 = Conv1D(128,20,strides=2, activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01))
            self.mp3 = MaxPooling1D(1,  strides=2,padding='valid')
            self.flatten = Flatten()
            self.d1 = Dense(2500, activation='relu',kernel_regularizer=regularizers.l2(0.01))
            self.do1 = Dropout(0.3)
            self.d2 = Dense(1000, activation='relu',kernel_regularizer=regularizers.l2(0.01))
            self.do2 = Dropout(0.3)
            self.ds = Dense(2, activation='softmax')

        def call(self, x):
            x = self.conv1(x)
            print(x.shape)
            x = self.mp1(x)
            print(x.shape)
            x = self.conv2(x)
            print(x.shape)
            x = self.mp2(x)
            print(x.shape)
            x = self.conv3(x)
            print(x.shape)
            x = self.mp3(x)
            x = self.flatten(x)
            print(x.shape)
            x = self.d1(x)
            x = self.do1(x)
            x = self.d2(x)
            x = self.do2(x)
            y = self.ds(x)
            return y

    model = MyModel()
    epoch_num=12
    batch_num=256
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epoch_num, batch_size=batch_num, validation_split=0.2)
    model.evaluate(x_test, y_test, verbose=2)

    model_name=str(round(aico,3))+'_'+str(round(aico+aico_delta,3))+'__'+str(epoch_num)+'__'+str(batch_num)
    model.save('%s/'%(path_model)+model_name)