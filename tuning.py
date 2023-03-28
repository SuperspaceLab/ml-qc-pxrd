from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import math
import random
import glob
import datetime
import pytz
today = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
#sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import generator as gen 
from pxrd import calc_QC_peaks
from pxrd import calc_virtualiQC
from pxrd import calc_multiQC
from pxrd import calc_others
import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D ,Dropout, MaxPooling1D, InputLayer
from keras.layers.pooling import GlobalMaxPooling1D
from tensorflow.keras import Model
from keras import regularizers
import optuna
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D

TAU = (1 + np.sqrt(5))/2

dic_wvl = {}
dic_wvl['Cu_Ka'] = 1.54059
dic_wvl['Cu_Kb'] = 1.3810
wvl = dic_wvl['Cu_Ka']

def create_model(num_Clayer, num_Dlayer, dense_units, num_filters, pool_sizes, Cstrides, Pstrides, dropout_rates, filter_sizes):
    model = Sequential()
    model.add(Conv1D(num_filters[0], filter_sizes[0], strides = Cstrides[0], activation = 'relu',padding = 'same', kernel_regularizer = regularizers.l2(0.01)))
    model.add(MaxPooling1D(pool_sizes[0], strides = Pstrides[0], padding = 'valid'))
    for i in range(1, num_Clayer):
        model.add(Conv1D(num_filters[i], filter_sizes[i], strides=Cstrides[i], activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01)))
        model.add(MaxPooling1D(pool_sizes[i], strides=Pstrides[i], padding='valid'))
    model.add(Flatten())
    model.add(Dense(dense_units[0], activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_rates[0]/10))
    for i in range(1, num_Dlayer):
        model.add(Dense(dense_units[i], activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(dropout_rates[i]/10))
    model.add(Dense(2, activation='softmax'))
    return model

if __name__ == '__main__':

    aico = 5.0  # icosahedral lattice constant in Ang
    aico_delta = 0.0
    hklmno_range = 6
    tth_min = 20.0 # in degree
    tth_max = 80.0
    tth_step = 0.01
    data_num_tune_qc     = 30000
    data_num_tune_non_qc = 30000
    path_dataset = '.'
    path_output = './tuning'
    output_file = 'output_'+str(today)[:-13].replace(':', '-')+'.txt'
    
    if os.path.isdir('%s'%(path_output))==False:
        os.mkdir('%s'%(path_output))
      
    num = int((tth_max-tth_min)/tth_step)
        
    # Generate reflections list
    #reflection_list = calc_QC_peaks(hklmno_range, aico, aico_delta, wvl, tth_min, tth_max)
    ref_list = gen.independent_reflection_list(hklmno_range, wvl, aico, tth_max)
    reflection_list = gen.independent_reflection_list_in_tth_range(ref_list, wvl, aico, tth_min, tth_max)

    # data for hyperparameter tuning
    x_data, y_data = gen.dataset(path_dataset, wvl, reflection_list, \
                                    aico, aico_delta, hklmno_range, \
                                    tth_min, tth_max, tth_step, \
                                    data_num_tune_qc, data_num_tune_non_qc)
    x_data = x_data.reshape(data_num_tune_qc + data_num_tune_non_qc,num,1)
    
    def objective(trial):
        print('Optimize Start')
        
        keras.backend.clear_session()
        
        num_Clayer=trial.suggest_int("num_Clayer", 1, 4)
        num_Dlayer=trial.suggest_int("num_Dlayer", 1, 3)
        dense_units = [int(trial.suggest_discrete_uniform("dense_units_"+str(i), 500, 3000, 500)) for i in range(num_Dlayer)]
        num_filters = [int(trial.suggest_discrete_uniform("num_filters_"+str(i), 32, 128, 32)) for i in range(num_Clayer)]
        filter_sizes=[int(trial.suggest_discrete_uniform("filter_sizes_"+str(i), 10, 50, 5)) for i in range(num_Clayer)]
        dropout_rates=[trial.suggest_discrete_uniform('dropout_rates_'+str(i), 1, 5, 1) for i in range(num_Dlayer)]
        pool_sizes=[trial.suggest_int("pool_sizes_"+str(i), 1, 3) for i in range(num_Clayer)]
        Cstrides=[trial.suggest_int("Cstrides_"+str(i), 1, 3) for i in range(num_Clayer)]
        Pstrides=[trial.suggest_int("Pstrides_"+str(i), 1, 3) for i in range(num_Clayer)]

        optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam"])

        model=create_model(num_Clayer, num_Dlayer, dense_units, num_filters, pool_sizes, Cstrides, Pstrides, dropout_rates, filter_sizes)
        model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
                    
        epoch_num=20
        batch_num=256
        history=model.fit(x_data, y_data, epochs=epoch_num, batch_size=batch_num, validation_split=0.2)
        score = 1 - history.history["accuracy"][-1]
    
        print('score', score)
        return score
        
    study = optuna.create_study()
    study.enqueue_trial({'num_Clayer': 3, \
                        'num_Dlayer': 2, \
                        'dense_units': [2000, 1000], \
                        'num_fileters': [64, 64, 64], \
                        'filter_sizes': [20, 20, 20],\
                        'dropout_rates': [0.3, 0.3], \
                        'pool_sizes': [3, 2, 1], \
                        'Cstrides': [1, 1, 2], \
                        'Pstrides': [3, 3, 2], \
                        'optimizer': 'adam'})

    study.optimize(objective, n_trials=50)
    print(study.best_params)
    
    f = open('%s/%s'%(path_output,output_file),'w', encoding="utf-8", errors="ignore")
    f.write(study.best_params)
    f.close()
