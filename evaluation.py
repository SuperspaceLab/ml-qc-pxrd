from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import random
import glob
import math
import tqdm
import generator as gen
import tensorflow as tf

dic_wvl = {}
dic_wvl['Cu_Ka'] = 1.54059

def run(path_model, epoch_num, batch_num, aico_min, aico_max, aico_delta, hklmno_range, tth_min, tth_max, tth_step, wvl, data_num_QC, data_num_nonQC, output_flnm):
    """
    run model evaluation uing synthetic datasets 
    """
    
    # Checking GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    aico_num = int((aico_max - aico_min)/aico_delta)
    tth_step_num = int((tth_max - tth_min)/tth_step)
    
    for num in range(aico_num):
        Result_QC = {aico_min: np.array([0]*data_num_QC) for num in range(aico_num)}
        Result_others = {aico_min: np.array([0]*data_num_nonQC) for num in range(aico_num)}   
    
    ref_list = gen.independent_reflection_list(hklmno_range, wvl, aico_max, tth_max) # generate QC ref list
    for num in range(aico_num):
        aico = aico_min + num*aico_delta
        #aico3 = round(aico_min + num*aico_delta, 3)
        aico3 = round(aico, 3)
        
        # Generating test data
        ref_list = gen.independent_reflection_list_in_tth_range(ref_list, wvl, aico3, tth_min, tth_max)
        MultiQC_test, others_test = gen.dataset(wvl, ref_list, aico3, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_QC, data_num_nonQC)
        MultiQC_test = MultiQC_test.reshape(data_num_QC, tth_step_num, 1)
        others_test = others_test.reshape(data_num_nonQC, tth_step_num, 1)
        
        # test trained model
        model_name = path_model+'/'++str(aico3)+'_'+str(round(aico3+aico_delta, 3))+'__'+str(epoch_num)+'__'+str(batch_num)
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_name, compile = False)

        Result_QC[aico3] = np.round(model.predict(MultiQC_test)[:, 1]).astype(int)
        Result_others[aico3] = np.round(model.predict(others_test)[:, 1]).astype(int)

    f = open('%s'%(output_flnm), 'w')

    print('    TP  FN  FP  TN', file = f)
    TP_sum = 0
    FN_sum = 0
    FP_sum = 0
    TN_sum = 0
    for i in Result_QC.keys():
        TP = np.count_nonzero(Result_QC[i])
        FN = data_num_QC - TP
        FP = np.count_nonzero(Result_others[i])
        TN = data_num_nonQC - FP
        print('{:<.3f}'.format(i), '{:<5d}'.format(TP), '{:<5d}'.format(FN),'{:<5d}'.format(FP),'{:<5d}'.format(TN), file = f)
        TP_sum += TP
        FN_sum += FN
        FP_sum += FP
        TN_sum += TN
    print('\n', file = f)
    print('ALL   TP  FN  FP  TN', file = f)
    print('{:<5d}'.format(TP_sum), '{:<5d}'.format(FN_sum),'{:<5d}'.format(FP_sum),'{:<5d}'.format(TN_sum), file = f)
    f.close()
    return 0
    
if __name__ == '__main__':
    
    path_model = './models'
    epoch_num = 12 
    batch_num = 256
    
    aico_min = 4.0 # icosahedral lattice constant in Ang.
    aico_max = 6.0
    aico_delta = 0.025
    hklmno_range = 2 # 6
    tth_min  = 20.0 # in degree
    tth_max  = 80.0
    tth_step = 0.01
    wvl = dic_wvl['Cu_Ka']
    data_num_QC    = 3000 # number of QC patterns for each single model
    data_num_nonQC = 3000 # number of non-QCpatterns for each single model
    output_flnm = 'evaluation_result.txt'

    run(path_model, epoch_num, batch_num, aico_min, aico_max, aico_delta, hklmno_range, tth_min, tth_max, tth_step, wvl, data_num_QC, data_num_nonQC, output_flnm)
    
