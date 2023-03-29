from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import random
import glob
import math
import tqdm
import generator as gen
from pxrd import calc_QC_peaks
from pxrd import calc_virtualiQC
from pxrd import calc_multiQC
from pxrd import calc_others

import tensorflow as tf

path_model = './models/20230322/'
epoch_num = 12 
batch_num = 256

TAU = (1 + np.sqrt(5))/2

dic_wvl = {}
dic_wvl['Cu_Ka'] = 1.54059
dic_wvl['Cu_Kb'] = 1.3810
#wvl = dic_wvl['Cu_Ka']

def generate_test_datasets(QC_peaks, wvl, aico_min, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_qc, data_num_nonqc):
    """
    generates test data for evaluating trained models.
    """
    aico = aico_min
    print('icosahedral attice constant [Å]', aico)
    
    # Multi-iQC dataset
    #QC_peaks = calc_QC_peaks(hklmno_range, aico, aico+aico_delta, wvl, tth_min, tth_max, 1.5)
    virtualQC_test = calc_virtualiQC(data_num_qc, QC_peaks, wvl, aico, aico+aico_delta, tth_min, tth_max, tth_step)
    MultiQC_test = calc_multiQC(virtualQC_test, tth_min, tth_max, tth_step)

    # Non-iQC dataset
    others_test = calc_others(data_num_nonqc, tth_min, tth_max, tth_step)
    
    return MultiQC_test, others_test

def run(aico_min, aico_max, aico_delta, hklmno_range, tth_min, tth_max, tth_step, wvl, data_num_QC, data_num_nonQC, output_flnm):
    """
    run model evaluation uing synthetic datasets 
    """
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
        MultiQC_test, others_test = generate_test_datasets(ref_list, wvl, aico3, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_QC, data_num_nonQC)
        MultiQC_test = MultiQC_test.reshape(data_num_QC, tth_step_num, 1)
        others_test = others_test.reshape(data_num_nonQC, tth_step_num, 1)
        
        # test trained model
        model_name = path_model+str(aico3)+'_'+str(round(aico3+aico_delta, 3))+'__'+str(epoch_num)+'__'+str(batch_num)
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
    
    aico_min = 5.0 # 4.0 # icosahedral lattice constant in Ang.
    aico_max = 5.1 # 6.0
    aico_delta = 0.025
    hklmno_range = 2 # 6
    tth_min  = 20.0 # in degree
    tth_max  = 80.0
    tth_step = 0.01
    wvl = dic_wvl['Cu_Ka']
    data_num_QC    = 3000 # number of QC patterns for each single model
    data_num_nonQC = 3000 # number of non-QCpatterns for each single model
    output_flnm = 'evaluation_result.txt'

    run(aico_min, aico_max, aico_delta, hklmno_range, tth_min, tth_max, tth_step, wvl, data_num_QC, data_num_nonQC, output_flnm)
    
