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
wvl = dic_wvl['Cu_Ka']

def generate_test_datasets(QC_peaks, wvl, aico_min, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_qc, data_num_nonqc, path_testdata):

    aico = aico_min
    print('Lattice constant [Å]', aico)
    
    # Multi-iQC dataset
    #QC_peaks = calc_QC_peaks(hklmno_range, aico, aico+aico_delta, wvl, tth_min, tth_max, 1.5)
    virtualQC_test = calc_virtualiQC(data_num_qc, QC_peaks, wvl, aico, aico+aico_delta, tth_min, tth_max, tth_step)
    MultiQC_test = calc_multiQC(virtualQC_test, tth_min, tth_max, tth_step)


    # Non-iQC dataset
    others_test = calc_others(data_num_nonqc, tth_min, tth_max, tth_step)

    tf.keras.backend.set_floatx('float64')
    np.save('%s/QC/'%(path_testdata)+str(aico_min)+'_testdata.npy',  MultiQC_test[..., tf.newaxis])
    np.save('%s/Others/'%(path_testdata)+str(aico_min)+'_testdata.npy',  others_test[..., tf.newaxis])
    
    return 0

if __name__ == '__main__':
    
    aico_min__ = 4.0
    aico_max__ = 6.0
    aico_delta = 0.025
    hklmno_range = 6
    tth_min  = 20.0
    tth_max  = 80.0
    tth_step = 0.01
    wvl = dic_wvl['Cu_Ka']
    data_num_QC    = 3000 # number of QC patterns for each single model
    data_num_nonQC = 3000 # number of non-QCpatterns for each single model
    output_flnm = 'evaluation_result.txt'
    path_testdata = './testdata'   

    if os.path.isdir('%s'%(path_testdata))==False:
        os.mkdir('%s/QC'%(path_testdata))
        os.mkdir('%s/Others'%(path_testdata))
        
    aico_num = int((aico_max__-aico_min__)/aico_delta)
    ref_list = gen.independent_reflection_list(hklmno_range, wvl, aico_max__, tth_max) # generate QC ref list
    for num in range(aico_num):
        aico = aico_min__+num*aico_delta
        aico_min = round(aico_min__+num*aico_delta, 3)
        ref_list = gen.independent_reflection_list_in_tth_range(ref_list, wvl, aico_min, tth_min, tth_max)
        generate_test_datasets(ref_list, wvl, aico_min, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_QC, data_num_nonQC, path_testdata)
   
    Result_QC = {round(aico_min__+num*aico_delta, 3): np.array([0]*data_num_test) for num in range(aico_num)}
    Result_others = {round(aico_min__+num*aico_delta, 3): np.array([0]*data_num_test) for num in range(aico_num)}
    screening = 0
    if screening == 0:
        for model_num in tqdm.tqdm(range(aico_num), desc = 'Progress'):
            model_aico = round(aico_min__+model_num*aico_delta, 3)
            model_name = path_model+str(model_aico)+'_'+str(round(model_aico+aico_delta, 3))+'__'+str(epoch_num)+'__'+str(batch_num)
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(model_name, compile = False)
            testdata_QC = np.load('%s/QC/'%(path_testdata)+str(model_aico)+'_testdata.npy')
            Result_QC[model_aico] = np.round(model.predict(testdata_QC)[:, 1]).astype(int)
            testdata_others = np.load('%s/Others/'%(path_testdata)+str(model_aico)+'_testdata.npy')
            Result_others[model_aico] = np.round(model.predict(testdata_others)[:, 1]).astype(int)
    else:
        for model_num in tqdm.tqdm(range(aico_num), desc = 'Progress'):
        #for model_num in tqdm.tqdm(range(1), desc = 'Progress'):
            model_aico = round(aico_min__+model_num*aico_delta, 3)
            model_name = path_model+str(model_aico)+'_'+str(round(model_aico+aico_delta, 3))+'__'+str(epoch_num)+'__'+str(batch_num)
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(model_name, compile = False)
            for num in range(aico_num):
                aico_min = round(aico_min__+num*aico_delta, 3)
                testdata_QC = np.load('%s/QC/'%(path_testdata)+str(aico_min)+'_testdata.npy')
                Result_QC[aico_min] += np.round(model.predict(testdata_QC)[:, 1]-0.45).astype(int)
                testdata_others = np.load('%s/Others/'%(path_testdata)+str(aico_min)+'_testdata.npy')
                Result_others[aico_min] += np.round(model.predict(testdata_others)[:, 1]-0.45).astype(int)
                
    f = open('%s'%(output_flnm), 'w')
    print('    TP  FN  FP  TN', file = f)
    TP_sum = 0
    FN_sum = 0
    FP_sum = 0
    TN_sum = 0
    for i in Result_QC.keys():
        TP = np.count_nonzero(Result_QC[i])
        FN = data_num_test - TP
        FP = np.count_nonzero(Result_others[i])
        TN = data_num_test - FP
        print('{:<.3f}'.format(i), '{:<5d}'.format(TP), '{:<5d}'.format(FN),'{:<5d}'.format(FP),'{:<5d}'.format(TN), file = f)
        TP_sum += TP
        FN_sum += FN
        FP_sum += FP
        TN_sum += TN
    print('ALL   TP  FN  FP  TN', file = f)
    print('{:<5d}'.format(TP_sum), '{:<5d}'.format(FN_sum),'{:<5d}'.format(FP_sum),'{:<5d}'.format(TN_sum), file = f)
    f.close()
