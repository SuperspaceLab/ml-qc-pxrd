from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import random
import glob
import math
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pxrd

TAU = (1 + np.sqrt(5))/2

dic_wvl = {}
dic_wvl['Cu_Ka'] = 1.54059 # in Ang.
dic_wvl['Cu_Kb'] = 1.3810
wvl = dic_wvl['Cu_Ka']

def reflection_list(hklmno_range, wvl, aico_max, tth_max, qperp_cutoff):
    """
    generates independent reflection list
    input:
    : int hklmno_range
    : float wavelength
    """
    return pxrd.independent_reflection_list(hklmno_range, wvl, aico_max, tth_max, qperp_cutoff)

def independent_reflection_list_in_tth_range(a, wvl, aico, tth_min, tth_max):
    return pxrd.selector(a, wvl, aico, tth_min, tth_max)

def dataset(path, wvl, QC_peaks, aico_min, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_qc, data_num_non_qc):

    aico = aico_min
    print('Lattice constant [Å]', aico)

    myData = []
    
    #QC_peaks = calc_QC_peaks(hklmno_range, aico_min, aico_max, wvl, tth_min, tth_max)
    
    # Multi-iQC
    virtualQC_data = pxrd.calc_virtualiQC(data_num_qc, QC_peaks, wvl, aico, aico+aico_delta, tth_min, tth_max, tth_step)
    multiQC_data = pxrd.calc_multiQC(virtualQC_data, tth_min, tth_max, tth_step)
    for i in multiQC_data:
        myData.append([i,1])

    # Non-iQC
    others_train = pxrd.calc_others(data_num_non_qc, tth_min, tth_max, tth_step)
    for i in others_train:
        myData.append([i,0])

    random.shuffle(myData)
    x_data = []
    y_data = []
    for feature, label in myData:
        x_data.append(feature)
        y_data.append(label)
    
    #np.save('%s/x_data.npy'%(path), x_train)
    #np.save('%s/y_data.npy'%(path), y_train)
    x_data = np.array(x_data, dtype='float64')
    y_data = np.array(y_data, dtype='float64')
    return x_data, y_data

if __name__ == '__main__':
    
    path = '.'
    aico = 5.0 # in Ang
    aico_delta = 0.025
    hklmno_range = 3
    #hklmno_range = 6
    tth_min  = 20.0 # in degree
    tth_max  = 80.0
    tth_step = 0.01
    qperp_cutoff = 1.5 # in r.l.u (Yamamoto's setting).  this corresponds to 1.5*sqrt(2)=2.12... in r.l.u in Cahn-Gratias setting. 
    wvl = dic_wvl['Cu_Ka']
    
    aico_max = aico + aico_delta
    #ref_list = calc_QC_peaks(hklmno_range, aico, aico_max, wvl, tth_min, tth_max, qperp_cutoff)
    tmp = reflection_list(hklmno_range, wvl, aico_max, tth_max, qperp_cutoff)
    ref_list = independent_reflection_list_in_tth_range(tmp, wvl, aico_max, tth_min, tth_max)
    #print(len(ref_list))
    
    # training data
    #data_num_train_qc    = 30000
    #data_num_train_other = 30000
    data_num_train_qc    = 20
    data_num_train_other = 20
    x_train, y_train = dataset(path, wvl, ref_list, aico, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_train_qc, data_num_train_other)
    
    # test data
    #data_num_test_qc    = 10000
    #data_num_test_other = 10000
    data_num_test_qc    = 20
    data_num_test_other = 20
    x_test, y_test = dataset(path, wvl, ref_list, aico, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_test_qc, data_num_test_other)
