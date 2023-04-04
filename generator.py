from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import random
import glob
import math
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pxrd

dic_wvl = {}
dic_wvl['Cu_Ka'] = 1.54059 # in Ang.

def reflection_list(hklmno_range, wvl, aico_max, tth_max, qperp_cutoff):
    """
    generates independent reflections list
    input:
    :param int hklmno_range
    :param float wvl
    :param float aico_max
    :param float tth_max
    :param float qperp_cutoff
    """
    return pxrd.independent_reflection_list(hklmno_range, wvl, aico_max, tth_max, qperp_cutoff)

def independent_reflection_list_in_tth_range(ref_list, wvl, aico, tth_min, tth_max):
    """
    selects independent reflections in a specified tth-range
    input:
    :param list  ref_list
    :param float wvl
    :param float aico_max
    :param float tth_max
    :param float qperp_cutoff
    """
    return pxrd.selector(ref_list, wvl, aico, tth_min, tth_max)

def dataset(wvl, QC_peaks, aico_min, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_qc, data_num_nonqc):
    """
    generates dataset.
    input:
    :param 
    
    output:
    list multiQC, multiphase iQC powder x-ray diffraction patterns 
    list other,  nonQC powder x-ray diffraction patterns 
    """
    aico = aico_min
    print('icosahedral lattice constant [Å]', aico)
    
    # Multi-iQC dataset
    #QC_peaks = calc_QC_peaks(hklmno_range, aico, aico+aico_delta, wvl, tth_min, tth_max, 1.5)
    virtualQC = pxrd.calc_virtualiQC(data_num_qc, QC_peaks, wvl, aico, aico+aico_delta, tth_min, tth_max, tth_step)
    multiQC = pxrd.calc_multiQC(virtualQC, tth_min, tth_max, tth_step)

    # Non-iQC dataset
    other = pxrd.calc_others(data_num_nonqc, tth_min, tth_max, tth_step)
    
    return multiQC, other

def dataset_labeled(wvl, QC_peaks, aico_min, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_qc, data_num_non_qc):
    """
    generates labeled dataset
    """
    #aico = aico_min
    #print('Lattice constant [Å]', aico)

    myData = []
    multiQC_data, others_data = dataset(wvl, QC_peaks, aico_min, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_qc, data_num_non_qc)
    
    # Multi-iQC
    #virtualQC_data = pxrd.calc_virtualiQC(data_num_qc, QC_peaks, wvl, aico, aico+aico_delta, tth_min, tth_max, tth_step)
    #multiQC_data = pxrd.calc_multiQC(virtualQC_data, tth_min, tth_max, tth_step)
    for i in multiQC_data:
        myData.append([i,1])

    # Non-iQC
    #others_data = pxrd.calc_others(data_num_non_qc, tth_min, tth_max, tth_step)
    for i in others_data:
        myData.append([i,0])

    random.shuffle(myData)
    x_data = []
    y_data = []
    for feature, label in myData:
        x_data.append(feature)
        y_data.append(label)
    
    #np.save('./x_data.npy', x_train)
    #np.save('./y_data.npy', y_train)
    x_data = np.array(x_data, dtype='float64')
    y_data = np.array(y_data, dtype='float64')
    
    return x_data, y_data

if __name__ == '__main__':
    
    
    ####
    
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
    #ref_list = pxrd.calc_QC_peaks(hklmno_range, aico, aico_max, wvl, tth_min, tth_max, qperp_cutoff)
    tmp = reflection_list(hklmno_range, wvl, aico_max, tth_max, qperp_cutoff)
    ref_list = independent_reflection_list_in_tth_range(tmp, wvl, aico_max, tth_min, tth_max)
    #print(len(ref_list))
    
    # training data
    #data_num_train_qc    = 30000
    #data_num_train_other = 30000
    data_num_train_qc    = 20
    data_num_train_other = 20
    x_train, y_train = dataset_labeled(wvl, ref_list, aico, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_train_qc, data_num_train_other)
    
    # test data
    #data_num_test_qc    = 10000
    #data_num_test_other = 10000
    data_num_test_qc    = 20
    data_num_test_other = 20
    x_test, y_test = dataset_labeled(wvl, ref_list, aico, aico_delta, hklmno_range, tth_min, tth_max, tth_step, data_num_test_qc, data_num_test_other)
