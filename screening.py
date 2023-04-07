import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import glob
import warnings
warnings.simplefilter('ignore')
import unicodedata
import tqdm
import copy
import time

def run(path_model, path_exptdata, extension, path_output, output_file_name, aico_min, aico_max, aico_delta):
    """
    screens PXRD patterns.
    input:
    :param char path_model
    :param char path_exptdata
    :param char extension
    :param char path_output
    :param char output_file_name
    :param float aico_min
    :param float aico_max
    :param float aico_delta
    output:
    """

    if os.path.isdir('%s'%(path_output))==False:
        os.mkdir('%s'%(path_output))
    
    file_list = glob.glob(path_exptdata+'/*.'+extension)
    len_file_list = len(file_list)
    files = {}
    count_error = 0
    
    for file_name in file_list[:]:
        try:
            f = open(file_name, 'r')
            Intensity_list = []
            for line in f.readlines():
                try:
                    if line[0] == '*':
                        continue
                    line_list = line[:-1].split()
                    tth = float(line_list[0])
                    if 20 <= tth < 80:
                        Intensity = float(line_list[1])
                        Intensity_list.append(Intensity)
                except:
                    pass
            f.close()
            
            x_test = np.array([Intensity_list], np.float64)
            x_test = x_test-np.min(x_test, axis = 1).reshape(1, 1)
            x_test = x_test/np.max(x_test, axis = 1).reshape(1, 1)
            tf.keras.backend.set_floatx('float64')
            x_test_ = x_test[..., tf.newaxis]
            files[file_name] = x_test_
        except:
            print("=============================")
            print('Error! Please check the file.')
            print(file_name[len(path_exptdata):])
            print("=============================")
            file_list.remove(file_name)
            count_error += 1
            print(count_error, '/', len_file_list, ', error / file')

    output_file = open(path_output+'/'+output_file_name, 'w')
    print('Expt. data: ', path_exptdata, '/', file = output_file)

    model_num = int((aico_max-aico_min)/aico_delta)
    epoch_num = 12 
    batch_num = 256 
    remove_overlap_files = 1 # Choose the highest prediction probability

    count_error = 0
    dic_detection = {'A': [], 'B': [], 'C': []}
    time_sum = 0.0
    except_list = []
    for num in tqdm.tqdm(range(model_num), desc = 'Progress'):
        aico = aico_min+aico_delta*num
        aico = round(aico, 3)
        model_name = path_model+'/'+str(aico)+'_'+str(round(aico+aico_delta, 3))+'__'+str(epoch_num)+'__'+str(batch_num)
        tf.keras.backend.clear_session()

        model = tf.keras.models.load_model(model_name, compile = False)
        ch = 0
        for file_name in file_list[:]:
            try:
                x_test_ = files[file_name]
                pred = float(model(x_test_, training=False)[0][1])
                pred = round(pred, 5)
                if pred < 0.95:
                    continue
                if ch == 0:
                    print("###########################")
                    print('icosahedral lattice constant', aico, 'Å')
                    print("###########################")
                    ch = 1
                if 0.95 <= pred < 0.99:
                    print('Detection lebel "C"')
                    print('file_name', file_name[len(path_exptdata):])
                    print('Prediction value', pred)
                    print("----------------------------------------------------------------")
                    dic_detection['C'].append([file_name[len(path_exptdata):], pred, aico])
                elif 0.99 <= pred < 0.999:
                    print('Detection lebel "B"')
                    print('file_name', file_name[len(path_exptdata):])
                    print('Prediction value', pred)
                    print("----------------------------------------------------------------")
                    dic_detection['B'].append([file_name[len(path_exptdata):], pred, aico])
                else:
                    print('Detection lebel "A"')
                    print('file_name', file_name[len(path_exptdata):])
                    print('Prediction value', pred)
                    print("----------------------------------------------------------------")
                    dic_detection['A'].append([file_name[len(path_exptdata):], pred, aico])
            except:
                print("=============================")
                print('Error! Please check the file.')
                print(file_name[len(path_exptdata):])
                print("=============================")
                file_list.remove(file_name)
                count_error += 1
                print(count_error, '/', len_file_list, ', error / file')
    print('files> ', len_file_list, file = output_file)
    print('error_files> ', count_error, file = output_file)
    print('files for screening> ', len_file_list-count_error, file = output_file)
    print('', file = output_file)

    sortsecond = lambda val : val[1]
    dicA_sorted = sorted(dic_detection['A'], key = sortsecond, reverse = True)
    dicB_sorted = sorted(dic_detection['B'], key = sortsecond, reverse = True)
    dicC_sorted = sorted(dic_detection['C'], key = sortsecond, reverse = True)

    def remove_overlap(list_sorted, file_names):
        for __ in list_sorted[:]:
            file_name = __[0]
            if file_name in file_names:
                list_sorted.remove(__)
            else:
                file_names.append(file_name)
        return list_sorted, file_names

    if remove_overlap_files == 1:
        file_names = []
        dicA_sorted, file_names = remove_overlap(dicA_sorted, file_names)
        dicB_sorted, file_names = remove_overlap(dicB_sorted, file_names)
        dicC_sorted, file_names = remove_overlap(dicC_sorted, file_names)

    dic_None = {'A': 0, 'B': 0, 'C': 0}
    for i in dic_detection:
        if len(dic_detection[i]) == 0:
            dic_detection[i] = [['None', 'None', 'None']]
            dic_None[i] = -1

    num_Adata, num_Bdata, num_Cdata = len(dicA_sorted)+dic_None['A'], len(dicB_sorted)+dic_None['B'], len(dicC_sorted)+dic_None['C']
    print('============================')
    print('Detection Level "A" >', num_Adata, 'data')
    print('Detection Level "B" >', num_Bdata, 'data')
    print('Detection Level "C" >', num_Cdata, 'data')
    print('============================')

    def get_len(word):
        count = 0
        for i in word:
            if unicodedata.east_asian_width(i) in "FWA":
                count += 1
        return len(word)+count

    def get_lenmax(row, list1, list2, list3):
        return max([get_len(str(x[row])) for x in list1]+[get_len(str(x[row])) for x in list2]+[get_len(str(x[row])) for x in list3])

    lenmax1 = get_lenmax(0, dicA_sorted, dicB_sorted, dicC_sorted)+5
    lenmax2 = get_lenmax(1, dicA_sorted, dicB_sorted, dicC_sorted)+5
    lenmax3 = get_lenmax(2, dicA_sorted, dicB_sorted, dicC_sorted)+5

    def len_just(word, lenmax):
        word = str(word)
        len_ = get_len(word)
        return word+' '*(lenmax-len_)

    len_sep = lenmax1+lenmax2+lenmax3+2-5

    print(len_just('file', lenmax1), len_just('pred_value', lenmax2), len_just('a [Å]', lenmax3), file = output_file)

    print('-'*len_sep, file = output_file)
    print('Detection Level A', file = output_file)
    print('-'*len_sep, file = output_file)
    for i in dicA_sorted:
        print(len_just(i[0], lenmax1), len_just(i[1], lenmax2), len_just(i[2], lenmax3), file = output_file)

    print('-'*len_sep, file = output_file)
    print('Detection Level B', file = output_file)
    print('-'*len_sep, file = output_file)
    for i in dicB_sorted:
        print(len_just(i[0], lenmax1), len_just(i[1], lenmax2), len_just(i[2], lenmax3), file = output_file)

    print('-'*len_sep, file = output_file)
    print('Detection Level C', file = output_file)
    print('-'*len_sep, file = output_file)
    for i in dicC_sorted:
        print(len_just(i[0], lenmax1), len_just(i[1], lenmax2), len_just(i[2], lenmax3), file = output_file)

    print('============================', file = output_file)
    print('Detection Level "A" >', num_Adata, 'data', file = output_file)
    print('Detection Level "B" >', num_Bdata, 'data', file = output_file)
    print('Detection Level "C" >', num_Cdata, 'data', file = output_file)
    print('============================', file = output_file)

    output_file.close()
    return 0

if __name__ == '__main__':
    
    import datetime
    import pytz
    today = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    
    path_model = './models/20230322'
    path_exptdata = '../datasets/ohhashi_imram'
    #path_exptdata = '../expt_data/xrd_iwasaki'
    extension = 'txt' # extension of exptdata
    path_output = './screening_result'
    output_file_name = 'result_'+str(today)[:-13].replace(':', '-')+'.txt'
    
    aico_min = 4.00
    aico_max = 6.00
    aico_delta = 0.025
    
    run(path_model, path_exptdata, extension, path_output, output_file_name, aico_min, aico_max, aico_delta)
