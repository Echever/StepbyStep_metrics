import pandas as pd

from os import listdir
from os.path import isfile, join

import numpy as np

from scipy import signal
from scipy.signal import butter, freqz, filtfilt

import re

import pickle


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, fs=None)
    return b, a

def butter_lowpass_filter(x, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, x)

def smooth_signal(data):
    dx = data.index[1]
    for col in data.columns:
        #smoothed = signal.butter(2, 6/60, btype='low', analog=True)
        #print(np.array([data.index,data[col]]).T)
        #filtered = signal.filtfilt(2, 0.25, np.array([data.index,data[col]]).T)

        #smoothed = signal.savgol_filter(data[col], 5, 2, deriv=1, delta=dx)
        data[col] = butter_lowpass_filter(data[col], 2, 60)
    return data


def extract_features(data):
    """[summary]
    This function returns the action sequence and indexes where changes 
    occur.
    Args:
        data (Pandas DataFrame): The dataframe containing the original 
                                 or cleaned data

    Returns:
        action_sequence [List]: Contains the cluster ID of the action 
                                executed.
        change_index [List]: Contains the frame ID where the action
                             contained in action_sequence
    """
    change_index = []
    action_sequence = []
    cur_value = data.iloc[0].values
    action_sequence.append(str(cur_value[0]))
    for i in range(len(data)):
        if(cur_value != data.iloc[i].values):
            cur_value = data.iloc[i].values
            change_index.append(i)
            action_sequence.append(str(cur_value[0]))
    
    return action_sequence, change_index

def PI_avg_gait_time(gait_seq, action_seq, index_seq):
    gait_length = len(gait_seq)
    gait_list = [int(m.start()) for m in re.finditer(gait_seq,
                                                action_seq)]
    return gait_list

def clean_data(data, n_frames_tol = 5):
    """[summary]
    Optional step. Consists on removing the miscalculations, understanding
    miscalculations as a couple of frames that are misclassified. 
    
    Args:
        data (Pandas DataFrame): The dataframe containing the original 
                                 data
        n_frames_tol (Int): The maximum ammount of frames of tolerance
                            to detect a misclassification.

    Returns:
        cleaned_data [Pandas DataFrame]: The dataframe after removing the 
                                         noise or misclassifications.
    """
    change_index = []
    change_ii = 0
    cur_value = data.iloc[0].values
    change_index.append(cur_value)
    cleaned_data = data.copy()
    for i in range(len(data)):
        if(cur_value != data.iloc[i].values):
            if(i - n_frames_tol > change_index[change_ii]):
                change_ii = change_ii + 1
                change_index.append(i)
            else:
                repeat_val = np.repeat( data.iloc[i].values, i - 
                                        change_index[change_ii])
                repeat_val = np.expand_dims(repeat_val, 1)
                cleaned_data[change_index[change_ii]:i] = repeat_val
            cur_value = data.iloc[i].values
    return cleaned_data

#Loading and preparing the data
mypath='Input/Data/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

test_file=files[0]
print(test_file)
test=pd.read_csv(mypath+test_file)

#test=test.drop(columns='Label')
test=test.drop(columns='Frame_Idx')
test=smooth_signal(test)


#Classify the movement as ascend or descend
clasi_sub_baj = pickle.load(open("Input/Models/model_descend_ascend.sav", 'rb'))
descend = clasi_sub_baj.predict(test[0:500].to_numpy().flatten().reshape(1,-1))

#Loading the correct model
if(descend==1):
    modelo=pickle.load(open("Input/Models/model_descend.sav", 'rb'))
else:
    modelo=pickle.load(open("Input/Models/model_ascend.sav", 'rb'))


#The variables are chosen and a window is created for each observation
S_test=test[['RK-R_X','LK-R_X']].to_numpy()
datos=[]

window=30

for i in range(window,S_test.shape[0]-window):
    datos.append(S_test[i-window:i+window,:].flatten())

X_test=np.asarray(datos)


#The prediction is performed and the observations removed due to the window are added again
pred=modelo.predict(X_test)
pred=np.asarray(pred,dtype=int)
pred=list(np.pad(pred, (window, window), 'constant'))


#Estimate events based on the sequence of predictions

action_seq = clean_data(pd.DataFrame(pred))
action_seq, index_seq=extract_features(action_seq)

separator = ''
action_seq = separator.join(action_seq)
list_gait_seq=["67","39","34","610"]
radius = 10

L_TO=[]
R_TO=[]

L_MS=[]
R_MS=[]

L_FS=[]
R_FS=[]

LK = S_test[:,1]
RK = S_test[:,0]
for i in list_gait_seq:
    gait_seq=i
    f=PI_avg_gait_time(gait_seq, action_seq, index_seq)
    for k in f:
        inic=index_seq[k-1]
        final=index_seq[k+1]
        if(gait_seq=="67"):
            data= LK[inic:final]

            L_TO.append(index_seq[k-1])
            L_MS.append(index_seq[k])
            L_FS.append(index_seq[k+1])
        
        elif(gait_seq=="34"):
            
            R_TO.append(index_seq[k-1])
            R_MS.append(index_seq[k])
            R_FS.append(index_seq[k+1])
            

P_TO=[]
S_TO=[]

P_MS=[]
S_MS=[]

P_FS=[]
S_FS=[]

if(R_TO[0]>=L_TO[0]):
    P_TO= L_TO
    S_TO= R_TO

    P_MS= L_MS
    S_MS= R_MS

    P_FS= L_FS
    S_FS= R_FS
elif(L_TO[0]>R_TO[0]):
    P_TO= R_TO
    S_TO= L_TO

    P_MS= R_MS
    S_MS= L_MS

    P_FS= R_FS
    S_FS= L_FS


import yaml

def yaml_export(data, output_dir, category, datatype):
    tmp_dict = dict(type=datatype, value=data.tolist())
    with open('{}/{}.yml'.format(output_dir,category), 'w') as file:
        _ = yaml.dump(tmp_dict, file, default_flow_style=True)
        
output_dir="Output/"

if(descend==1):
    yaml_export(np.mean([S_FS[-1]-P_TO[0]]), output_dir,  'total_time_descend', 'scalar')
    #weight_acceptance_descend (definiedo cycle como foot strike de la misma pierna)

    resultados=[]

    for i in range(len(P_FS)-1):
        gait_time=P_FS[i+1]-P_FS[i]
        resultados.append((S_TO[i]-P_FS[i])/(P_FS[i+1]-P_FS[i]))

    for i in range(len(S_FS)-1):
        gait_time=S_FS[i+1]-S_FS[i]
        resultados.append((P_TO[i+1]-S_FS[i])/(S_FS[i+1]-S_FS[i]))

    resultados = [0 if i < 0 else i for i in resultados]

    yaml_export(np.mean(resultados), output_dir,  'weight_acceptance_ascend', 'scalar')
    
    # forward_continuance_descend (definiedo cycle como foot strike de la misma pierna)

    resultados=[]

    for i in range(len(P_FS)-1):
        gait_time=P_FS[i+1]-P_FS[i]
        resultados.append((S_MS[i]-S_TO[i])/(P_FS[i+1]-P_FS[i]))

    for i in range(len(S_FS)-1):
        gait_time=S_FS[i+1]-S_FS[i]
        resultados.append((P_MS[i]-P_TO[i])/(S_FS[i+1]-S_FS[i]))


    resultados = [0 if i < 0 else i for i in resultados]

    yaml_export(np.mean(resultados), output_dir,  'pull_up', 'scalar')
    
    # Controlled_lowering_descend
    
    resultados=[]

    for i in range(len(P_FS)-1):
        gait_time=P_FS[i+1]-P_FS[i]
        resultados.append((P_TO[i+1]-S_MS[i])/(P_FS[i+1]-P_FS[i]))

    for i in range(len(S_FS)-1):
        gait_time=S_FS[i+1]-S_FS[i]
        resultados.append((S_TO[i]-P_MS[i])/(S_FS[i+1]-S_FS[i]))


    resultados = [0 if i < 0 else i for i in resultados]

    yaml_export(np.mean(resultados), output_dir,  'forward_continuance_ascend', 'scalar')
    
    #  Leg_pull_through_descend

    resultados=[]

    for i in range(len(P_FS)-1):
        gait_time=P_FS[i+1]-P_FS[i]
        resultados.append((P_MS[i+1]-P_TO[i+1])/(P_FS[i+1]-P_FS[i]))

    for i in range(len(S_FS)-1):
        gait_time=S_FS[i+1]-S_FS[i]
        resultados.append((S_MS[i+1]-S_TO[i+1])/(S_FS[i+1]-S_FS[i]))


    resultados = [0 if i < 0 else i for i in resultados]

    yaml_export(np.mean(resultados), output_dir,  'swing_foot_clearance', 'scalar')

    # Foot_placement_descend
    resultados=[]

    for i in range(len(P_FS)-1):
        gait_time=P_FS[i+1]-P_FS[i]
        resultados.append((P_FS[i+1]-P_MS[i+1])/(P_FS[i+1]-P_FS[i]))

    for i in range(len(S_FS)-1):
        gait_time=S_FS[i+1]-S_FS[i]
        resultados.append((S_FS[i+1]-S_MS[i+1])/(S_FS[i+1]-S_FS[i]))


    resultados = [0 if i < 0 else i for i in resultados]

    yaml_export(np.mean(resultados), output_dir,  'swing_foot_placement', 'scalar')



# In[16]:


#Subida
if(descend==0):
    yaml_export(np.mean([S_FS[-1]-P_TO[0]]), output_dir,  'total_time_ascend', 'scalar')
    #weight_acceptance_ascend (definiedo cycle como foot strike de la misma pierna)

    resultados=[]

    for i in range(len(P_FS)-1):
        gait_time=P_FS[i+1]-P_FS[i]
        resultados.append((S_TO[i]-P_FS[i])/(P_FS[i+1]-P_FS[i]))

    for i in range(len(S_FS)-1):
        gait_time=S_FS[i+1]-S_FS[i]
        resultados.append((P_TO[i+1]-S_FS[i])/(S_FS[i+1]-S_FS[i]))


    resultados = [0 if i < 0 else i for i in resultados]


    yaml_export(np.mean(resultados), output_dir,  'weight_acceptance_ascend', 'scalar')

    #pull up 

    resultados=[]

    for i in range(len(P_FS)-1):
        gait_time=P_FS[i+1]-P_FS[i]
        resultados.append((S_MS[i]-S_TO[i])/(P_FS[i+1]-P_FS[i]))

    for i in range(len(S_FS)-1):
        gait_time=S_FS[i+1]-S_FS[i]
        resultados.append((P_MS[i]-P_TO[i])/(S_FS[i+1]-S_FS[i]))


    resultados = [0 if i < 0 else i for i in resultados]

    yaml_export(np.mean(resultados), output_dir,  'pull_up', 'scalar')

    # forward_continuance_ascend

    resultados=[]

    for i in range(len(P_FS)-1):
        gait_time=P_FS[i+1]-P_FS[i]
        resultados.append((S_FS[i]-S_MS[i])/(P_FS[i+1]-P_FS[i]))

    for i in range(len(S_FS)-1):
        gait_time=S_FS[i+1]-S_FS[i]
        resultados.append((P_FS[i]-P_MS[i])/(S_FS[i+1]-S_FS[i]))

    print(resultados) #Hacer la media

    resultados = [0 if i < 0 else i for i in resultados]

    yaml_export(np.mean(resultados), output_dir,  'forward_continuance_ascend', 'scalar')


    # push up

    resultados=[]

    for i in range(len(P_FS)-1):
        gait_time=P_FS[i+1]-P_FS[i]
        resultados.append((P_TO[i+1]-S_FS[i])/(P_FS[i+1]-P_FS[i]))

    for i in range(len(S_FS)-1):
        gait_time=S_FS[i+1]-S_FS[i]
        resultados.append((S_TO[i]-P_FS[i])/(S_FS[i+1]-S_FS[i]))

    resultados = [0 if i < 0 else i for i in resultados]

    yaml_export(np.mean(resultados), output_dir,  'push_up', 'scalar')


    # swing_foot_clearance

    resultados=[]

    for i in range(len(P_FS)-1):
        gait_time=P_FS[i+1]-P_FS[i]
        resultados.append((P_MS[i+1]-P_TO[i+1])/(P_FS[i+1]-P_FS[i]))

    for i in range(len(S_FS)-1):
        gait_time=S_FS[i+1]-S_FS[i]
        resultados.append((S_MS[i+1]-S_TO[i+1])/(S_FS[i+1]-S_FS[i]))

    resultados = [0 if i < 0 else i for i in resultados]

    yaml_export(np.mean(resultados), output_dir,  'swing_foot_clearance', 'scalar')

    # swing_foot_placement
    resultados=[]

    for i in range(len(P_FS)-1):
        gait_time=P_FS[i+1]-P_FS[i]
        resultados.append((P_FS[i+1]-P_MS[i+1])/(P_FS[i+1]-P_FS[i]))

    for i in range(len(S_FS)-1):
        gait_time=S_FS[i+1]-S_FS[i]
        resultados.append((S_FS[i+1]-S_MS[i+1])/(S_FS[i+1]-S_FS[i]))


    resultados = [0 if i < 0 else i for i in resultados]

    yaml_export(np.mean(resultados), output_dir,  'swing_foot_placement', 'scalar')


test=pd.read_csv('Input/Data/'+test_file)

def PI_Sensors_Hip(sensors_data):
    return  np.max(sensors_data['Hips-R_Z']),             np.min(sensors_data['Hips-R_Z']),             np.mean(sensors_data['Hips-R_Z']),             np.std(sensors_data['Hips-R_Z'])

def PI_Sensors_LUL(sensors_data):
    return  np.max(sensors_data['LUL-R_Y']),             np.min(sensors_data['LUL-R_Y']),             np.mean(sensors_data['LUL-R_Y']),             np.std(sensors_data['LUL-R_Y'])

def PI_Sensors_LK(sensors_data):
    return  np.max(sensors_data['LK-R_Y']),             np.min(sensors_data['LK-R_Y']),             np.mean(sensors_data['LK-R_Y']),             np.std(sensors_data['LK-R_Y'])

def PI_Sensors_LF(sensors_data):
    return  np.max(sensors_data['LF-R_Y']),             np.min(sensors_data['LF-R_Y']),             np.mean(sensors_data['LF-R_Y']),             np.std(sensors_data['LF-R_Y'])


def PI_Sensors_RUL(sensors_data):
    return  np.max(sensors_data['RUL-R_Y']),             np.min(sensors_data['RUL-R_Y']),             np.mean(sensors_data['RUL-R_Y']),             np.std(sensors_data['RUL-R_Y'])

def PI_Sensors_RK(sensors_data):
    return  np.max(sensors_data['RK-R_Y']),             np.min(sensors_data['RK-R_Y']),             np.mean(sensors_data['RK-R_Y']),             np.std(sensors_data['RK-R_Y'])

def PI_Sensors_RF(sensors_data):
    return  np.max(sensors_data['RF-R_Y']),             np.min(sensors_data['RF-R_Y']),             np.mean(sensors_data['RF-R_Y']),             np.std(sensors_data['RF-R_Y'])



hips_max, hips_min, hips_mean, hips_std = PI_Sensors_Hip(test)
print('Hips Rotation (Max angle Z) : {}'.format(hips_max))
print('Hips Rotation (Min angle Z) : {}'.format(hips_min))
print('Hips Rotation (Mean angle Z): {}'.format(hips_mean))
print('Hips Rotation (Std angle Z) : {}'.format(hips_std))
hips_vector = np.array([hips_max, hips_min, hips_mean, hips_std])
yaml_export(hips_vector, output_dir,  files[0]+ '_hips_angle', 'vector')

lul_max, lul_min, lul_mean, lul_std = PI_Sensors_LUL(test)
print('\n \nlul Rotation (Max angle Y) : {}'.format(lul_max))
print('lul Rotation (Min angle Y) : {}'.format(lul_min))
print('lul Rotation (Mean angle Y): {}'.format(lul_mean))
print('lul Rotation (Std angle Y) : {}'.format(lul_std))
lul_vector = np.array([lul_max, lul_min, lul_mean, lul_std])
yaml_export(lul_vector, output_dir,  files[0]+'_lul_angle', 'vector')

lk_max, lk_min, lk_mean, lk_std = PI_Sensors_LK(test)
print('\n \nlk Rotation (Max angle Y) : {}'.format(lk_max))
print('lk Rotation (Min angle Y) : {}'.format(lk_min))
print('lk Rotation (Mean angle Y): {}'.format(lk_mean))
print('lk Rotation (Std angle Y) : {}'.format(lk_std))
lk_vector = np.array([lk_max, lk_min, lk_mean, lk_std])
yaml_export(lk_vector, output_dir,  files[0]+'_lk_angle', 'vector')

lf_max, lf_min, lf_mean, lf_std = PI_Sensors_LF(test)
print('\n \nlf Rotation (Max angle Y) : {}'.format(lf_max))
print('lf Rotation (Min angle Y) : {}'.format(lf_min))
print('lf Rotation (Mean angle Y): {}'.format(lf_mean))
print('lf Rotation (Std angle Y) : {}'.format(lf_std))
lf_vector = np.array([lf_max, lf_min, lf_mean, lf_std])
yaml_export(lf_vector, output_dir, files[0]+ '_lf_angle', 'vector')


rul_max, rul_min, rul_mean, rul_std = PI_Sensors_RUL(test)
print('\n \nrul Rotation (Max angle Y) : {}'.format(rul_max))
print('rul Rotation (Min angle Y) : {}'.format(rul_min))
print('rul Rotation (Mean angle Y): {}'.format(rul_mean))
print('rul Rotation (Std angle Y) : {}'.format(rul_std))
rul_vector = np.array([rul_max, rul_min, rul_mean, rul_std])
yaml_export(rul_vector, output_dir,  files[0]+'_rul_angle', 'vector')

rk_max, rk_min, rk_mean, rk_std = PI_Sensors_RK(test)
print('\n \nrk Rotation (Max angle Y) : {}'.format(rk_max))
print('rk Rotation (Min angle Y) : {}'.format(rk_min))
print('rk Rotation (Mean angle Y): {}'.format(rk_mean))
print('rk Rotation (Std angle Y) : {}'.format(rk_std))
rk_vector = np.array([rk_max, rk_min, rk_mean, rk_std])
yaml_export(rk_vector, output_dir, files[0]+ '_rk_angle', 'vector')

rf_max, rf_min, rf_mean, rf_std = PI_Sensors_RF(test)
print('\n \nrf Rotation (Max angle Y) : {}'.format(rf_max))
print('rf Rotation (Min angle Y) : {}'.format(rf_min))
print('rf Rotation (Mean angle Y): {}'.format(rf_mean))
print('rf Rotation (Std angle Y) : {}'.format(rf_std))
rf_vector = np.array([rf_max, rf_min, rf_mean, rf_std])
yaml_export(rf_vector, output_dir,  files[0]+'_rf_angle', 'vector')


# In[ ]:




