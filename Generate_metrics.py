import pandas as pd
import os
import numpy as np
from scipy import signal
from scipy.signal import butter, freqz, filtfilt
import re
import pickle
import yaml
import sys
from termcolor import colored



def smooth_signal(data):
    """[summary]
    This function applies a smoothing filter to the data
    Args:
        data (Pandas DataFrame): The dataframe containing the original  data

    Returns:
        data (Pandas DataFrame): Smoothed data
    """
    for col in data.columns:
        data[col] = butter_lowpass_filter(data[col], 2, 60)
    return data

def butter_lowpass(cutoff, fs, order=5):
    """[summary]
    Calculating butterworth filter parameters for smoothing the signal
    Args:
        cutoff (Scalar): Parameter to control the time window that is going to be smoothed.
        fs (Scalar): Sample rate

    Returns:
        data (Pandas DataFrame): Smoothed data
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, fs=None)
    return b, a

def butter_lowpass_filter(x, cutoff, fs, order=5):
    """[summary]
    This function applies butterworth filter to the data in both directions
    Args:
        x (Pandas DataFrame): One variable of the dataframe
        cutoff (Scalar): Parameter to control the time window that is going to be smoothed.
        fs (Scalar): Sample rate

    Returns:
        data (Pandas DataFrame): Smoothed data
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, x)


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

def generate_gait_seq(gait_seq, action_seq, index_seq):
    """[summary]
    From a sequence of numbers, it emits the position of the indices 
    in which a specific sequence is found
    Args:
        gait_seq [List]: The desired digit sequence
        action_sequence [List]: Contains the cluster ID of the action 
                                executed.
        change_index [List]: Contains the frame ID where the action
                             contained in action_sequence
    Returns:
        gait_list [List]: The position of the searched 
                            sequence in the index list
    """
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

def yaml_export(data, output_dir, category, datatype):
        """[summary]
        Create a yaml file
        Args:
            data (Pandas DataFrame): The dataframe containing the original 
                                    or cleaned data
            output_dir (str): Output directory
            category (str): Name of the metric
            datatype (str): Type of the data, scalar or list.


        Returns:
            action_sequence [List]: Contains the cluster ID of the action 
                                    executed.
            change_index [List]: Contains the frame ID where the action
                                contained in action_sequence
        """

        tmp_dict = dict(type=datatype, value=data.tolist())
        with open('{}/{}.yml'.format(output_dir,category), 'w') as file:
            _ = yaml.dump(tmp_dict, file, default_flow_style=True)

def PI_Sensors_Hip(sensors_data):
    """
    Maximum, minimum, mean and std of the Hips sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['Hips-R_Z']), \
            np.min(sensors_data['Hips-R_Z']), \
            np.mean(sensors_data['Hips-R_Z']), \
            np.std(sensors_data['Hips-R_Z'])

def PI_Sensors_LUL(sensors_data):
    """
    Maximum, minimum, mean and std of the LUL sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['LUL-R_Y']), \
            np.min(sensors_data['LUL-R_Y']), \
            np.mean(sensors_data['LUL-R_Y']), \
            np.std(sensors_data['LUL-R_Y'])

def PI_Sensors_LK(sensors_data):
    """
    Maximum, minimum, mean and std of the LK sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['LK-R_Y']), \
            np.min(sensors_data['LK-R_Y']), \
            np.mean(sensors_data['LK-R_Y']), \
            np.std(sensors_data['LK-R_Y'])

def PI_Sensors_LF(sensors_data):
    """
    Maximum, minimum, mean and std of the LF sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['LF-R_Y']), \
            np.min(sensors_data['LF-R_Y']), \
            np.mean(sensors_data['LF-R_Y']), \
            np.std(sensors_data['LF-R_Y'])


def PI_Sensors_RUL(sensors_data):
    """
    Maximum, minimum, mean and std of the RUL sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['RUL-R_Y']), \
            np.min(sensors_data['RUL-R_Y']), \
            np.mean(sensors_data['RUL-R_Y']), \
            np.std(sensors_data['RUL-R_Y'])

def PI_Sensors_RK(sensors_data):
    """
    Maximum, minimum, mean and std of the RK sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['RK-R_Y']), \
            np.min(sensors_data['RK-R_Y']), \
            np.mean(sensors_data['RK-R_Y']), \
            np.std(sensors_data['RK-R_Y'])

def PI_Sensors_RF(sensors_data):
    """
    Maximum, minimum, mean and std of the RF sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['RF-R_Y']), \
            np.min(sensors_data['RF-R_Y']), \
            np.mean(sensors_data['RF-R_Y']), \
            np.std(sensors_data['RF-R_Y'])


if __name__ == "__main__":

    USAGE = """usage: run_pi file_in folder_out
            file_in: csv file containing at least a timestamp column
            folder_out: folder where the PI yaml file yaml file will be stored
            """
    if len(sys.argv) != 3:
        print(colored("Wrong input parameters !", "red"))
        print(colored(USAGE, "yellow"))
        sys.exit(-1)

    file_in = sys.argv[1]
    output_dir = sys.argv[2]

    # check input parameters are good
    if not os.path.exists(file_in):
        print(colored("Input file {} does not exist".format(file_in), "red"))
        sys.exit(-1)
    if not os.path.isfile(file_in):
        print(colored("Input path {} is not a file".format(file_in), "red"))
        sys.exit(-1)

    if not os.path.exists(output_dir):
        print(colored(
            "Output folder {} does not exist".format(output_dir),
            "red"))
        sys.exit(-1)
    if not os.path.isfile(file_in):
        print(colored("Output path {} is not a folder".format(file_in), "red"))
        sys.exit(-1)
        

    #Loading and preparing the data
    #print(file_in)

    try:
        test = pd.read_csv(file_in)
    except FileNotFoundError as err:
        print(colored(
            "Could not load file {}: {}".format(file_in, err),
            "red"))
        sys.exit(-1)

    test=test.drop(columns='Label')
    test=test.drop(columns='Frame_Idx')
    test=smooth_signal(test)

    #Classify the movement as ascend or descend
    clas_ascend_descend = pickle.load(open("Input/Models/model_descend_ascend.sav", 'rb'))
    descend = clas_ascend_descend.predict(test[0:500].to_numpy().flatten().reshape(1,-1))

    #Loading the correct model
    if(descend==1):
        print("Loading descend model")
        model=pickle.load(open("Input/Models/model_descend.sav", 'rb'))
    else:
        print("Loading ascend model")
        model=pickle.load(open("Input/Models/model_ascend.sav", 'rb'))

    #The variables are chosen and a window is created for each observation
    S_test=test[['RK-R_X','LK-R_X']].to_numpy()
    datos=[]
    window=30
    for i in range(window,S_test.shape[0]-window):
        datos.append(S_test[i-window:i+window,:].flatten())
    X_test=np.asarray(datos)

    #The prediction is performed and the observations removed due to the window are added again
    pred=model.predict(X_test)
    pred=np.asarray(pred,dtype=int)
    pred=list(np.pad(pred, (window, window), 'constant'))

    #Estimate events based on the sequence of predictions, 67 is left leg and 34 rigth leg
    action_seq = clean_data(pd.DataFrame(pred))
    action_seq, index_seq = extract_features(action_seq)

    separator = ''
    action_seq = separator.join(action_seq)
    list_gait_seq = ["67","34"]

    # Create lists of Left/Rigth Toe-Off, Mid-swing and Foot-Strike
    L_TO=[]
    R_TO=[]

    L_MS=[]
    R_MS=[]

    L_FS=[]
    R_FS=[]

    #Define the Toe-Off, Mid-swing and Foot-Strike as the first, second and third element of each sequence
    for i in list_gait_seq:
        gait_seq=i
        f=generate_gait_seq(gait_seq, action_seq, index_seq)
        print(f)
        for k in f:
            if(gait_seq=="67"):
                L_TO.append(index_seq[k-1])
                L_MS.append(index_seq[k])
                L_FS.append(index_seq[k+1])
            elif(gait_seq=="34"):
                R_TO.append(index_seq[k-1])
                R_MS.append(index_seq[k])
                R_FS.append(index_seq[k+1])
                

    # Decide which leg has gone first and assign it to the events to the P_TO, P_MS 
    #   and P_FS lists and the second leg to the S lists S_TO, S_MS and S_FS
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


    #Based on de prediction decide which metrics to compute
    if(descend==1):
        yaml_export(np.mean([S_FS[-1]-P_TO[0]]), output_dir,  'total_time_descend', 'scalar')
        #weight_acceptance_descend 

        results=[]

        for i in range(len(P_FS)-1):
            gait_time=P_FS[i+1]-P_FS[i]
            results.append((S_TO[i]-P_FS[i])/(P_FS[i+1]-P_FS[i]))

        for i in range(len(S_FS)-1):
            gait_time=S_FS[i+1]-S_FS[i]
            results.append((P_TO[i+1]-S_FS[i])/(S_FS[i+1]-S_FS[i]))

        results = [0 if i < 0 else i for i in results]

        yaml_export(np.mean(results), output_dir,  'weight_acceptance_ascend', 'scalar')
        
        # forward_continuance_descend 

        results=[]

        for i in range(len(P_FS)-1):
            gait_time=P_FS[i+1]-P_FS[i]
            results.append((S_MS[i]-S_TO[i])/(P_FS[i+1]-P_FS[i]))

        for i in range(len(S_FS)-1):
            gait_time=S_FS[i+1]-S_FS[i]
            results.append((P_MS[i]-P_TO[i])/(S_FS[i+1]-S_FS[i]))


        results = [0 if i < 0 else i for i in results]

        yaml_export(np.mean(results), output_dir,  'pull_up', 'scalar')
        
        # Controlled_lowering_descend
        
        results=[]

        for i in range(len(P_FS)-1):
            gait_time=P_FS[i+1]-P_FS[i]
            results.append((P_TO[i+1]-S_MS[i])/(P_FS[i+1]-P_FS[i]))

        for i in range(len(S_FS)-1):
            gait_time=S_FS[i+1]-S_FS[i]
            results.append((S_TO[i]-P_MS[i])/(S_FS[i+1]-S_FS[i]))


        results = [0 if i < 0 else i for i in results]

        yaml_export(np.mean(results), output_dir,  'forward_continuance_ascend', 'scalar')
        
        #  Leg_pull_through_descend

        results=[]

        for i in range(len(P_FS)-1):
            gait_time=P_FS[i+1]-P_FS[i]
            results.append((P_MS[i+1]-P_TO[i+1])/(P_FS[i+1]-P_FS[i]))

        for i in range(len(S_FS)-1):
            gait_time=S_FS[i+1]-S_FS[i]
            results.append((S_MS[i+1]-S_TO[i+1])/(S_FS[i+1]-S_FS[i]))


        results = [0 if i < 0 else i for i in results]

        yaml_export(np.mean(results), output_dir,  'swing_foot_clearance', 'scalar')

        # Foot_placement_descend
        results=[]

        for i in range(len(P_FS)-1):
            gait_time=P_FS[i+1]-P_FS[i]
            results.append((P_FS[i+1]-P_MS[i+1])/(P_FS[i+1]-P_FS[i]))

        for i in range(len(S_FS)-1):
            gait_time=S_FS[i+1]-S_FS[i]
            results.append((S_FS[i+1]-S_MS[i+1])/(S_FS[i+1]-S_FS[i]))


        results = [0 if i < 0 else i for i in results]

        yaml_export(np.mean(results), output_dir,  'swing_foot_placement', 'scalar')


    #Ascend metrics
    if(descend==0):
        yaml_export(np.mean([S_FS[-1]-P_TO[0]]), output_dir,  'total_time_ascend', 'scalar')
        #weight_acceptance_ascend 

        results=[]

        for i in range(len(P_FS)-1):
            gait_time=P_FS[i+1]-P_FS[i]
            results.append((S_TO[i]-P_FS[i])/(P_FS[i+1]-P_FS[i]))

        for i in range(len(S_FS)-1):
            gait_time=S_FS[i+1]-S_FS[i]
            results.append((P_TO[i+1]-S_FS[i])/(S_FS[i+1]-S_FS[i]))


        results = [0 if i < 0 else i for i in results]


        yaml_export(np.mean(results), output_dir,  'weight_acceptance_ascend', 'scalar')

        #pull up 

        results=[]

        for i in range(len(P_FS)-1):
            gait_time=P_FS[i+1]-P_FS[i]
            results.append((S_MS[i]-S_TO[i])/(P_FS[i+1]-P_FS[i]))

        for i in range(len(S_FS)-1):
            gait_time=S_FS[i+1]-S_FS[i]
            results.append((P_MS[i]-P_TO[i])/(S_FS[i+1]-S_FS[i]))


        results = [0 if i < 0 else i for i in results]

        yaml_export(np.mean(results), output_dir,  'pull_up', 'scalar')

        # forward_continuance_ascend

        results=[]

        for i in range(len(P_FS)-1):
            gait_time=P_FS[i+1]-P_FS[i]
            results.append((S_FS[i]-S_MS[i])/(P_FS[i+1]-P_FS[i]))

        for i in range(len(S_FS)-1):
            gait_time=S_FS[i+1]-S_FS[i]
            results.append((P_FS[i]-P_MS[i])/(S_FS[i+1]-S_FS[i]))

        results = [0 if i < 0 else i for i in results]

        yaml_export(np.mean(results), output_dir,  'forward_continuance_ascend', 'scalar')


        # push up

        results=[]

        for i in range(len(P_FS)-1):
            gait_time=P_FS[i+1]-P_FS[i]
            results.append((P_TO[i+1]-S_FS[i])/(P_FS[i+1]-P_FS[i]))

        for i in range(len(S_FS)-1):
            gait_time=S_FS[i+1]-S_FS[i]
            results.append((S_TO[i]-P_FS[i])/(S_FS[i+1]-S_FS[i]))

        results = [0 if i < 0 else i for i in results]

        yaml_export(np.mean(results), output_dir,  'push_up', 'scalar')


        # swing_foot_clearance

        results=[]

        for i in range(len(P_FS)-1):
            gait_time=P_FS[i+1]-P_FS[i]
            results.append((P_MS[i+1]-P_TO[i+1])/(P_FS[i+1]-P_FS[i]))

        for i in range(len(S_FS)-1):
            gait_time=S_FS[i+1]-S_FS[i]
            results.append((S_MS[i+1]-S_TO[i+1])/(S_FS[i+1]-S_FS[i]))

        results = [0 if i < 0 else i for i in results]

        yaml_export(np.mean(results), output_dir,  'swing_foot_clearance', 'scalar')

        # swing_foot_placement
        results=[]

        for i in range(len(P_FS)-1):
            gait_time=P_FS[i+1]-P_FS[i]
            results.append((P_FS[i+1]-P_MS[i+1])/(P_FS[i+1]-P_FS[i]))

        for i in range(len(S_FS)-1):
            gait_time=S_FS[i+1]-S_FS[i]
            results.append((S_FS[i+1]-S_MS[i+1])/(S_FS[i+1]-S_FS[i]))


        results = [0 if i < 0 else i for i in results]

        yaml_export(np.mean(results), output_dir,  'swing_foot_placement', 'scalar')


    #Readingh again the original file
    test=pd.read_csv(file_in)

    #Calculate the metrics of maximum, minimum, mean and std of various variables
    hips_max, hips_min, hips_mean, hips_std = PI_Sensors_Hip(test)
    hips_vector = np.array([hips_max, hips_min, hips_mean, hips_std])
    yaml_export(hips_vector, output_dir,'hips_angle', 'vector')

    lul_max, lul_min, lul_mean, lul_std = PI_Sensors_LUL(test)
    lul_vector = np.array([lul_max, lul_min, lul_mean, lul_std])
    yaml_export(lul_vector, output_dir,'lul_angle', 'vector')

    lk_max, lk_min, lk_mean, lk_std = PI_Sensors_LK(test)
    lk_vector = np.array([lk_max, lk_min, lk_mean, lk_std])
    yaml_export(lk_vector, output_dir,'lk_angle', 'vector')

    lf_max, lf_min, lf_mean, lf_std = PI_Sensors_LF(test)
    lf_vector = np.array([lf_max, lf_min, lf_mean, lf_std])
    yaml_export(lf_vector, output_dir,'lf_angle', 'vector')


    rul_max, rul_min, rul_mean, rul_std = PI_Sensors_RUL(test)
    rul_vector = np.array([rul_max, rul_min, rul_mean, rul_std])
    yaml_export(rul_vector, output_dir,'rul_angle', 'vector')

    rk_max, rk_min, rk_mean, rk_std = PI_Sensors_RK(test)
    rk_vector = np.array([rk_max, rk_min, rk_mean, rk_std])
    yaml_export(rk_vector, output_dir,'rk_angle', 'vector')

    rf_max, rf_min, rf_mean, rf_std = PI_Sensors_RF(test)
    rf_vector = np.array([rf_max, rf_min, rf_mean, rf_std])
    yaml_export(rf_vector, output_dir,'rf_angle', 'vector')








