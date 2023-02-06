import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", help="Folder name", type=str, default="part00010")
    parser.add_argument("--csv_1_path", "-csv_1", help="csv1 file path", type=str, default="./part00010_0DLC_resnet50_Yolo_1000_trainAug7shuffle1_150000.csv")
    parser.add_argument("--csv_2_path", "-ncsv_2", help="csv2 file path", type=str, default="./part00010_1DLC_resnet50_Yolo_1000_trainAug7shuffle1_150000.csv")
    args = parser.parse_args()
    return args

args = parse_args()
likelihood = 0.8
img_root_0 = './yolov5-master/runs/detect/' + str(args.name) + '_0/crops/Joint_1/'
img_root_1 = './yolov5-master/runs/detect/' + str(args.name) + '_1/crops/Joint_1/'

dir_list_0 = os.listdir(img_root_0)
dir_list_1 = os.listdir(img_root_1)

dir_list = list(set(dir_list_0).intersection(set(dir_list_1)))

def bubbleSort(arr):
    n = len(arr)
    # Iterate over all array elements
    for i in tqdm(range(n)):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if (int(arr[j]) - int(arr[j + 1 ])) > 0:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
print('bubbleSort')
dir_list = bubbleSort(dir_list)
#print(dir_list)
#print(dir_list[0:8])

df_0 = pd.read_csv(str(args.csv_2_path))
df_1 = pd.read_csv(str(args.csv_2_path))

df_arr_0 = df_0.to_numpy()
df_arr_1 = df_1.to_numpy()
dir_arr = np.array(dir_list)#.reshape(len(dir_list),1)
output_arr = np.ones((len(dir_list),13))

for i in range(len(dir_list)): #range(len(dir_list)): #range(8):
    txt_root_0 = img_root_0 + str(dir_list[i]) + '/output.txt'
    txt_root_1 = img_root_1 + str(dir_list[i]) + '/output.txt'
    txt_file_0 = open(txt_root_0,'r')
    txt_file_1 = open(txt_root_1,'r')
    file_content_0=txt_file_0.read()
    txt_file_0.close()
    file_content_1=txt_file_1.read()
    txt_file_1.close()

    txt_list_0 = file_content_0.split('_')
    txt_list_1 = file_content_1.split('_')

    Top_lef_0_x, Bottom_rig_0_x = txt_list_0[0], txt_list_0[2]
    Top_lef_0_y, Bottom_rig_0_y = txt_list_0[1], txt_list_0[3]
    Top_lef_1_x, Bottom_rig_1_x = txt_list_1[0], txt_list_1[2]
    Top_lef_1_y, Bottom_rig_1_y = txt_list_1[1], txt_list_1[3]
    
    #print(Top_lef_0_x)
    #Top_lef_0_x = float(Top_lef_0_x) #- float(Bottom_rig_0_x) #- float(Top_lef_0_x)) #* 0.1
    #print(Top_lef_0_x)
    #Top_lef_1_x = float(Top_lef_1_x) #- float(Bottom_rig_1_x) #- float(Top_lef_1_x)) * 0.1
    #Top_lef_0_y = float(Top_lef_0_y) #- float(Bottom_rig_0_y) #- float(Top_lef_0_y)) * 0.1

    n_x = [1,4,7,10]
    n_y = [2,5,8,11]
    output_arr[i,0] = int(dir_arr[i])
    for num_x in n_x:
        output_arr[i,num_x] = float(df_arr_0[i+2,num_x]) +  float(Top_lef_0_x)
        output_arr[i,num_x+1] = float(df_arr_1[i+2,num_x]) +  float(Top_lef_1_x)
    for num_y in n_y:
        output_arr[i,num_y+1] = float(df_arr_0[i+2,num_y]) + float(Top_lef_0_y)


#print(output_arr.shape)
output_df = pd.DataFrame(data=output_arr,columns=['frame','left_tip_x','left_tip_x','left_tip_y','right_tip_x','right_tip_x','right_tip_y','pitch1_x','pitch1_x','pitch1_y','pitch2_x','pitch2_x','pitch2_y'])
#print(output_df.shape)
path_output = './Output/output.csv'
if (os.path.exists(path_output)) :
    os.remove(path_output)
output_df.to_csv(path_output)

############################################################
left_tip_arr = np.full([len(dir_list),6], np.nan) #np.ones((len(dir_list),4))
right_tip_arr = np.full([len(dir_list),6], np.nan) #np.ones((len(dir_list),4))
pitch1_arr = np.full([len(dir_list),6], np.nan) #np.ones((len(dir_list),4))
pitch2_arr = np.full([len(dir_list),6], np.nan) #np.ones((len(dir_list),4))

left_tip_col = 0
right_tip_col = 0
pitch1_col = 0
pitch2_col = 0

for i in range(len(dir_list)):
    txt_root_0 = img_root_0 + str(dir_list[i]) + '/output.txt'
    txt_root_1 = img_root_1 + str(dir_list[i]) + '/output.txt'
    txt_file_0 = open(txt_root_0,'r')
    txt_file_1 = open(txt_root_1,'r')
    file_content_0=txt_file_0.read()
    txt_file_0.close()
    file_content_1=txt_file_1.read()
    txt_file_1.close()

    txt_list_0 = file_content_0.split('_')
    txt_list_1 = file_content_1.split('_')

    Top_lef_0_x, Bottom_rig_0_x = txt_list_0[0], txt_list_0[2]
    Top_lef_0_y, Bottom_rig_0_y = txt_list_0[1], txt_list_0[3]
    Top_lef_1_x, Bottom_rig_1_x = txt_list_1[0], txt_list_1[2]
    Top_lef_1_y, Bottom_rig_1_y = txt_list_1[1], txt_list_1[3]

    #Top_lef_0_x = float(Top_lef_0_x) #- float(Bottom_rig_0_x) #- float(Top_lef_0_x)) * 0.1
    #Top_lef_1_x = float(Top_lef_1_x) #- float(Bottom_rig_1_x) #- float(Top_lef_1_x)) * 0.1
    #Top_lef_0_y = float(Top_lef_0_y) #- float(Bottom_rig_0_y) #- float(Top_lef_0_y)) * 0.1

    #n_x = [1,4,7,10]
    #n_y = [2,5,8,11]
    #############################################################################
    if (float(df_arr_0[i+2,3]) > likelihood and float(df_arr_1[i+2,3]) > likelihood):
        left_tip_arr[left_tip_col,0] = int(dir_arr[i])
        #for num_x in n_x:
        left_tip_arr[left_tip_col,1] = float(df_arr_0[i+2,1]) +  float(Top_lef_0_x)
        left_tip_arr[left_tip_col,2] = float(df_arr_1[i+2,1]) +  float(Top_lef_1_x)
        #for num_y in n_y:
        left_tip_arr[left_tip_col,3] = float(df_arr_0[i+2,2]) + float(Top_lef_0_y)
        left_tip_arr[left_tip_col,4] = float(df_arr_1[i+2,2]) + float(Top_lef_1_y)
        left_tip_arr[left_tip_col,5] = np.abs(left_tip_arr[left_tip_col,3] - left_tip_arr[left_tip_col,4])
        left_tip_col += 1
        #############################################################################
    #############################################################################
    if (float(df_arr_0[i+2,6]) > likelihood and float(df_arr_1[i+2,6]) > likelihood):
        right_tip_arr[right_tip_col,0] = int(dir_arr[i])
        #for num_x in n_x:
        right_tip_arr[right_tip_col,1] = float(df_arr_0[i+2,4]) +  float(Top_lef_0_x)
        right_tip_arr[right_tip_col,2] = float(df_arr_1[i+2,4]) +  float(Top_lef_1_x)
        #for num_y in n_y:
        right_tip_arr[right_tip_col,3] = float(df_arr_0[i+2,5]) + float(Top_lef_0_y)
        right_tip_arr[right_tip_col,4] = float(df_arr_0[i+2,5]) + float(Top_lef_1_y)
        right_tip_arr[right_tip_col,5] = np.abs(right_tip_arr[right_tip_col,3]-right_tip_arr[right_tip_col,4])
        right_tip_col += 1
        #############################################################################
    #############################################################################
    '''
    if (float(df_arr_0[i+2,9]) > likelihood and float(df_arr_1[i+2,9]) > likelihood) :#and ((np.abs(pitch1_arr[pitch1_col,3]-pitch1_arr[pitch1_col,4]))<5):
        pitch1_arr[pitch1_col,0] = int(dir_arr[i])
        #for num_x in n_x:
        pitch1_arr[pitch1_col,1] = float(df_arr_0[i+2,7]) +  float(Top_lef_0_x)
        pitch1_arr[pitch1_col,2] = float(df_arr_1[i+2,7]) +  float(Top_lef_1_x)
        #for num_y in n_y:
        pitch1_arr[pitch1_col,3] = float(df_arr_0[i+2,8]) + float(Top_lef_0_y)
        pitch1_arr[pitch1_col,4] = float(df_arr_0[i+2,8]) + float(Top_lef_1_y)
        pitch1_arr[pitch1_col,5] = np.abs(pitch1_arr[pitch1_col,3]-pitch1_arr[pitch1_col,4])
        pitch1_col += 1
    '''
    if (float(df_arr_0[i+2,9]) > likelihood and float(df_arr_1[i+2,9]) > likelihood):
        pitch1_ly = float(df_arr_0[i + 2, 8]) + float(Top_lef_0_y)
        pitch1_ry = float(df_arr_0[i + 2, 8]) + float(Top_lef_1_y)
        pitch1_error = np.abs(pitch1_ly - pitch1_ry)
        #print(aa)
        if (pitch1_error<5):
            pitch1_arr[pitch1_col, 0] = int(dir_arr[i])
            # for num_x in n_x:
            pitch1_arr[pitch1_col, 1] = float(df_arr_0[i + 2, 7]) + float(Top_lef_0_x)
            pitch1_arr[pitch1_col, 2] = float(df_arr_1[i + 2, 7]) + float(Top_lef_1_x)
            # for num_y in n_y:
            pitch1_arr[pitch1_col, 3] = pitch1_ly
            pitch1_arr[pitch1_col, 4] = pitch1_ry
            pitch1_arr[pitch1_col, 5] = pitch1_error
            pitch1_col += 1
        #############################################################################
    #############################################################################
    if (float(df_arr_0[i+2,12]) > likelihood and float(df_arr_1[i+2,12]) > likelihood):
        pitch2_arr[pitch2_col,0] = int(dir_arr[i])
        #for num_x in n_x:
        pitch2_arr[pitch2_col,1] = float(df_arr_0[i+2,10]) +  float(Top_lef_0_x)
        pitch2_arr[pitch2_col,2] = float(df_arr_1[i+2,10]) +  float(Top_lef_1_x)
        #for num_y in n_y:
        pitch2_arr[pitch2_col,3] = float(df_arr_0[i+2,11]) + float(Top_lef_0_y)
        pitch2_arr[pitch2_col,4] = float(df_arr_0[i+2,11]) + float(Top_lef_1_y)
        pitch2_arr[pitch2_col,5] = np.abs(pitch2_arr[pitch2_col,3]-pitch2_arr[pitch2_col,4])
        pitch2_col += 1
        #############################################################################

#############################################################################
left_tip_df = pd.DataFrame(data=left_tip_arr,columns=['frame','left_tip_Lx','left_tip_Rx','left_tip_Ly','left_tip_Ry','error'])
left_tip_df.dropna(axis=0, how='any', subset=None, inplace=True)
path_left_tip = './Output/left_tip.csv'
if (os.path.exists(path_left_tip)) :
    os.remove(path_left_tip)
left_tip_df.to_csv(path_left_tip)
#############################################################################
right_tip_df = pd.DataFrame(data=right_tip_arr,columns=['frame','right_tip_Lx','right_tip_Rx','right_tip_Ly','right_tip_Ry','error'])
right_tip_df.dropna(axis=0, how='any', subset=None, inplace=True)
path_right_tip = './Output/right_tip.csv'
if (os.path.exists(path_right_tip)) :
    os.remove(path_right_tip)
right_tip_df.to_csv(path_right_tip)
#############################################################################
pitch1_df = pd.DataFrame(data=pitch1_arr,columns=['frame','pitch1_Lx','pitch1_Rx','pitch1_Ly','pitch1_Ry','error'])
pitch1_df.dropna(axis=0, how='any', subset=None, inplace=True)
path_pitch1 = './Output/pitch1.csv'
if (os.path.exists(path_pitch1)) :
    os.remove(path_pitch1)
pitch1_df.to_csv(path_pitch1)
#############################################################################
pitch2_df = pd.DataFrame(data=pitch2_arr,columns=['frame','pitch2_Lx','pitch2_Rx','pitch2_Ly','pitch2_Ry','error'])
pitch2_df.dropna(axis=0, how='any', subset=None, inplace=True)
path_pitch2= './Output/pitch2.csv'
if (os.path.exists(path_pitch2)) :
    os.remove(path_pitch2)
pitch2_df.to_csv(path_pitch2)
#############################################################################

