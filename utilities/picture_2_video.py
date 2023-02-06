import os
import cv2
from tqdm import tqdm

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", help="Folder name", type=str, default="part00010")
    args = parser.parse_args()
    return args

def bubbleSort(arr):
    n = len(arr)
    # Iterate over all array elements
    for i in tqdm(range(n)):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if (int(arr[j]) - int(arr[j + 1 ])) > 0:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

args = parse_args()
Video_num = args.name
Video_Name = str(Video_num) 
# Folder path
img_root_0 = './yolov5-master/runs/detect/' + Video_Name + '_0/crops/Joint_1/'
img_root_1 = './yolov5-master/runs/detect/' + Video_Name + '_1/crops/Joint_1/'

# img list
dir_list_0 = os.listdir(img_root_0)
dir_list_1 = os.listdir(img_root_1)
dir_list = list(set(dir_list_0).intersection(set(dir_list_1)))

print('bubbleSort')
dir_list = bubbleSort(dir_list)


fps = 30
# path
file_path='./Video/Yolo/Yolo_output_0.mp4'
size=(1080,720)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4
videoWriter = cv2.VideoWriter(file_path,fourcc,fps,size)

print('frame to video')
for i in tqdm(range(len(dir_list))):
    frame = cv2.imread(img_root_0 + str(dir_list[i]) + '/EndoscopeImageMemory_0_All' + '.jpg')
    img2 = cv2.copyMakeBorder(frame, 0, 720 - frame.shape[0], 0, 1080 - frame.shape[1], cv2.BORDER_CONSTANT,
                              value=[255, 255, 255])
    #cv2.imwrite('./output/'+ str(i) + '.jpg', img2)
    #img3 = cv2.imread('./output/'+ str(i) + '.jpg')
    videoWriter.write(img2)
videoWriter.release() #release


fps = 30
# path
file_path ='./Video/Yolo/Yolo_output_1.mp4'
size = (1080,720)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4
videoWriter = cv2.VideoWriter(file_path,fourcc,fps,size)

print('frame to video')
for i in tqdm(range(len(dir_list))):
    frame = cv2.imread(img_root_1 + str(dir_list[i]) + '/EndoscopeImageMemory_1_All' + '.jpg')
    img2 = cv2.copyMakeBorder(frame, 0, 720 - frame.shape[0], 0, 1080 - frame.shape[1], cv2.BORDER_CONSTANT,
                              value=[255, 255, 255])
    videoWriter.write(img2)
videoWriter.release() #release

