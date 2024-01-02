Scalpel Robot Hand-Eye Calibration

Download Yolo_V5 master and add it to the folder

#############################################################
## install 
# install requirements.txt in a Python>=3.7.0 environment, including PyTorch, tensorflow-gpu.

# install tensorflow-gpu=2.10 in cuda 11.3
pip install tensorflow-gpu==2.10
conda install -c conda-forge cudatoolkit

# install pytorch=1.12.1 in cuda 11.3
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# install Deeplabcut
pip install deeplabcut

# install Yolo
pip install -r requirements.txt


#############################################################
## Running programs
1.Put stereo video into ./Video
2.run jupyter notebook
3.run main.ipynb


#############################################################
## train Yolo V5
python ./yolov5-master/train.py --data Jumbo_1000.yaml --weights yolov5s.pt --batch-size 128 --epochs 20000
