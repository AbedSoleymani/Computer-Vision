import os
import wget
import zipfile

def generate_dataset():
    original_dir = os.getcwd()
    if os.path.exists(original_dir + "/P1-Facial-Keypoints-Detection/data"):
        print('The data set is already existed!')
    else:
        os.chdir(original_dir + "/P1-Facial-Keypoints-Detection")
        os.system('mkdir ./data')
        os.chdir(original_dir + "/P1-Facial-Keypoints-Detection/data")
        wget.download('https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip')
        with zipfile.ZipFile('./train-test-data.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
        os.chdir(original_dir)
        print('The data set is generated!')