import os
import wget
import zipfile

def generate_dataset():
    original_dir = os.getcwd()
    if os.path.exists(original_dir + "/10_3D-Images/PointNet/data"):
        print('The data set is already existed!')
    else:
        os.chdir(original_dir + "/10_3D-Images/PointNet")
        os.system('mkdir ./data')
        os.chdir(original_dir + "/10_3D-Images/PointNet/data")
        wget.download('http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip')
        with zipfile.ZipFile('./ModelNet10.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
        os.chdir(original_dir)
        print('The data set is generated!')
