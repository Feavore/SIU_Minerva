import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
sys.path.append("/workspace/competitions/AIC_2023/SIU_Minerva/models")
from models.clip_l14 import CLIP14


# Locate directory
KEYFRAME_DIR = '/dataset/AIC2023/minerva_dataset/edited/2/frames'
FEATURE_DIR = '/dataset/AIC2023/minerva_dataset/edited/2/features/clip_l14'

# Initialize the model
model = CLIP14()

def process(keyframe_folder):
    _, folders, _ = next(os.walk(keyframe_folder))
    for folder in folders:
        data = []
        lst_image = []
        folder_name = folder
        folder = os.path.join(keyframe_folder, folder)
        lst_image = sorted(os.listdir(folder))
        for image in tqdm(lst_image, desc='Extracting {}'.format(folder_name)):
            image_path = os.path.join(folder, image)
            feature = model.get_image_features(image_path)
            data.append(feature)
        data = np.array(data)
        # Save the data array in .npy file 
        np.save(os.path.join(FEATURE_DIR, folder_name), data)
        
def main():
    lst_dir = os.listdir(KEYFRAME_DIR)
    for folder in lst_dir:
        print("Processing {}".format(folder))
        folder = os.path.join(KEYFRAME_DIR, folder, 'keyframes')
        process(keyframe_folder=folder)
        
if __name__ == "__main__":
    main()
    print("Saved in {}".format(FEATURE_DIR))