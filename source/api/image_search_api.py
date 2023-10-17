import flask
from flask import jsonify, request
import os 
import numpy as np
from tqdm import tqdm
import sys
import faiss
import requests
from io import BytesIO
sys.path.append("/workspace/competitions/AIC_2023/SIU_Minerva")
from models.clip_l14 import CLIP14


# Đường dẫn lưu file feature của clip model
CLIP_FEATURES_PATH= ['/dataset/AIC2023/minerva_dataset/edited/0/features/clip_l14/',
                     '/dataset/AIC2023/minerva_dataset/edited/1/features/clip_l14/',
                     '/dataset/AIC2023/minerva_dataset/edited/2/features/clip_l14/']
KEYFRAME_FOLDER_PATH = "/dataset/AIC2023/minerva_dataset/edited/"

model = CLIP14()

# Function load extracted feature into faiss    
def indexing_methods_faiss(clip_features_path):
    faiss_db = faiss.IndexFlatL2(768)
    db = []
    for idx_folder, folder_path in enumerate(clip_features_path):
        # print(folder_path)
        for feat_npy in tqdm(os.listdir(folder_path)):
            # print(feat_npy)
            video_name = feat_npy.split('.')[0]
            feats_arr = np.load(os.path.join(folder_path, feat_npy))
            for idx, feat in enumerate(feats_arr):
            #Lưu mỗi records với 3 trường thông tin là video_name, keyframe_id, feature_of_keyframes
                instance = (video_name, idx, idx_folder)
                db.append(instance)
                faiss_db.add(feat.reshape(1,-1).astype('float32'))
    db = dict(enumerate(db))
    return db, faiss_db

def preprocess_image(image_url):
    global model
    if "image//" in image_url:
        image_url = "/" + image_url.split("image//")[1]
    image_url = image_url.lstrip()
    if (image_url[0:4]=="http"):
        print(image_url)
        # if "image//" in image_url:
        #     image_url = "/" + image_url.split("image//")[1]
        response = requests.get(image_url)
        image_path = BytesIO(response.content)
    else:
        image_path=image_url
    img_vector = model.get_image_features(image_path)
    return img_vector

def transform_result(I, D):
    search_results = []
    for instance in zip(I[0],D[0]):
        ins_id, distance = instance
        video_name, idx, idx_folder = db[ins_id]
        
        print(video_name, idx, idx_folder)
        frames_folder = KEYFRAME_FOLDER_PATH + str(idx_folder) + "/frames" + "/Keyframes_" + str(video_name.split('_')[0]) +'/keyframes/'+ video_name
        
        keyframe_id = sorted(os.listdir(frames_folder))[idx].split('.')[0]
        
        video_name = video_name + '.mp4'
        
        result = {"idx_folder": str(idx_folder), "video_name":str(video_name),
                                "keyframe_id": str(keyframe_id),
                                "score": str(distance)}
        print("result: ", result)
        search_results.append(result)
    return search_results


# Gọi lên chạy 
app = flask.Flask("Image Search")
app.config["DEBUG"] = True

# Load faisss 
db, faiss_db = indexing_methods_faiss(CLIP_FEATURES_PATH)

@app.route('/predict', methods=['POST', 'GET'])
def updateCurrentCode():
    global KEYFRAME_FOLDER_PATH
    image_url = ""
    if request.method == "POST":
        image_url = request.json['image_url']

    else:
        image_url = request.args.get('image_url')

    # init database 
    global db, faiss_db

    # preprocessing image
    img_vector = preprocess_image(image_url)

    D, I = faiss_db.search(img_vector, k=200)
    
    search_results = transform_result(I, D)
    response = flask.jsonify(search_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  


#preprocess vector đặc trưng của image (truyền vào đường dẫn của image)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8403, debug=False)