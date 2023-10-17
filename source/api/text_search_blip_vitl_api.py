import flask
from flask import jsonify, request
from datetime import datetime
import os 
import numpy as np
from tqdm import tqdm
from PIL import Image
from time import  time
import sys 
import faiss
sys.path.append("/workspace/competitions/AIC_2023/SIU_Minerva")
from models.blip2 import BLIP2ViTL

# Đường dẫn lưu file feature của clip model
FEATURES_PATH= ['/dataset/AIC2023/minerva_dataset/edited/0/features/blip_vitl/',
                     '/dataset/AIC2023/minerva_dataset/edited/1/features/blip_vitl/',
                     '/dataset/AIC2023/minerva_dataset/edited/2/features/blip_vitl/']


KEYFRAME_FOLDER_PATH = "/dataset/AIC2023/minerva_dataset/edited/"

model = BLIP2ViTL()

# Function load extracted feature into faiss    
def indexing_methods_faiss(clip_features_path):
    faiss_db = faiss.IndexFlatL2(256)
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

def preprocessing_text(text):
    global model
    text_feat_arr = model.get_text_features(text) 
    text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32') #=> float32
    return text_feat_arr


def transform_result(I, D):
    search_results = []
    for instance in zip(I[0],D[0]):
        ins_id, distance = instance
        video_name, idx, idx_folder = db[ins_id]
        
        print(video_name, idx, idx_folder)
        frames_folder = KEYFRAME_FOLDER_PATH + str(idx_folder) + "/thumbnails" + "/Keyframes_" + str(video_name.split('_')[0]) +'/keyframes/'+ video_name
        print(frames_folder)
        
        # Modify flag
        if os.path.exists(frames_folder):
            keyframe_id = sorted(os.listdir(frames_folder))[idx].split('.')[0]
            video_name = video_name + '.mp4'
            result = {"idx_folder": str(idx_folder),"video_name":str(video_name),
                                    "keyframe_id": str(keyframe_id),
                                    "score": str(distance)}
            print("result: ", result)
            search_results.append(result)
    return search_results

# Gọi lên chạy 
app = flask.Flask("API Text Search")
app.config["DEBUG"] = True
# Load faisss 
db, faiss_db = indexing_methods_faiss(FEATURES_PATH)
# text_embedd = model.get_text_features()
@app.route('/predict', methods=['POST', 'GET'])

def updateCurrentCode():
    global KEYFRAME_FOLDER_PATH
    text ="" #câu truy vấn => duong dan anh

    if request.method == "POST":
        text = request.json['text']
    else:
        text = request.args.get('text')
    
    # init database 
    global db, faiss_db
    # preprocessing text 
    
    print(f"text: {text}")

    text_feat_arr = preprocessing_text(text)
    D, I = faiss_db.search(text_feat_arr, k=200)

    search_results = transform_result(I, D)  
    response = flask.jsonify(search_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8404, debug=False)