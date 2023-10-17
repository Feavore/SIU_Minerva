
import cv2
import math
import sys
import requests
import os
import json 
sys.path.append("/workspace/competitions/AIC_2023/SIU_Minerva")
from models.vi2en import Translator
from flask import Flask, request, render_template, send_from_directory

#translator = Translator()

app = Flask(__name__)

true_frame_path = "/dataset/AIC2022/Keyframe_P_JSON"
basepath = "/dataset/AIC2022/"

path_a = "0"
path_b = "KeyFramesC00_V00"
path_c = "C00_V0000"

path_a_lst = ["0", "1"]
path_b_lst = ["KeyFramesC00_V0", "KeyFramesC01_V0", "KeyFramesC02_V0"]
path_c_lst = [str(i).zfill(2) for i in range(0, 100)]  # 00 --> 99

DATASET_PATH_ORIGIN = '/dataset/AIC2023/original_dataset/'
DATASET_PATH_TEAM = '/dataset/AIC2023/minerva_dataset/edited/'

@app.route('/image/<path:filename>')
def download_file(filename):
    print("File name: ", filename)
    directory = "/".join(filename.split("/")[:-1])
    video_name = filename.split("/")[-1]
    print("Dir: ", directory)
    print("Vid_name: ", video_name)
    return send_from_directory(directory="/" + directory, path=video_name)

@app.route('/video/<path:filename>/<path:keyframe>')
def video(filename, keyframe):
    # print("File name: ", filename)
    # print("Keyframe: ",)
    filename = filename + '/' + keyframe
    filename = filename.split("/dataset/")[0]
    video_name = filename.split('/', filename.count("/"))[-1]
    frame_name = keyframe.split('/', keyframe.count("/"))[-1]
    
    # Get video fps
    fps = get_fps(filename)
    
    true_id = int(os.path.splitext(frame_name)[0])
    true_id = int(true_id) / fps
    
    mi = str(int(true_id//60))
    if len(mi)==1: 
        mi="0"+mi
    se = str(int(true_id%60))
    if len(se)==1: 
        se="0"+se
    video_info = video_name + ", " + mi +":"+se + ", " + str(fps)

    print("File name: ", filename)
    directory = "/".join(filename.split("/")[:-1])
    video_name = filename.split("/")[-1]
    print("Dir: ", directory)
    print("Vid_name: ", video_name)
    return send_from_directory(directory="/" + directory, path=video_name)
    #return render_template('video.html', source=filename, keyframe=true_id-1, id=video_info)


def get_fps(video_path):
    video = cv2.VideoCapture("/"+video_path)
    fps = round(video.get(cv2.CAP_PROP_FPS))
    return fps


@app.route('/', methods=['GET', 'POST'])
def index():
    global org2id_dict
    if request.method == 'POST':
        text = request.form['query']
        image = request.form['fname']
        model = request.form['model'] 
        
        if text != "": 
            # API TRANSLATE
            url_text = "http://192.168.1.252:8410/predict?text={}".format(text)
            text = requests.get(url_text).json()

            
            
            url_text = "http://192.168.1.252:{}/predict?text={}".format(int(model),text)
            result = requests.get(url_text).json()
            lst_video_name_text = [(res['idx_folder'], res['video_name'], res['keyframe_id']) for res in result]

        else: 
            lst_video_name_text = []

        if image != "": 
            url_text = "http://192.168.1.252:8403/predict?image_url={}".format(image)
            result = requests.get(url_text).json()
            lst_video_name_img = [(res['idx_folder'], res['video_name'], res['keyframe_id']) for res in result]
        else: 
            lst_video_name_img = []
        
        lst_video_name = lst_video_name_text + lst_video_name_img 
        
        files = []

        for index, info in enumerate(lst_video_name):

            video_path = DATASET_PATH_ORIGIN + str(info[0]) +"/videos/Videos_" + str(info[1]).split("_")[0]  + "/video/" + str(info[1])
            frame_path = DATASET_PATH_TEAM + str(info[0]) +"/thumbnails/Keyframes_" + str(info[1]).split("_")[0]  + "/keyframes/"  + str(info[1].split(".")[0]) + "/" + info[2] + ".jpg"
            fps = get_fps(video_path) 
            frame_name = info[2].replace("'","") + '.jpg'
            # print("Info:", info[0], info[1], info[2])
            # print("Video: ", video_path)
            # print("Frame path: ", frame_path)
            
            true_id = int(os.path.splitext(frame_name)[0])
            time = math.floor(true_id/fps)
            mi = str(time//60)
            if len(mi)==1: 
                mi="0"+mi
            se = str(int(time%60))
            if len(se)==1: 
                se="0"+se
            video_info  = info[1]

            files.append((index, frame_path, frame_name, video_info, video_path, time-1, fps))
        return render_template('web.html', files=files, query=text, image=image, 
                               count=str(len(files)) +' files found.', 
                               model=model)
    else:
        mypath = "/dataset/AIC2022/0/KeyFramesC00_V00/C00_V0000"
        files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        files.sort()    
        scores = [('zero', f) for f in files]

        return render_template('web.html')
    

if __name__ == "__main__":
    org2id_dict_path = "/dataset/AIC2023/original_dataset/0/map_keyframes/true_id_map.json"
    with open(org2id_dict_path) as json_file:
        true_id_map = json.load(json_file)

    app.run("0.0.0.0", port=8402)
 