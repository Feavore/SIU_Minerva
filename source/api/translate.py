import sys
import flask
from flask import jsonify, request
sys.path.append("/workspace/competitions/AIC_2023/SIU_Minerva")
from models.vi2en import Translator 

translator = Translator()

def TextTranslate(src_text):
    translation = translator.translate_vi2en(src_text)
    return translation

app = flask.Flask("API Translate")
app.config["DEBUG"] = True

@app.route('/predict', methods=['POST', 'GET'])
def TranslateAPI():
    text = ""
    if request.method == "POST":
        text = request.json['text']
    else:
        text = request.args.get('text')
        
    translated_txt = TextTranslate(text)
    
    response = flask.jsonify(translated_txt)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8410, debug=False)