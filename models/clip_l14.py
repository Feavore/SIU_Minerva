from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torch

class CLIP14:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "openai/clip-vit-large-patch14"
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)

    def get_image_features(self, image_path : str) -> np.array:
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()

    def get_text_features(self, text: str) -> np.array:       
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().detach().numpy()
    

# ___Test___
# clip = CLIP14()
# # get image feature
# image_path = "/dataset/AIC2023/pumkin_dataset/0/frames/pyscenedetect/Keyframes_L01/keyframes/L01_V001/00000.jpg"
# image_features = clip.get_image_features(image_path=image_path)
# print(image_features[0][:5])