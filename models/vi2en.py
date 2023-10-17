import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self):
        self.tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")
        self.model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")
        self.device_vi2en = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_vi2en.to(self.device_vi2en)
    
    def translate_vi2en(self, vi_texts: str) -> str:
        input_ids = self.tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to(self.device_vi2en)
        output_ids = self.model_vi2en.generate(
            **input_ids,
            decoder_start_token_id=self.tokenizer_vi2en.lang_code_to_id["en_XX"],
            num_return_sequences=1,
            num_beams=3,
            early_stopping=True
        )
        en_texts = self.tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
        return en_texts[0]
    
# ______Test______
'''    
The input may consist of multiple text sequences, 
with the number of text sequences in the input ranging from 1 up to 8, 16, 32, or even higher, 
depending on the GPU memory.
'''
# vi_texts = ["Một thợ lặn đang lặn dưới nước."]
# trans = Translator()
# print(trans.translate_vi2en(vi_texts))
# print(len(vi_texts[0].split()))
