import torch
from torch import cuda
from transformers import AutoModelWithLMHead, AutoTokenizer,T5Tokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

def summarize(text, max_length=150):
  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

  generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
  return preds[0]

class Summarizer:
  def __init__(self,model_weight_path,max_length=150):
    self.device='cuda' if cuda.is_available() else 'cpu'
    #self.tokenizer=T5Tokenizer.from_pretrained('t5-base')
    self.tokenizer=AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
    self.MAX_SEQ_LEN=max_length
    self.model=AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
    #self.model.to(self.device)

  def encode_reviews(self, reviews):
    input_ids = self.tokenizer.encode(reviews, return_tensors="pt", add_special_tokens=True)
    generated_ids = self.model.generate(input_ids=input_ids, num_beams=2, max_length=self.MAX_SEQ_LEN,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
    return generated_ids

  def predict(self, reviews):
    encoded_output=self.encode_reviews(reviews)
    summarized_result=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in encoded_output]
    return summarized_result[0]
