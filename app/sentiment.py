import torch
from torch import cuda
from transformers import XLMRobertaForSequenceClassification
from transformers import XLMRobertaTokenizer

class SentimentPredictor:
  def __init__(self,model_weight_path,max_length=64):
    self.device='cuda' if cuda.is_available() else 'cpu'
    self.tokenizer=XLMRobertaTokenizer.from_pretrained(model_weight_path)
    self.MAX_SEQ_LEN=max_length
    self.model=XLMRobertaForSequenceClassification.from_pretrained(model_weight_path)
    #self.model.to(self.device)

  def encode_reviews(self, reviews):
    encode=self.tokenizer.encode_plus(
                        reviews,
                        add_special_tokens = True,
                        max_length = self.MAX_SEQ_LEN,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True
                    )
    #encode.to(self.device)
    return encode

  def predict(self, reviews):
    encoded_output=self.encode_reviews(reviews)
    result=self.model(input_ids=encoded_output['input_ids'],attention_mask=encoded_output['attention_mask'])
    _,prediction=torch.max(result[0],dim=1)
    output=prediction.item()
    sentiment='Positive' if output==1 else 'Negative'
    return sentiment
