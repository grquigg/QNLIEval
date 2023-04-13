import transformers
from mt_dnn.mt_dnn.model import MTDNNModel
from transformers import BertConfig, BertTokenizer, AutoModelForMaskedLM, AutoTokenizer, AlbertTokenizer, AlbertForSequenceClassification, BertModel 
import torch

def loadStructBERT(config_path, model_path, vocab_path):
    config = BertConfig.from_json_file(config_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(vocab_path, config=config)
    return model, tokenizer

def loadMT_DNN(model_path): #this one doesn't have a tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    model = BertModel.from_pretrained("bert-large-uncased")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["state"])
    return model, tokenizer

def loadALBERT(model_name):
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer


