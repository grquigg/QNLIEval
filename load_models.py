import transformers
from mt_dnn.model import MTDNNModel
from experiments.exp_def import TaskDefs, EncoderModelType
from transformers import BertTokenizer, AutoModel, AutoTokenizer, AlbertTokenizer, AlbertForSequenceClassification, BertModel 
from modeling import BertConfig, BertForSequenceClassificationMultiTask
import tokenization
import torch

def loadStructBERT(config_path, vocab_path, model_path):
    bert_config = BertConfig.from_json_file(config_path)
    label_list = [["entailment", "not_entailment"]]
    model = BertForSequenceClassificationMultiTask(bert_config, label_list, "bert")
    new_state_dict = {}
    state_dict = torch.load(model_path, map_location='cuda')
    for key in state_dict:
        if key.startswith('bert.'):
            new_state_dict[key[5:]] = state_dict[key]
        elif key.startswith('module.bert.'):
            new_state_dict[key[12:]] = state_dict[key]
        else:
            pass
    model.bert.load_state_dict(new_state_dict)    
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path)
    return model, tokenizer

def loadMT_DNN(model_path): #this one doesn't have a tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    state_dict = torch.load(model_path)
    #most of this code is borrowed from mt_dnn/predict.py
    task = "qnli"
    task_defs = TaskDefs("experiments/glue/glue_task_gen_def.yml") 
    prefix = task.split("_")[0]
    task_def = task_defs.get_task_def(prefix)
    data_type = task_defs._data_type_map[task]
    task_type = task_defs._task_type_map[task]
    print(task_type)
    metric_meta = task_defs._metric_meta_map[task]
    config = state_dict["config"]

    task_def = task_defs.get_task_def(prefix)
    task_def_list = [task_def]
    config["task_def_list"] = task_def_list
    config["fp16"] = False
    config["answer_opt"] = 0
    config["adv_train"] = False
    config["encoder_type"] = EncoderModelType.BERT
    config["update_bert_opt"] = 1
    config["cuda"] = False
    config["optimizer"] = "sgd"
    config["learning_rate"] = 0.01
    config["weight_decay"] = 1e-2
    config["scheduler_type"] = 3
    config["warmup"] = 1000
    config["local_rank"] = -1
    config["multi_gpu_on"] = False
    model = MTDNNModel(config, state_dict=state_dict, tokenizer=tokenizer)
    encoder_type = config.get("encoder_type", EncoderModelType.BERT)

    return model, tokenizer

def loadALBERT(model_name):
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer


