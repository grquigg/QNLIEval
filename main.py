from load_models import loadStructBERT, loadALBERT, loadMT_DNN
import pandas as pd
import tokenization
import torch
from experiments.exp_def import TaskDefs
from mt_dnn.batcher import SingleTaskDataset
LABELS = {"not_entailment": 0, "entailment": 1}

def convert_to_seq(data, prompt):
    seqs = []
    for entry in data:
        seqs.append(prompt.format(entry[0], entry[1]))
    return seqs

def tokenizeForStructBERT(data, tokenizer, max_seq_length):
    input_id = []
    segment_id = []
    input_masks = []
    for entry in data:
        tokens_a = tokenizer.tokenize(entry[0])
        tokens_b = tokenizer.tokenize(entry[1])
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        input_id.append(input_ids)
        segment_id.append(segment_ids)
        input_masks.append(input_mask)
      
    return {"input_ids": torch.tensor(input_id), "segment_ids": torch.tensor(segment_id), "input_masks": torch.tensor(input_masks)}
def tokenizeForMTDNN(data):
    pass
def tokenizerFn(data, tokenizer, model, prompt, max_seq_len):
    seqs = convert_to_seq(data, prompt)
    if(model == "albert"):
        inputs = tokenizer(seqs, return_tensors="pt", padding=True)
    elif(model == "structBERT"):
        inputs = tokenizeForStructBERT(data, tokenizer, max_seq_len)
    elif(model == "mt_dnn"):
        inputs = tokenizer(seqs, return_tensors="pt", padding=True)
    return inputs

def load_data(path):
    df = pd.read_csv(path, delimiter='\t', quoting=3)
    #process input
    #the prefix is "qnli question: {} sentence: {}"
    data = []
    labels = []
    for i in range(len(df)):
        data.append((df.iloc[i]["question"], df.iloc[i]["sentence"]))
        labels.append(LABELS[df.iloc[i]["label"]])
    return data, labels

if __name__ == "__main__":
    QNLI_PATH = "QNLI/QNLI/train.tsv"
    BATCH_SIZE = 4
    MAX_SEQ_LEN = 256
    PROMPT = "[CLS] {} [SEP] {} [SEP]"
    data, labels = load_data(QNLI_PATH)
    print(len(data))
    num_correct = 0
    #structBert, tokenizerBERT = loadStructBERT("config.json", "vocab.txt", "pytorch_model.bin")
    MT_DNN, tokenizerMT_DNN = loadMT_DNN("mt_dnn_large.pt")
    print(MT_DNN)
    for i in range(0, len(data), BATCH_SIZE):
        encoding = tokenizerFn(data[i:i+BATCH_SIZE], tokenizerMT_DNN, "mt_dnn", PROMPT, MAX_SEQ_LEN)
        print(encoding)
        label = torch.tensor(labels[i:i+BATCH_SIZE])
        label = torch.reshape(label, (1,1,len(data[i:i+BATCH_SIZE])))
        task = "qnli"
        task_defs = TaskDefs("experiments/glue/glue_task_gen_def.yml")
        assert task in task_defs._task_type_map
        assert task in task_defs._data_type_map
        assert task in task_defs._metric_meta_map
        prefix = task.split("_")[0]
        task_def = task_defs.get_task_def(prefix)
        print(task_def)
        batch_meta = {
            "task_id": "qnli",
            "task_def": task_def.__dict__,
            "input_len": len(encoding.keys())
        }
        batch_data = [encoding["input_ids"], encoding["token_type_ids"], encoding["attention_mask"]]
        print(batch_meta["input_len"])
        MT_DNN.predict(batch_meta, batch_data)

    #print("Accuracy: {}".format(num_correct.item() / len(data)))
    #encoded_ALBERT = tokenizerFn(data[0:BATCH_SIZE], tokenizerALBERT, "albert", PROMPT, MAX_SEQ_LEN)
    #output_ALBERT = ALBERT(**encoded_ALBERT)
    #albert_predictions = torch.argmax(output_ALBERT.logits, axis=1)
    #print(albert_predictions)
    #output_MT_DNN = tokenizerFn(data[0:BATCH_SIZE], tokenizerMT_DNN, "mt_dnn", PROMPT, MAX_SEQ_LEN)
    #print(output_MT_DNN)
    #task = "qnli"
    #task_defs = TaskDefs("experiments/glue/glue_task_gen_def.yml")
    #assert task in task_defs._task_type_map
    #assert task in task_defs._data_type_map
    #assert task in task_defs._metric_meta_map
    #prefix = task.split("_")[0]
    #task_def = task_defs.get_task_def(prefix)
    #print(task_def)   
    #batch_meta = {
    #    "task_id": "qnli",
    #    "task_def": task_def.__dict__,
    #    "input_len": len(output_MT_DNN.keys()) 
    #}
    #batch_data = [output_MT_DNN["input_ids"], output_MT_DNN["token_type_ids"], output_MT_DNN["attention_mask"]]
    #print(batch_meta["input_len"])
    #MT_DNN.predict(batch_meta, batch_data)
