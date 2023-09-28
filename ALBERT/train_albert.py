import torch
import transformers
from load_models import loadALBERT
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import numpy as np
from tqdm import tqdm
# imports pytorch
import torch

# imports the torch_xla package
import torch_xla
import torch_xla.core.xla_model as xm

tokenizer_fn = lambda tok, text: tok.encode_plus(
    text,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding="max_length",
    max_length=256,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)

LABELS = {"not_entailment": 0, "entailment": 1}

class QNLI(Dataset):
    def __init__(self, data, labels, tokenizer, seqFn, prompt):
         self.data = seqFn(data, prompt)
         self.labels = torch.tensor(labels, dtype=torch.long)
         self.tokenizer = tokenizer
         eye = np.eye(
            2, dtype=np.float64
         )  # An identity matrix to easily switch to and from one-hot encoding.
         self.targets = [eye[int(i)] for i in self.labels]

    def __getitem__(self, item):
         encoding = tokenizer_fn(self.tokenizer, self.data[item])
         return (encoding, self.labels[item], self.targets[item])
    def __len__(self):
         return len(self.data)

def convert_to_seq(data, prompt):
    seqs = []
    for entry in data:
        seqs.append(prompt.format(entry[0], entry[1]))
    return seqs

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
    DEV = "QNLI/QNLI/dev.tsv"
    OUTPUT_DIR = "ALBERT/best_bert_model.bin"
    BATCH_SIZE = 4
    MAX_SEQ_LEN = 256
    EPOCHS = 4
    learning_rate = 1e-5
    device = torch.device("cuda")
    PROMPT = "[CLS] {} [SEP] {} [SEP]"
    data, labels = load_data(QNLI_PATH)
    dev_data, dev_labels = load_data(DEV)
    ALBERT, tokenizerALBERT = loadALBERT("albert-xxlarge-v2")
    ALBERT.to(device)

    #datasets and dataloaders
    qnliData = QNLI(data, labels, tokenizerALBERT, convert_to_seq, PROMPT)
    devData = QNLI(dev_data, dev_labels, tokenizerALBERT, convert_to_seq, PROMPT)
    dataloader = DataLoader(qnliData, batch_size=BATCH_SIZE)
    dev_dataloader = DataLoader(devData, batch_size=BATCH_SIZE)
    optimizer = Adam(ALBERT.parameters(), lr= learning_rate)
    total_steps = len(qnliData) * EPOCHS
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    loss = nn.CrossEntropyLoss()
    for i in range(EPOCHS):
        print(f"Running epoch {i}")
        num_correct = 0
        for d in tqdm(dataloader):
            BATCH_SIZE, _ , encoding = d[0]["input_ids"].size()
            input_ids = d[0]["input_ids"].view(BATCH_SIZE, encoding).to(device)
            attention_mask = d[0]["attention_mask"].view(BATCH_SIZE, encoding).to(device)
            labels = d[1].to(device)
            targets = d[2].to(device)
            output = ALBERT(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(output.logits, axis=1)
            l = loss(output.logits, targets)
            l.backward()
            nn.utils.clip_grad_norm_(ALBERT.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            num_correct += torch.sum(labels == predictions)
        print(f"Evaluating model")
        num_correct = 0
        for d_test in tqdm(dev_dataloader):
            input_ids = d[0]["input_ids"][0].to(device)
            attention_mask = d[0]["attention_mask"][0].to(device)
            labels = d[1].to(device)
            output = ALBERT(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(output.logits, axis=1)
            num_correct += torch.sum(labels == predictions)
        accuracy = num_correct.double().item() / len(data)
        print(f"Accuracy: {accuracy}")
   

