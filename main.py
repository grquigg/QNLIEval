from load_models import loadStructBERT, loadALBERT, loadMT_DNN
import pandas as pd
import tokenization
import torch
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
        print(tokens_a)
        print(tokens_b)
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
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

def tokenizerFn(data, tokenizer, model, prompt, max_seq_len):
    seqs = convert_to_seq(data, prompt)
    if(model == "albert"):
        inputs = tokenizer(seqs, return_tensors="pt", padding=True)
    elif(model == "structBERT"):
        input = tokenizeForStructBERT(data, tokenizer, max_seq_len)
    return input

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
    QNLI_PATH = "QNLI/QNLI/dev.tsv"
    BATCH_SIZE = 8
    MAX_SEQ_LEN = 256
    PROMPT = "[CLS] {} [SEP] {} [SEP]"
    data, labels = load_data(QNLI_PATH)
    structBert, tokenizerBERT = loadStructBERT("config.json", "vocab.txt", "pytorch_model.bin")
    ALBERT, tokenizerALBERT = loadALBERT("albert-xxlarge-v2")
    MT_DNN, tokenizerMT_DNN = loadMT_DNN("mt_dnn_large.pt")
    encoding = tokenizerFn(data[0:BATCH_SIZE], tokenizerBERT, "structBERT", PROMPT, MAX_SEQ_LEN)
    label = torch.tensor(labels[0:BATCH_SIZE])
    label = torch.reshape(label, (1,1,BATCH_SIZE))
    logits = structBert(encoding["input_ids"], encoding["segment_ids"], encoding["input_masks"], label, None)
    print(logits)
    raise NotImplementedError()
    encoded_BERT = tokenizerBERT.tokenize(data[0]) #use this for both Mt_DNN and for structBERT
    print(encoded_BERT)
    encoded_ALBERT = tokenizerALBERT(data[0:BATCH_SIZE], return_tensors="pt", padding=True)
    print(encoded_ALBERT["input_ids"].size())
    output_SBERT = structBert(**encoded_BERT)
    output_ALBERT = ALBERT(**encoded_ALBERT)
    print(output_SBERT["pooler_output"].size())
    print(output_SBERT["last_hidden_state"].size())
    print(output_ALBERT)
    #output_MT_DNN = MT_DNN.predict(**encoded_BERT)
