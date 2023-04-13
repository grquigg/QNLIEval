from load_models import loadStructBERT, loadALBERT, loadMT_DNN
import pandas as pd

LABELS = {"not_entailment": 0, "entailment": 1}

def load_data(path):
    df = pd.read_csv(path, delimiter='\t', quoting=3)
    #process input
    #the prefix is "qnli question: {} sentence: {}"
    data = []
    labels = []
    prefix = "qnli question: {} sentence: {}"
    for i in range(len(df)):
        data.append(prefix.format(df.iloc[i]["question"], df.iloc[i]["sentence"]))
        labels.append(LABELS[df.iloc[i]["label"]])
    return data, labels

if __name__ == "__main__":
    QNLI_PATH = "QNLI/QNLI/train.tsv"
    BATCH_SIZE = 16
    data, labels = load_data(QNLI_PATH)
    structBert, tokenizerBERT = loadStructBERT("config.json", ".", ".")
    ALBERT, tokenizerALBERT = loadALBERT("albert-xxlarge-v2")
    MT_DNN, tokenizerMT_DNN = loadMT_DNN("mt_dnn_large.pt")

    print(data[0])
    encoded_BERT = tokenizerBERT(data[0:BATCH_SIZE], return_tensors="pt", padding=True) #use this for both Mt_DNN and for structBERT
    encoded_ALBERT = tokenizerALBERT(data[0:BATCH_SIZE], return_tensors="pt", padding=True)
    print(encoded_BERT["input_ids"].size())
    print(encoded_ALBERT["input_ids"].size())
    output_SBERT = structBert(**encoded_BERT)
    output_ALBERT = ALBERT(**encoded_ALBERT)
    output_MT_DNN = MT_DNN(**encoded_BERT)
