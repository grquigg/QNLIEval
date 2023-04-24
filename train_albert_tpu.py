import transformers
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd
import numpy as np
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import torch.optim as optim
from transformers import AlbertTokenizer, AlbertForSequenceClassification

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

SERIAL_EXEC = xmp.MpSerialExecutor()

QNLI_PATH = "QNLI/QNLI/train.tsv"
DEV = "QNLI/QNLI/dev.tsv"
OUTPUT_DIR = "ALBERT/best_bert_model.bin"
BATCH_SIZE = 8
NUM_WORKERS = 1
MAX_SEQ_LEN = 256
LOG_STEPS = 20
EPOCHS = 4
learning_rate = 1e-5
PROMPT = "[CLS] {} [SEP] {} [SEP]"
model_name = "albert-xxlarge-v2"
tokenizerALBERT = AlbertTokenizer.from_pretrained(model_name)
WRAPPED_MODEL = xmp.MpModelWrapper(AlbertForSequenceClassification.from_pretrained(model_name))
data, labels = load_data(QNLI_PATH)
dev_data, dev_labels = load_data(DEV)
def train():
  def get_dataset():
    trainData = QNLI(data, labels, tokenizerALBERT, convert_to_seq, PROMPT)
    devData = QNLI(dev_data, dev_labels, tokenizerALBERT, convert_to_seq, PROMPT)
    return trainData, devData

  train_dataset, test_dataset = SERIAL_EXEC.run(get_dataset)

  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=True)
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=BATCH_SIZE,
      sampler=train_sampler,
      num_workers=NUM_WORKERS)
  test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=NUM_WORKERS)

  lr = learning_rate * xm.xrt_world_size()

  device = xm.xla_device()
  ALBERT = WRAPPED_MODEL.to(device)
  optimizer = optim.Adam(ALBERT.parameters(), lr= lr)
  total_steps = len(train_dataset) * EPOCHS
  scheduler = transformers.get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=1986, num_training_steps=total_steps
  )
  loss_fn = nn.CrossEntropyLoss()

  def train_loop_fn(loader):
      tracker = xm.RateTracker()
      ALBERT.train()
      num_correct = 0
      count = 0
      for x, d in enumerate(loader):
          batch, _ , encoding = d[0]["input_ids"].size()
          input_ids = d[0]["input_ids"].view(batch, encoding)
          attention_mask = d[0]["attention_mask"].view(batch, encoding)
          labels = d[1]
          targets = d[2]
          output = ALBERT(input_ids=input_ids, attention_mask=attention_mask)
          predictions = torch.argmax(output.logits, axis=1)
          loss = loss_fn(output.logits, targets)
          loss.backward()
          nn.utils.clip_grad_norm_(ALBERT.parameters(), max_norm=1.0)
          xm.optimizer_step(optimizer)
          scheduler.step()
          optimizer.zero_grad()
          num_correct += torch.sum(labels == predictions)
          tracker.add(BATCH_SIZE)
          if count % LOG_STEPS == 0:
              print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                  xm.get_ordinal(), x, loss.item(), tracker.rate(),
                  tracker.global_rate(), time.asctime()), flush=True)
          count += 1
  def test_loop_fn(loader):
      total_samples = 0
      correct = 0
      ALBERT.eval()
      for x, d in tqdm(loader):
          input_ids = d[0]["input_ids"][0]
          attention_mask = d[0]["attention_mask"][0]
          labels = d[1]
          output = ALBERT(input_ids=input_ids, attention_mask=attention_mask)
          predictions = torch.argmax(output.logits, axis=1)
          num_correct += torch.sum(labels == predictions)
          total_samples += input_ids.size()[0]
      accuracy = num_correct.double().item() / total_samples
      print('[xla:{}] Accuracy={:.2f}%'.format(
          xm.get_ordinal(), accuracy), flush=True)
      return accuracy
  accuracy = 0.0
  for epoch in range(EPOCHS):
      para_loader = pl.ParallelLoader(train_loader, [device])
      train_loop_fn(para_loader.per_device_loader(device))
      xm.master_print("Finished training epoch {}".format(epoch))

      para_loader = pl.ParallelLoader(test_loader, [device])
      accuracy = test_loop_fn(para_loader.per_device_loader(device))


def _mp_fn(rank):
  torch.set_default_tensor_type('torch.FloatTensor')
  train()

xmp.spawn(_mp_fn, nprocs=8,
          start_method='fork')
