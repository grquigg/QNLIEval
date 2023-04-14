#StructBERT
wget https://raw.githubusercontent.com/alibaba/AliceMind/main/StructBERT/modeling.py
wget https://raw.githubusercontent.com/alibaba/AliceMind/main/StructBERT/tokenization.py
wget https://raw.githubusercontent.com/alibaba/AliceMind/main/StructBERT/config/large_bert_config.json && mv large_bert_config.json config.json
wget https://raw.githubusercontent.com/alibaba/AliceMind/main/StructBERT/config/vocab.txt
wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/en_model && mv en_model pytorch_model.bin
#MT-DNN
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_large.pt

#QNLI dataset
wget https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip
unzip QNLIv2.zip -d QNLI

git clone https://github.com/namisan/mt-dnn.git
mv mt-dnn mt_dnn
mv mt_dnn/* .
