# BERT

## 事前準備

```
git clone https://github.com/steven0129/bert.git
cd bert
scp steven@140.123.97.228:/data/bert-model .
```

## Docker環境配置

```
sudo docker run --runtime=nvidia --rm -it -e LANG=C.UTF-8 -v /data/bert:/home -p 8097:8097 steven0129/cuda-cudnn-pytorch:9.1-7-1.0.0
```

## 安裝相關python套件

Docker container run起來後，執行以下指令

```
cd /home
pip install -r requirements.txt
cd /
git clone https://github.com/steven0129/pytorch-pretrained-BERT.git
cd pytorch-pretrained-BERT
pip install .
cd /home
```

## 啟動Visdom以監視訓練狀態

```
python -m visdom.server &
```

## 預訓練Generator

```
python pretrain-generator.py
```

## 訓練GAN

```
python main.py
```
