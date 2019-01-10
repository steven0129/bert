import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')
model.eval()

# Tokenized input
text1 = '紅印花加蓋暫作郵票之原票，是由英國倫敦華德路公司於1896年9月採用印製鈔票同等級的雕刻凹版技術精印。'
text2 = '為了搭配紅印花精美幾何網紋的印刷，郵票紙質是一種無水印的厚白洋紙。'

tokenized_text = tokenizer.tokenize(f'[CLS] {text1} [SEP] {text2} [SEP]')
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

predictions = model(torch.LongTensor([indexed_tokens]))
print(predictions)