import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# 你用的模型（最强中文之一）
MODEL_NAME = "../core_model/Roberta-wwm-ext-large"

# 加载
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# 自动使用 GPU（如果有）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()  # 推理模式


# 提取句子嵌入（最标准：cls 向量）
def get_embedding(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,  # 投诉文本足够用
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 作为句子向量（分类最稳）
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.cpu().numpy().squeeze()


# 测试
if __name__ == "__main__":
    text = "我家网络经常掉线，网速特别慢"
    emb = get_embedding(text)
    print("向量维度:", emb.shape)  # (1024,) large 模型是 1024 维