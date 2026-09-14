# %%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# %%
# 1. 配置与模型加载
MODEL_NAME = './Qwen3-Embedding-0.6B'
print(f"正在加载模型: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME,device='cuda')

# %%
# 2. 读取数据
df = pd.read_csv("Full_100.csv")
X_raw = df['text'].tolist()
y = df['label'].values

# %%
# 3. 提取向量特征 (Embedding)
# 为 Qwen-instruct 提供指令前缀，能显著提升分类效果
def get_qwen_embeddings(texts):
    # 针对分类任务的指令
    instruction = "判断相关文本是否属于固定资产类型，以便进行准确的文本分类："
    # 拼接指令
    full_texts = [f"{instruction}{t}" for t in texts]
    return model.encode(full_texts, batch_size=16, show_progress_bar=True)

print("正在提取 100 条数据的向量特征...")
X_embeddings = get_qwen_embeddings(X_raw)

# %%
# 4. 划分数据集 (70% 训练, 30% 验证)
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.3, random_state=42, stratify=y
)

# %%
# 5. 训练分类器 KNN-1
# 使用逻辑回归，并开启 class_weight='balanced' 以处理可能的类别轻微不平衡
clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
clf.fit(X_train, y_train)

# %%
# 6. 验证集评估
y_pred = clf.predict(X_test)
print("\n--- 验证集评估报告 (30条数据) ---")
print(classification_report(y_test, y_pred, target_names=['其他类型 (0)', '固定资产 (1)']))
print(f"总体准确率: {accuracy_score(y_test, y_pred):.2%}")

# %%
# 7. 应用于海量未标注数据 (假设你有一个列表 unlabeled_list)
def predict_new_data(new_texts):
    print(f"正在处理 {len(new_texts)} 条新数据...")
    new_embeddings = get_qwen_embeddings(new_texts)
    preds = clf.predict(new_embeddings)
    # 获取置信度
    probs = clf.predict_proba(new_embeddings).max(axis=1)
    return preds, probs

# %%
# 示例预测
sample_unlabeled = [
    "某某大学办公家具采购合同。办公桌椅。木质家具", 
    "某某局网络安全等级保护测评服务定点采购合同"
]
labels, confidences = predict_new_data(sample_unlabeled)

for txt, lab, conf in zip(sample_unlabeled, labels, confidences):
    print(f"预测结果: {'实物' if lab==1 else '服务'} | 置信度: {conf:.4f} | 文本: {txt[:30]}...")

# %%



