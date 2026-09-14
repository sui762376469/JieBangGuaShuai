# 功能: 使用 Qwen3-Embedding-0.6B 生成文本向量，
#       再用 逻辑回归 / XGBoost 进行文本分类
# 数据集: THUCNews 10万条中文新闻标题
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
from sentence_transformers import SentenceTransformer
# ──────────────────────────────────────────────
# 第一步：加载数据
# ──────────────────────────────────────────────
DATA_FILE = "../simulated_data_use.txt"
EMBEDDING_SAVE_PATH = "./exist/embeddings_10w.npy"       # 缓存向量，避免重复计算
LABELS_SAVE_PATH = "labels_10w.npy"
TEXTS_SAVE_PATH = "texts_split.npy"               # 记录切分信息

print("=" * 60)
print("【第1步】加载数据...")

labels = []
texts = []

with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 数据格式: "类别\t文本"
        parts = line.split("\t", maxsplit=1)
        if len(parts) == 2:
            labels.append(parts[0])
            texts.append(parts[1])

print(f"  共读取 {len(texts)} 条数据")
print(f"  类别分布:")


# 加载本地模型
model_path = r"..\qwen_model\Qwen3-Embedding-0.6B"
print(f"  正在加载模型: {model_path} ...")
model = SentenceTransformer(model_path,device="cuda")
print(f"  模型加载成功！向量维度: {model.get_sentence_embedding_dimension()}")

print(f"  ⏳ 这一步需要较长时间（10万条数据），请耐心等待...")

start_time = time.time()

# batch_size 根据你的显存调整:
#   显存 8GB  -> batch_size=64
#   显存 12GB -> batch_size=128
#   显存 24GB -> batch_size=256
all_embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,       # 显示进度条
    normalize_embeddings=True     # L2归一化，对分类效果有帮助
)
start_time = time.time()
