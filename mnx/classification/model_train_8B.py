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
from sklearn.ensemble import RandomForestClassifier
# ──────────────────────────────────────────────
# 第一步：加载数据
# ──────────────────────────────────────────────
DATA_FILE = "../dataset/thucnews_train_10w.txt"
EMBEDDING_SAVE_PATH = "./embeddings_10w1.npy"       # 缓存向量，避免重复计算
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

from collections import Counter
category_counts = Counter(labels)
for cat, count in category_counts.most_common():
    print(f"    {cat}: {count} 条")

# ──────────────────────────────────────────────
# 第二步：标签编码 (把文字标签变成数字 0,1,2...)
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第2步】标签编码...")

label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)
print(f"  共有 {num_classes} 个类别: {list(label_encoder.classes_)}")
print(f"  标签示例: {labels[:5]} -> {label_ids[:5].tolist()}")

# ──────────────────────────────────────────────
# 第三步：划分训练集 / 测试集 (8万 / 2万)
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第3步】划分训练集 / 测试集...")

X_train, X_test, y_train, y_test = train_test_split(
    texts, label_ids,
    test_size=0.2,
    stratify=label_ids,     # 分层抽样，保证每个类别的比例一致
    random_state=42
)

print(f"  训练集: {len(X_train)} 条")
print(f"  测试集: {len(X_test)} 条")

# ──────────────────────────────────────────────
# 第四步：生成 Embedding 向量（GGUF 模型版）
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第4步】使用 Qwen3-Embedding-8B-Q6_K.gguf 生成向量...")

# 检查缓存
if os.path.exists(EMBEDDING_SAVE_PATH):
    print(f"  检测到已有缓存文件 {EMBEDDING_SAVE_PATH}")
    all_embeddings = np.load(EMBEDDING_SAVE_PATH)
    print(f"  直接加载，跳过生成步骤")
else:
    # 安装依赖：pip install llama-cpp-python
    from llama_cpp import Llama

    # ====================== 模型路径（你自己改这里）======================
    MODEL_PATH = r"../qwen_model/Qwen3-Embedding-8B-Q6_K.gguf"  # 改成你实际路径
    # ====================================================================

    print(f"  正在加载 GGUF 模型: {MODEL_PATH} ...")

    # 加载 GGUF Embedding 模型（关键！必须设置 embedding=True）
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=256,          # 文本长度，标题分类足够
        n_gpu_layers=100,   # 全部放显存（8G 完全够）
        embedding=True,     # 必须开启！
        verbose=False,
        n_threads=8         # 根据CPU核心调整
    )

    # 合并文本
    all_texts = X_train + X_test
    print(f"  开始生成 {len(all_texts)} 条文本向量...")

    # Qwen3-Embedding 官方推荐的分类指令前缀（非常重要！提升准确率）
    instruction = "判断下列内容属于以下类别：education, entertainment, finance, game, politics, realty, science, society, sports, stocks。内容："

    start_time = time.time()
    all_embeddings = []

    # 批量生成（速度快，显存友好）
    BATCH_SIZE = 32  # 8G 显存安全值
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i:i+BATCH_SIZE]
        batch_with_prompt = [f"{instruction}{t}" for t in batch_texts]
        
        # 生成向量
        outputs = llm.create_embedding(batch_with_prompt)
        batch_embeddings = [np.array(data["embedding"]) for data in outputs["data"]]
        all_embeddings.extend(batch_embeddings)

        # 进度
        print(f"  已完成：{min(i+BATCH_SIZE, len(all_texts))}/{len(all_texts)}", end="\r")

    all_embeddings = np.array(all_embeddings)
    elapsed = time.time() - start_time

    print(f"\n  ✅ 向量生成完成！耗时: {elapsed:.1f} 秒")
    print(f"  向量形状: {all_embeddings.shape}")

    # 保存缓存
    np.save(EMBEDDING_SAVE_PATH, all_embeddings)
    print(f"  向量已保存到: {EMBEDDING_SAVE_PATH}")

# 切分训练/测试集
split_idx = len(X_train)
X_train_emb = all_embeddings[:split_idx]
X_test_emb = all_embeddings[split_idx:]

print(f"  训练集向量形状: {X_train_emb.shape}")
print(f"  测试集向量形状: {X_test_emb.shape}")