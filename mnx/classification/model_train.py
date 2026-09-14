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
# import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
# ──────────────────────────────────────────────
# 第一步：加载数据
# ──────────────────────────────────────────────
DATA_FILE = "../dataset/text_true.txt"
# DATA_FILE = "../output/bm25_api_测试_output.txt"
EMBEDDING_SAVE_PATH = "./embeddings_10w21.npy"       # 缓存向量，避免重复计算
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
        # 数据格式: "文本\t类别"
        parts = line.split("\t", maxsplit=1)
        if len(parts) == 2:
            texts.append(parts[0])
            labels.append(parts[1])

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

print("\n" + "=" * 60)
print("【第2步】标签编码...")

# 注意：虽然标签已是数字(0-16)，但读取时为字符串格式
# LabelEncoder 会将字符串 "0"-"16" 转换为整数 0-16
label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)
print(f"  共有 {num_classes} 个类别: {list(label_encoder.classes_)}")
print(f"  标签示例: {labels[:5]} -> {label_ids[:5].tolist()}")

# 定义类别含义映射（用于后续报告展示）
category_names = {
    0: "非公益诉讼",
    1: "生态环境和资源保护",
    2: "食品药品安全",
    3: "国有财产保护",
    4: "国有土地使用权出让",
    5: "英雄烈士保护",
    6: "未成年人保护",
    7: "安全生产",
    8: "个人信息保护",
    9: "军人地位和权益保障",
    10: "反垄断和反不正当竞争",
    11: "反电信网络诈骗",
    12: "农产品质量安全",
    13: "妇女权益保障",
    14: "无障碍环境建设",
    15: "文物和文化遗产保护",
    16: "国防和军事利益保护"
}
print(f"\n  类别含义:")
for idx, name in category_names.items():
    print(f"    {idx}: {name}")


# ──────────────────────────────────────────────
# 第三步：划分训练集 / 测试集 (8万 / 2万)
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第3步】划分训练集 / 测试集...")

X_train, X_test, y_train, y_test = train_test_split(
    texts, label_ids,
    test_size=0.2,
    # stratify=label_ids,     # 分层抽样，保证每个类别的比例一致
    random_state=42
)

print(f"  训练集: {len(X_train)} 条")
print(f"  测试集: {len(X_test)} 条")

# ──────────────────────────────────────────────
# 第四步：生成 Embedding 向量
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第4步】使用 Qwen3-Embedding-0.6B 生成向量...")

# 检查是否已有缓存的 embedding 文件
# 如果你修改了数据或重新划分，请删除 .npy 文件重新生成
if os.path.exists(EMBEDDING_SAVE_PATH):
    print(f"  检测到已有缓存文件 {EMBEDDING_SAVE_PATH}")
    all_embeddings = np.load(EMBEDDING_SAVE_PATH)
    print(f"  直接加载，跳过生成步骤")
else:
    from sentence_transformers import SentenceTransformer

    # 加载本地模型
    model_path = r"../core_model/Qwen3-Embedding-0.6B"
    print(f"  正在加载模型: {model_path} ...")
    model = SentenceTransformer(model_path,device="cuda")
    print(f"  模型加载成功！向量维度: {model.get_sentence_embedding_dimension()}")

    # 合并训练集+测试集的文本，一起生成 embedding（更快）
    all_texts = X_train + X_test
    print(f"  开始生成 {len(all_texts)} 条文本的向量...")
    print(f"  ⏳ 这一步需要较长时间（10万条数据），请耐心等待...")

    start_time = time.time()

    # batch_size 根据你的显存调整:
    #   显存 8GB  -> batch_size=64
    #   显存 12GB -> batch_size=128
    #   显存 24GB -> batch_size=256
    # all_embeddings = model.encode(
    #     all_texts,
    #     batch_size=1024,
    #     show_progress_bar=True,       # 显示进度条
    #     normalize_embeddings=True     # L2归一化，对分类效果有帮助
    # )

    # 添加检索任务前缀
    instruction = "query:"
    full_texts = [f"{instruction}{t}" for t in all_texts]

    all_embeddings = model.encode(
        full_texts,
        batch_size=16,              # 保持你的值，比16更高效
        show_progress_bar=True,
        normalize_embeddings=True    # 保留！分类任务建议开启
    )

    elapsed = time.time() - start_time
    print(f"  ✅ 向量生成完成！耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"  向量形状: {all_embeddings.shape}")

    # 保存到文件，下次不用重新生成
    np.save(EMBEDDING_SAVE_PATH, all_embeddings)
    print(f"  向量已保存到: {EMBEDDING_SAVE_PATH}")

# 按照之前的切分，把 embedding 也分开
split_idx = len(X_train)
X_train_emb = all_embeddings[:split_idx]
X_test_emb = all_embeddings[split_idx:]

print(f"  训练集向量形状: {X_train_emb.shape}")
print(f"  测试集向量形状: {X_test_emb.shape}")

# ──────────────────────────────────────────────
# 第五步：训练 逻辑回归
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第5步】训练 逻辑回归 (Logistic Regression)...")

lr_model = LogisticRegression(
    max_iter=1000,           # 最大迭代次数
    C=1.0,                   # 正则化强度的倒数，越大越不容易过拟合
    solver="lbfgs",          # 优化算法，适合中小数据集
    n_jobs=-1,               # 使用所有CPU核心
    random_state=42
)

lr_start = time.time()
lr_model.fit(X_train_emb, y_train)
lr_time = time.time() - lr_start

y_pred_lr = lr_model.predict(X_test_emb)
lr_acc = accuracy_score(y_test, y_pred_lr)

print(f"  训练耗时: {lr_time:.2f} 秒")
print(f"  测试集准确率: {lr_acc:.4f} ({lr_acc*100:.2f}%)")

# print("\n" + "=" * 60)
# print("【第5步】训练 随机森林 (Random Forest)...")

# rf_model = RandomForestClassifier(
#     n_estimators=1000,      # 树越多越准
#     max_depth=8,            # 限制深度，防过拟合 → 大幅提升准确率
#     min_samples_split=10,   # 减少过拟合
#     min_samples_leaf=4,     # 让模型更泛化
#     max_features="sqrt",    # 随机选特征，增强多样性
#     bootstrap=True,         
#     oob_score=True,        
#     n_jobs=-1,              
#     random_state=42
# )

# rf_start = time.time()
# rf_model.fit(X_train_emb, y_train)
# rf_time = time.time() - rf_start

# y_pred_rf = rf_model.predict(X_test_emb)
# rf_acc = accuracy_score(y_test, y_pred_rf)

# print(f"  训练耗时: {rf_time:.2f} 秒")
# print(f"  测试集准确率: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

# ──────────────────────────────────────────────
# 第五步：使用 余弦相似度 进行分类 (基于类别质心)
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第5步】基于余弦相似度的分类...")

cos_start = time.time()

from sklearn.metrics.pairwise import cosine_similarity

# ===== 计算每个类别的质心向量 =====
unique_classes = np.unique(y_train)
class_centers = {}          # 存储每个类的中心向量
class_counts = {}           # 每个类的样本数

for cls in unique_classes:
    # 获取属于该类的所有向量
    mask = (y_train == cls)
    class_vectors = X_train_emb[mask]
    
    # 计算该类的平均向量（质心）
    center = np.mean(class_vectors, axis=0)
    # L2归一化（因为后面用余弦相似度）
    center = center / np.linalg.norm(center)
    
    class_centers[cls] = center
    class_counts[cls] = len(class_vectors)

# print(f"  已计算 {len(unique_classes)} 个类别的质心向量")
# for cls in unique_classes:
#     print(f"    类别 {cls} ({label_encoder.inverse_transform([cls])[0]}): {class_counts[cls]} 条")
print(f"  已计算 {len(unique_classes)} 个类别的质心向量")
for cls in unique_classes:
    class_name = category_names.get(cls, f"类别{cls}")
    print(f"    类别 {cls} ({class_name}): {class_counts[cls]} 条")

# ===== 对测试集进行预测 =====
y_pred_cos = []

for test_vec in X_test_emb:
    # 归一化测试向量
    test_vec_norm = test_vec / np.linalg.norm(test_vec)
    
    # 计算与各类别质心的余弦相似度
    similarities = {}
    for cls, center in class_centers.items():
        sim = np.dot(test_vec_norm, center)  # 归一化后点积 = 余弦相似度
        similarities[cls] = sim
    
    # 选相似度最高的类别
    pred_cls = max(similarities.items(), key=lambda x: x[1])[0]
    y_pred_cos.append(pred_cls)

y_pred_cos = np.array(y_pred_cos)
cos_acc = accuracy_score(y_test, y_pred_cos)
cos_time = time.time() - cos_start

print(f"  训练+预测耗时: {cos_time:.2f} 秒")
print(f"  测试集准确率: {cos_acc:.4f} ({cos_acc*100:.2f}%)")


# ──────────────────────────────────────────────
# 第六步：训练 XGBoost
# ──────────────────────────────────────────────

# print("\n" + "=" * 60)
# print("【第6步】训练 XGBoost...")

# xgb_model = xgb.XGBClassifier(
#     n_estimators=300,         # 树的数量
#     max_depth=6,              # 每棵树的最大深度
#     learning_rate=0.1,        # 学习率
#     subsample=0.8,            # 每棵树使用80%的样本
#     colsample_bytree=0.8,     # 每棵树使用80%的特征
#     tree_method="hist",       # 使用histogram方法，训练更快
#     n_jobs=-1,                # 使用所有CPU核心
#     random_state=42,
#     use_label_encoder=False,
#     eval_metric="mlogloss"
# )

# xgb_start = time.time()
# xgb_model.fit(X_train_emb, y_train)
# xgb_time = time.time() - xgb_start

# y_pred_xgb = xgb_model.predict(X_test_emb)
# xgb_acc = accuracy_score(y_test, y_pred_xgb)

# print(f"  训练耗时: {xgb_time:.2f} 秒")
# print(f"  测试集准确率: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")

# ──────────────────────────────────────────────
# 第七步：详细评估报告
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第7步】详细评估报告")
print("=" * 60)

# class_names = list(label_encoder.classes_)
class_names = [category_names.get(i, f"类别{i}") for i in range(num_classes)]
print("\n📌 逻辑回归 分类报告:")
print(classification_report(y_test, y_pred_lr, target_names=class_names))
print("-" * 50)
# print("\n📌 随机森林 分类报告:")
# print(classification_report(y_test, y_pred_rf, target_names=class_names))
print("\n📌 余弦相似度 分类报告:")
print("-" * 50)
print(classification_report(y_test, y_pred_cos, target_names=class_names))

# print("\n📌 XGBoost 分类报告:")
# print("-" * 50)
# print(classification_report(y_test, y_pred_xgb, target_names=class_names))

# ──────────────────────────────────────────────
# 第八步：结果对比总结
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【总结】")
print("=" * 60)
print(f"  {'模型':<20} {'准确率':>10} {'训练耗时':>10}")
print(f"  {'-'*42}")
print(f"  {'逻辑回归 (LR)':<18} {lr_acc:>10.4f} {lr_time:>9.2f}s")
# print(f"  {'随机森林 (RF)':<18} {rf_acc:>10.4f} {rf_time:>9.2f}s")
# print(f"  {'XGBoost':<18} {xgb_acc:>10.4f} {xgb_time:>9.2f}s")
print(f"  {'余弦相似度 (Cos)':<18} {cos_acc:>10.4f} {cos_time:>9.2f}s")
print(f"\n  向量维度: {X_train_emb.shape[1]}")
print(f"  训练样本: {len(X_train_emb)}")
print(f"  测试样本: {len(X_test_emb)}")
print(f"  类别数量: {num_classes}")

# ──────────────────────────────────────────────
# 第九步（可选）：保存模型，方便以后直接预测
# ──────────────────────────────────────────────

# print("\n" + "=" * 60)
print("【第9步】保存模型...")

import joblib

joblib.dump(lr_model, "lr_model.pkl")
# joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("  ✅ 模型已保存:")
print("     - lr_model.pkl (逻辑回归)")
# print("     - xgb_model.pkl (XGBoost)")
print("     - label_encoder.pkl (标签编码器)")
print("\n  使用方法:")
print("     model = joblib.load('lr_model.pkl')")
print("     encoder = joblib.load('label_encoder.pkl')")
print("     pred = model.predict(embedding)")
print("     label = encoder.inverse_transform(pred)")

print("\n" + "=" * 60)
print("🎉 全部完成！")
print("=" * 60)
