import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

MODEL_NAME = 'qwen_model\Qwen3-Embedding-0.6B'
print(f"正在加载模型：{MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME, device='cuda')

# 2. 读取数据
df = pd.read_csv("generated_5category_1000data.csv")
X_raw = df['text'].tolist()
y = df['label'].values

# 3. 提取向量特征 (Embedding)
def get_qwen_embeddings(texts):
    # 针对多分类任务的通用指令
    instruction = "请对以下文本进行理解和特征提取，以便进行准确的文本分类："
    full_texts = [f"{instruction}{t}" for t in texts]
    return model.encode(full_texts, batch_size=50, show_progress_bar=True)

print("正在提取 100 条数据的向量特征...")
X_embeddings = get_qwen_embeddings(X_raw)

# 4. 定义多种分类算法（重点测试 XGBoost）
models = {
    "逻辑回归 (LR)": LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000),
    "支持向量机 (SVM)": SVC(kernel='linear', probability=True, random_state=42),
    "随机森林 (RF)": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN-1": KNeighborsClassifier(n_neighbors=1),
    "XGBoost": XGBClassifier(
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=6,
        objective='multi:softprob',  # 多分类
        num_class=len(np.unique(y)),  # 自动识别类别数
        random_state=42,
        use_label_encoder=False
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, 
        learning_rate=0.1,
        objective='multiclass',  # 多分类
        num_class=len(np.unique(y)),
        random_state=42
    )
}

# 5. 交叉验证评估（更可靠的评估方式）
print("\n=== 开始 5 折交叉验证 ===")
cv_results = []
for name, clf in models.items():
    print(f"正在评估 {name}...", end=" ")
    scores = cross_val_score(clf, X_embeddings, y, cv=5)
    cv_results.append({
        "算法": name, 
        "平均准确度": scores.mean(),
        "标准差": scores.std()
    })
    print(f"平均准确度：{scores.mean():.4f} (+/- {scores.std():.4f})")

# 6. 展示对比结果
res_df = pd.DataFrame(cv_results).sort_values("平均准确度", ascending=False)
print("\n=== 算法性能排名 ===")
print(res_df.to_string(index=False))

# 7. 绘制柱状图
plt.figure(figsize=(12, 7))
sns.set_context("talk")
ax = sns.barplot(x="平均准确度", y="算法", data=res_df, palette="viridis")

# 标注数值和误差棒
for i, (_, row) in enumerate(res_df.iterrows()):
    ax.annotate(f'{row["平均准确度"]:.2%}', 
                xy=(row["平均准确度"], i),
                ha='left', va='center',
                xytext=(5, 0), textcoords='offset points',
                fontsize=10, weight='bold')

plt.title("Qwen-Embedding + 不同分类器的 5 分类性能对比", fontsize=14, pad=20)
plt.xlabel("平均准确度 (5 折交叉验证)", fontsize=12)
plt.ylabel("分类算法", fontsize=12)
plt.xlim(0.7, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 8. 使用最佳模型进行训练和预测
best_model_name = res_df.iloc[0]["算法"]
print(f"\n=== 最佳模型：{best_model_name} ===")

# 重新训练最佳模型（用全部数据或划分训练集）
if best_model_name == "XGBoost":
    best_clf = models[best_model_name]
elif best_model_name == "逻辑回归 (LR)":
    best_clf = models[best_model_name]
elif best_model_name == "支持向量机 (SVM)":
    best_clf = models[best_model_name]
elif best_model_name == "随机森林 (RF)":
    best_clf = models[best_model_name]
elif best_model_name == "KNN-1":
    best_clf = models[best_model_name]
elif best_model_name == "LightGBM":
    best_clf = models[best_model_name]

# 划分训练集和测试集（用于最终评估）
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.3, random_state=42, stratify=y
)

# 训练最佳模型
best_clf.fit(X_train, y_train)

# 9. 在测试集上评估
y_pred = best_clf.predict(X_test)
print("\n--- 测试集评估报告 (30% 数据) ---")
target_names = ['类别 0', '类别 1', '类别 2', '类别 3', '类别 4']
print(classification_report(y_test, y_pred, target_names=target_names))
print(f"总体准确率：{accuracy_score(y_test, y_pred):.2%}")

# 10. 应用于新数据
def predict_new_data(new_texts):
    print(f"正在处理 {len(new_texts)} 条新数据...")
    new_embeddings = get_qwen_embeddings(new_texts)
    preds = best_clf.predict(new_embeddings)
    probs = best_clf.predict_proba(new_embeddings).max(axis=1)
    return preds, probs

# 示例预测
sample_unlabeled = [
    "某某大学办公家具采购合同。办公桌椅。木质家具", 
    "某某局网络安全等级保护测评服务定点采购合同"
    "华南师范大学办公区域日常保洁服务采购合同。服务范围：办公室、会议室、走廊、卫生间，每日保洁，每周深度清洁。服务期2年，费用38000元/年。采购要求：符合国家相关标准，按期交付。"
]
labels, confidences = predict_new_data(sample_unlabeled)

for txt, lab, conf in zip(sample_unlabeled, labels, confidences):
    print(f"预测结果：类别{lab} | 置信度：{conf:.4f} | 文本：{txt[:30]}...")