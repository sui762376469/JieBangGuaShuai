# 功能: 使用 Qwen3-Embedding-0.6B 生成文本向量，
#       再用 MLP 神经网络进行文本分类
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# 第一步：加载数据
# ──────────────────────────────────────────────
DATA_FILE = "../dataset/text_true.txt"
EMBEDDING_SAVE_PATH = "./embeddings_10w21.npy"

print("=" * 60)
print("【第1步】加载数据...")

labels = []
texts = []

with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", maxsplit=1)
        if len(parts) == 2:
            texts.append(parts[0])
            labels.append(parts[1])

print(f"  共读取 {len(texts)} 条数据")

from collections import Counter
category_counts = Counter(labels)
for cat, count in category_counts.most_common():
    print(f"    {cat}: {count} 条")

# ──────────────────────────────────────────────
# 第二步：标签编码
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第2步】标签编码...")

label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)
print(f"  共有 {num_classes} 个类别: {list(label_encoder.classes_)}")

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

# ──────────────────────────────────────────────
# 第三步：划分训练集 / 测试集
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第3步】划分训练集 / 测试集...")

X_train, X_test, y_train, y_test = train_test_split(
    texts, label_ids,
    test_size=0.2,
    random_state=42
)

print(f"  训练集: {len(X_train)} 条")
print(f"  测试集: {len(X_test)} 条")

# ──────────────────────────────────────────────
# 第四步：生成 Embedding 向量
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第4步】使用 Qwen3-Embedding-0.6B 生成向量...")

if os.path.exists(EMBEDDING_SAVE_PATH):
    print(f"  检测到已有缓存文件 {EMBEDDING_SAVE_PATH}")
    all_embeddings = np.load(EMBEDDING_SAVE_PATH)
    print(f"  直接加载，跳过生成步骤")
    model_path = r"../qwen_model/Qwen3-Embedding-0.6B"
    print(f"  正在加载模型用于预测: {model_path} ...")
    encoder_model = SentenceTransformer(model_path, device="cuda")
    print(f"  模型加载成功！")
else:
    model_path = r"../qwen_model/Qwen3-Embedding-0.6B"
    print(f"  正在加载模型: {model_path} ...")
    encoder_model = SentenceTransformer(model_path, device="cuda")
    print(f"  模型加载成功！向量维度: {encoder_model.get_sentence_embedding_dimension()}")

    all_texts = X_train + X_test
    print(f"  开始生成 {len(all_texts)} 条文本的向量...")

    start_time = time.time()

    instruction = "query:"
    full_texts = [f"{instruction}{t}" for t in all_texts]

    all_embeddings = encoder_model.encode(
        full_texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    elapsed = time.time() - start_time
    print(f"  ✅ 向量生成完成！耗时: {elapsed:.1f} 秒")

    np.save(EMBEDDING_SAVE_PATH, all_embeddings)

split_idx = len(X_train)
X_train_emb = all_embeddings[:split_idx]
X_test_emb = all_embeddings[split_idx:]

print(f"  训练集向量形状: {X_train_emb.shape}")
print(f"  测试集向量形状: {X_test_emb.shape}")

# ──────────────────────────────────────────────
# 第五步：转换为 PyTorch 张量
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第5步】转换为 PyTorch 张量...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  使用设备: {device}")

# 转换为张量
X_train_tensor = torch.FloatTensor(X_train_emb)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_emb)
y_test_tensor = torch.LongTensor(y_test)

# 创建 DataLoader
BATCH_SIZE = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"  训练集张量: {X_train_tensor.shape}")
print(f"  测试集张量: {X_test_tensor.shape}")
print(f"  Batch Size: {BATCH_SIZE}")

# ──────────────────────────────────────────────
# 第六步：定义 MLP 神经网络分类器
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第6步】定义 MLP 神经网络分类器...")

input_dim = X_train_emb.shape[1]  # embedding 维度

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.BatchNorm1d(hidden_dim // 2),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(hidden_dim // 2, hidden_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(hidden_dim // 4, num_classes)
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5), # 增加 Dropout 比例，强迫模型学习更鲁棒的特征
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3), # 增加 Dropout 比例
            nn.Linear(hidden_dim // 2, num_classes)            
        )
    
    def forward(self, x):
        return self.network(x)

model = MLPClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dim=512).to(device)

# 打印模型结构
print(f"  输入维度: {input_dim}")
print(f"  类别数: {num_classes}")
print(f"\n  模型结构:")
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  总参数量: {total_params:,}")
print(f"  可训练参数量: {trainable_params:,}")

# ──────────────────────────────────────────────
# 第七步：训练配置
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第7步】配置训练参数...")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

EPOCHS = 100
print(f"  损失函数: CrossEntropyLoss")
print(f"  优化器: AdamW (lr=1e-3, weight_decay=1e-4)")
print(f"  学习率调度: CosineAnnealingLR")
print(f"  训练轮数: {EPOCHS}")

# ──────────────────────────────────────────────
# 第八步：训练循环
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第8步】开始训练...")
print("=" * 60)

best_acc = 0.0
best_model_state = None
train_losses = []

# 加入早停机制
# best_acc = 0
# patience = 10
# no_improve = 0

for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    scheduler.step()
    
    train_loss = epoch_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        X_test_gpu = X_test_tensor.to(device)
        outputs = model(X_test_gpu)
        _, y_pred = torch.max(outputs.data, 1)
        y_pred_cpu = y_pred.cpu().numpy()
    
    val_acc = accuracy_score(y_test, y_pred_cpu)
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = model.state_dict().copy()
    
    # -----------早停机制-----------
    # if val_acc > best_acc:
    #     no_improve = 0

    # else:
    #     no_improve += 1
    #     if no_improve >= patience:
    #         print("早停！")
    #         break
    # -----------早停机制-----------

    # 每10轮打印一次
    if (epoch + 1) % 10 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

print("\n  ✅ 训练完成！")
print(f"  最佳验证准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")

# 加载最佳模型
model.load_state_dict(best_model_state)

# ──────────────────────────────────────────────
# 第九步：最终评估
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第9步】最终评估...")

model.eval()
with torch.no_grad():
    X_test_gpu = X_test_tensor.to(device)
    outputs = model(X_test_gpu)
    _, y_pred = torch.max(outputs.data, 1)
    y_pred = y_pred.cpu().numpy()

final_acc = accuracy_score(y_test, y_pred)

print(f"  测试集准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")

# 详细分类报告
class_names = [category_names.get(i, f"类别{i}") for i in range(num_classes)]
print("\n📌 MLP 神经网络 分类报告:")
all_labels = list(range(17)) 
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0,labels=all_labels))

# ──────────────────────────────────────────────
# 第十步：保存模型
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第10步】保存模型...")

torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,
    'num_classes': num_classes,
    'hidden_dim': 512,
}, "mlp_model.pth")
print("  ✅ 模型已保存: mlp_model.pth")

# ──────────────────────────────────────────────
# 第十一步：预测示例
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("【第11步】预测示例...")

def predict(text):
    model.eval()
    with torch.no_grad():
        emb = encoder_model.encode([f"query:{text}"], normalize_embeddings=True)
        emb_tensor = torch.FloatTensor(emb).to(device)
        output = model(emb_tensor)
        pred = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

# 测试预测
test_texts = [
    "工厂排放废水污染了河流",
    "商家销售过期食品",
    "未成年人遭受校园欺凌"
]

for text in test_texts:
    pred = predict(text)
    print(f"  文本: {text}")
    print(f"  预测: {pred}\n")

print("=" * 60)
print("🎉 全部完成！")
print("=" * 60)
