import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import joblib
from pathlib import Path
from collections import Counter
from tqdm import tqdm


# ──────────────────────────────────────────────
# 核心计算设备智能探针
# ──────────────────────────────────────────────
def get_compute_device():
    """全局统一的设备检测，支持双模式无缝切换"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_roberta_embeddings(texts, model_path, device, batch_size=16, show_progress=True):
    """
    使用 Roberta-wwm-ext-large 生成文本向量
    
    Args:
        texts: 文本列表
        model_path: 模型路径
        device: 计算设备
        batch_size: 批次大小
        show_progress: 是否显示进度条
    
    Returns:
        numpy array of embeddings (shape: [len(texts), 1024])
    """
    cleaned_texts = []
    for t in texts:
        if t is None or (isinstance(t, float) and np.isnan(t)) or str(t).strip() == '':
            cleaned_texts.append("")  # 空值统一为空字符串
        else:
            cleaned_texts.append(str(t).strip())

    texts_to_process = cleaned_texts
    # 加载 tokenizer 和 model（如果还没加载的话）
    if "roberta_tokenizer" not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
        model.eval()
        _MODEL_CACHE["roberta_tokenizer"] = tokenizer
        _MODEL_CACHE["roberta_model"] = model
    else:
        tokenizer = _MODEL_CACHE["roberta_tokenizer"]
        model = _MODEL_CACHE["roberta_model"]
    
    all_embeddings = []
    
    # 分批处理
    for i in tqdm(range(0, len(texts_to_process), batch_size), desc="文本向量化"):
        batch_texts = texts_to_process[i:i + batch_size]
        
        # tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # 获取模型输出
        with torch.no_grad():
            outputs = model(**encoded)
            # 使用 [CLS] token 的输出作为句子表示 (或者用 mean pooling)
            # 方案1: [CLS] token (推荐用于分类任务)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 方案2: Mean Pooling (备选)
            # attention_mask = encoded['attention_mask']
            # input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            # sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
            # sum_mask = input_mask_expanded.sum(1)
            # sum_mask = torch.clamp(sum_mask, min=1e-9)
            # cls_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        all_embeddings.append(cls_embeddings)
    
    # 合并所有批次
    all_embeddings = np.vstack(all_embeddings)
    
    # L2 归一化 (与原来的 normalize_embeddings=True 保持一致)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-10, a_max=None)  # 避免除以0
    all_embeddings = all_embeddings / norms
    
    return all_embeddings

# ──────────────────────────────────────────────
# 全局：定义 MLP 神经网络结构 (双模式共用)
# ──────────────────────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ──────────────────────────────────────────────
# API 区：供 Streamlit (app.py) 调用的动态训练与推理接口
# ──────────────────────────────────────────────
_MODEL_CACHE = {}
CURRENT_DIR = Path(__file__).resolve().parent


def api_train_model(df_train, df_mapping, text_col, label_col, map_id_col, map_name_col, progress_cb=None):
    device = get_compute_device()
    if progress_cb: progress_cb(5, f"正在读取映射表与清洗数据... (当前设备: {device.upper()})")

    mapping_dict = dict(zip(df_mapping[map_id_col], df_mapping[map_name_col]))

    map_save_path = CURRENT_DIR / "category_map.json"
    with open(map_save_path, "w", encoding="utf-8") as f:
        safe_map = {str(k): str(v) for k, v in mapping_dict.items()}
        json.dump(safe_map, f, ensure_ascii=False, indent=2)

    texts = df_train[text_col].astype(str).tolist()
    raw_labels = df_train[label_col].tolist()

    target_names = [mapping_dict.get(lbl, str(lbl)) for lbl in raw_labels]

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(target_names)
    num_classes = len(encoder.classes_)

    if progress_cb: progress_cb(15, "正在加载 Embedding 模型...")
    embed_path = CURRENT_DIR.parent / "core_model" / "Roberta-wwm-ext-large"
    input_dim = 1024

    if progress_cb: progress_cb(30, "正在生成高维特征向量 (耗时较长，请耐心等待)...")
    
    embeddings = get_roberta_embeddings(
        texts, 
        str(embed_path), 
        device, 
        batch_size=16, 
        show_progress=True
    )

    if progress_cb: progress_cb(60, "向量生成完毕，开始训练 3层 MLP 神经网络...")
    X_all = np.array(embeddings)
    y_all = np.array(encoded_labels)

    # 划分训练集 / 测试集 (与 main() 保持一致: 8:2)
    X_train_emb, X_val_emb, y_train_arr, y_val_arr = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    # 转换为 PyTorch 张量
    X_train_tensor = torch.FloatTensor(X_train_emb)
    y_train_tensor = torch.LongTensor(y_train_arr)
    X_val_tensor = torch.FloatTensor(X_val_emb)       # ← 新增测试集张量
    y_val_tensor = torch.LongTensor(y_val_arr)         # ← 新增测试集标签
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


    model = MLPClassifier(input_dim, num_classes, 512).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    epochs = 100

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 
    # model.train()
    # for epoch in range(epochs):
    #     total_loss = 0
    #     for batch_X, batch_y in dataloader:
    #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(batch_X)
    #         loss = criterion(outputs, batch_y)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()

    #     if progress_cb and (epoch % 5 == 0 or epoch == epochs - 1):
    #         prog = 60 + int(35 * ((epoch + 1) / epochs))
    #         progress_cb(prog, f"Epoch [{epoch + 1}/{epochs}] | Loss: {total_loss / len(dataloader):.4f}")
    model.train()
    best_val_acc = 0.0          # ← 新增：跟踪最佳验证准确率
    best_model_state = None     # ← 新增：保存最佳模型参数

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        # ── 计算训练集指标 ──
        train_loss = total_loss / len(dataloader)
        train_acc = correct / total if total > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']

        # ── 在测试集上评估（与 main() 第 337-344 行一致）──
        model.eval()
        with torch.no_grad():
            X_val_gpu = X_val_tensor.to(device)
            val_outputs = model(X_val_gpu)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_pred_cpu = val_predicted.cpu().numpy()
        val_acc = accuracy_score(y_val_arr, val_pred_cpu)   # ← 需要 from sklearn.metrics import accuracy_score（已有）

        # ── 保存最佳模型（与 main() 第 345-347 行一致）──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        model.train()  # ← 切回训练模式，继续下一个 epoch
        scheduler.step()

        # ── 输出日志 ──
        if (epoch % 5 == 0 or epoch == epochs - 1):
            prog = 60 + int(35 * ((epoch + 1) / epochs))
            log_msg = (
                f"  Epoch [{epoch + 1:3d}/{epochs}] | "
                f"Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "          # ← 新增
                f"LR: {current_lr:.6f}"
            )
            
            print(log_msg)
            
            if progress_cb:
                progress_cb(prog, log_msg)

    print(f"\n  ✅ 训练完成！最佳验证准确率: {best_val_acc:.4f}")   # ← 新增总结输出

    # ── 恢复最佳模型（与 main() 第 355 行一致）──
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("  ✓ 已恢复最佳模型参数")

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_val_arr, val_pred_cpu)
    print("\n各类别准确率：")
    # 注意：api_train_model 没有 class_names，用 encoder.classes_ 代替
    for i, name in enumerate(encoder.classes_):
        if i < num_classes:
            class_total = cm[i].sum()
            class_correct = cm[i][i]
            if class_total > 0:
                print(f"  {name}: {class_correct}/{class_total} = {class_correct/class_total:.2%}")


    if progress_cb: progress_cb(95, "训练完成，正在保存最新模型...")
    model_save_path = CURRENT_DIR / "mlp_model.pth"
    encoder_save_path = CURRENT_DIR / "mlp_encoder.pkl"

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'hidden_dim': 512,
    }, str(model_save_path))
    joblib.dump(encoder, str(encoder_save_path))

    _MODEL_CACHE.clear()

    if progress_cb: progress_cb(100, "🎉 模型重训完成！已自动部署。")
    return True


def api_predict_categories(texts, batch_size=16):
    """前端一键真实的AI大模型预测 API"""
    if not texts: return []
    device = get_compute_device()

    if "model" not in _MODEL_CACHE:
        model_path = CURRENT_DIR / "mlp_model.pth"
        encoder_path = CURRENT_DIR / "mlp_encoder.pkl"
        if not model_path.exists() or not encoder_path.exists():
            raise FileNotFoundError("未找到训练好的神经网络，请先在左侧执行模型训练！")

        checkpoint = torch.load(str(model_path), map_location=device)
        encoder = joblib.load(str(encoder_path))

        model = MLPClassifier(checkpoint['input_dim'], checkpoint['num_classes'], checkpoint['hidden_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        embed_path = CURRENT_DIR.parent / "core_model" / "Roberta-wwm-ext-large"
        _ = get_roberta_embeddings(["初始化"], str(embed_path), device, batch_size=1, show_progress=False)
        
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["encoder"] = encoder


    model = _MODEL_CACHE["model"]
    encoder = _MODEL_CACHE["encoder"]

    embed_path = CURRENT_DIR.parent / "core_model" / "Roberta-wwm-ext-large"
    embeddings = get_roberta_embeddings(
        texts, 
        str(embed_path), 
        device, 
        batch_size=batch_size, 
        show_progress=False
    )                                        

    with torch.no_grad():
        tensor_emb = torch.FloatTensor(embeddings).to(device)
        # 用我们训练好的 3层 MLP 神经网络进行最终分类
        outputs = model(tensor_emb)
        predicted_ids = torch.argmax(outputs, dim=1).cpu().numpy()

    predicted_labels = encoder.inverse_transform(predicted_ids)
    return predicted_labels.tolist()


# ──────────────────────────────────────────────
# 终端脚本区：保留原程序的完整独立测试功能
# ──────────────────────────────────────────────
def main():
    DATA_FILE = "../dataset/text_true.txt"
    EMBEDDING_SAVE_PATH = "./embeddings_10w21.npy"
    device = get_compute_device()  # ✅ 单测模式下同样智能识别

    print("=" * 60)
    print("【第1步】加载数据...")

    labels = []
    texts = []

    if not os.path.exists(DATA_FILE):
        print(f"找不到测试文件 {DATA_FILE}，请先准备数据。")
        return

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

    category_counts = Counter(labels)
    for cat, count in category_counts.most_common():
        print(f"    {cat}: {count} 条")

    print("\n" + "=" * 60)
    print("【第2步】标签编码...")

    label_encoder = LabelEncoder()
    label_ids = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    category_names = {
        0: "非公益诉讼", 1: "生态环境和资源保护", 2: "食品药品安全", 3: "国有财产保护",
        4: "国有土地使用权出让", 5: "英雄烈士保护", 6: "未成年人保护", 7: "安全生产",
        8: "个人信息保护", 9: "军人地位和权益保障", 10: "反垄断和反不正当竞争",
        11: "反电信网络诈骗", 12: "农产品质量安全", 13: "妇女权益保障",
        14: "无障碍环境建设", 15: "文物和文化遗产保护", 16: "国防和军事利益保护",
        17: "控烟", 18: "预付式消费", 19: "拖欠工资"
    }

    print("\n" + "=" * 60)
    print("【第3步】划分训练集 / 测试集...")

    X_train, X_test, y_train, y_test = train_test_split(texts, label_ids, test_size=0.2, random_state=42)

    print("\n" + "=" * 60)
    print(f"【第4步】使用 Qwen3-Embedding-0.6B 生成向量 (设备: {device.upper()})...")

    if os.path.exists(EMBEDDING_SAVE_PATH):
        print(f"  检测到已有缓存文件 {EMBEDDING_SAVE_PATH}")
        all_embeddings = np.load(EMBEDDING_SAVE_PATH)
        model_path = r"../qwen_model/Qwen3-Embedding-0.6B"
        encoder_model = SentenceTransformer(model_path, device=device)  # ✅ 修复此处的硬编码
    else:
        model_path = r"../qwen_model/Qwen3-Embedding-0.6B"
        encoder_model = SentenceTransformer(model_path, device=device)  # ✅ 修复此处的硬编码

        all_texts = X_train + X_test
        start_time = time.time()
        instruction = "query:"
        full_texts = [f"{instruction}{t}" for t in all_texts]

        all_embeddings = encoder_model.encode(full_texts, batch_size=16, show_progress_bar=True,
                                              normalize_embeddings=True)

        elapsed = time.time() - start_time
        print(f"  ✅ 向量生成完成！耗时: {elapsed:.1f} 秒")
        np.save(EMBEDDING_SAVE_PATH, all_embeddings)

    split_idx = len(X_train)
    X_train_emb = all_embeddings[:split_idx]
    X_test_emb = all_embeddings[split_idx:]

    print("\n" + "=" * 60)
    print("【第5步】转换为 PyTorch 张量...")

    X_train_tensor = torch.FloatTensor(X_train_emb)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_emb)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    print("\n" + "=" * 60)
    print("【第6步】定义 MLP 神经网络分类器...")

    input_dim = X_train_emb.shape[1]
    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dim=512).to(device)

    print("\n" + "=" * 60)
    print("【第7步/第8步】配置参数并开始训练...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    EPOCHS = 100

    best_acc = 0.0
    best_model_state = None

    for epoch in range(EPOCHS):
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

        model.eval()
        with torch.no_grad():
            X_test_gpu = X_test_tensor.to(device)
            outputs = model(X_test_gpu)
            _, y_pred = torch.max(outputs.data, 1)
            y_pred_cpu = y_pred.cpu().numpy()

        val_acc = accuracy_score(y_test, y_pred_cpu)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"  Epoch [{epoch + 1:3d}/{EPOCHS}] | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

    print("\n  ✅ 训练完成！")
    model.load_state_dict(best_model_state)

    print("\n" + "=" * 60)
    print("【第9步】最终评估...")
    model.eval()
    with torch.no_grad():
        X_test_gpu = X_test_tensor.to(device)
        outputs = model(X_test_gpu)
        _, y_pred = torch.max(outputs.data, 1)
        y_pred = y_pred.cpu().numpy()

    class_names = [category_names.get(i, f"类别{i}") for i in range(num_classes)]
    print("\n📌 MLP 神经网络 分类报告:")
    all_labels = list(range(17))
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0, labels=all_labels))

    print("\n" + "=" * 60)
    print("【第10步】保存模型...")
    torch.save(
        {'model_state_dict': model.state_dict(), 'input_dim': input_dim, 'num_classes': num_classes, 'hidden_dim': 512},
        "mlp_model.pth")
    print("  ✅ 模型已保存: mlp_model.pth")

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
        "未成年人遭受校园欺凌",
        "等一排只剩一两家不修，村中间（第一排后面）的又修一间，是搞特殊吗？",
    ]

    for text in test_texts:
        pred = predict(text)
        print(f"  文本: {text}")
        print(f"  预测: {pred}\n")

        print("=" * 60)
        print("🎉 全部完成！")
        print("=" * 60)


if __name__ == "__main__":
    main()