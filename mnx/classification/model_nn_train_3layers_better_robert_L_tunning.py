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


# ──────────────────────────────────────────────
# 全局模型缓存（推理用：冻结 RoBERTa + MLP）
# ──────────────────────────────────────────────
_MODEL_CACHE = {}
CURRENT_DIR = Path(__file__).resolve().parent


# ──────────────────────────────────────────────
# 冻结 RoBERTa Embedding 提取（推理 / 降级方案）
# ──────────────────────────────────────────────
def get_roberta_embeddings(texts, model_path, device, batch_size=16, show_progress=True):
    """
    使用冻结的 Roberta-wwm-ext-large 生成文本向量
    （仅在没有端到端微调模型时作为降级方案使用）
    """
    cleaned_texts = []
    for t in texts:
        if t is None or (isinstance(t, float) and np.isnan(t)) or str(t).strip() == '':
            cleaned_texts.append("")
        else:
            cleaned_texts.append(str(t).strip())

    texts_to_process = cleaned_texts

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

    for i in tqdm(range(0, len(texts_to_process), batch_size), desc="文本向量化", disable=not show_progress):
        batch_texts = texts_to_process[i:i + batch_size]
        encoded = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        all_embeddings.append(cls_embeddings)

    all_embeddings = np.vstack(all_embeddings)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-10, a_max=None)
    all_embeddings = all_embeddings / norms

    return all_embeddings


# ══════════════════════════════════════════════
# ★ 新增：MLP 分类头（独立模块，便于复用）
# ══════════════════════════════════════════════
class MLPHead(nn.Module):
    """3 层 MLP 分类头，输入 1024 维 [CLS] 向量"""
    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=20):
        super().__init__()
        self.classifier = nn.Sequential(
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
        return self.classifier(x)


# 兼容旧接口（API 区仍引用此名称）
MLPClassifier = MLPHead


# ══════════════════════════════════════════════
# ★ 新增：端到端微调模型 = RoBERTa + MLP
# ══════════════════════════════════════════════
class RoBERTaMLPClassifier(nn.Module):
    """
    端到端微调分类器

    结构:
        RoBERTa-wwm-ext-large (冻结大部分层，最后 N 层可训练)
            └─ [CLS] hidden state (1024-d)
                └─ MLPHead (3 层全连接)
    """
    def __init__(self, model_path, num_classes, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_path, hidden_dropout_prob=dropout)
        hidden_size = self.roberta.config.hidden_size  # 1024 for roberta-wwm-ext-large
        self.mlp = MLPHead(input_dim=hidden_size, hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.mlp(cls_output)

    def unfreeze_last_n_layers(self, n=2):
        """
        解冻策略：
          1) 先冻结全部参数
          2) 解冻 encoder 最后 n 层
          3) 解冻 pooler
          4) MLP 始终可训练
        """
        # 第一步：全部冻结
        for param in self.roberta.parameters():
            param.requires_grad = False

        # 第二步：解冻最后 n 层 encoder layer
        total_layers = len(self.roberta.encoder.layer)
        for i in range(total_layers - n, total_layers):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = True

        # 第三步：解冻 pooler
        if self.roberta.pooler is not None:
            for param in self.roberta.pooler.parameters():
                param.requires_grad = True

        # 第四步：MLP 始终可训练（默认 requires_grad=True）

        # 统计
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  解冻策略: RoBERTa 最后 {n} 层 + Pooler + MLP 全部可训练")
        print(f"  可训练参数: {trainable:,} | 冻结参数: {frozen:,}")
        return trainable, frozen


# ══════════════════════════════════════════════
# API 区：供 Streamlit (app.py) 调用的接口
# ══════════════════════════════════════════════

def api_train_model(df_train, df_mapping, text_col, label_col,
                    map_id_col, map_name_col, progress_cb=None):
    device = get_compute_device()
    if progress_cb:
        progress_cb(5, f"正在读取映射表与清洗数据... (设备: {device.upper()})")

    # ── 1. 映射表 ──
    mapping_dict = dict(zip(df_mapping[map_id_col], df_mapping[map_name_col]))
    map_save_path = CURRENT_DIR / "category_map.json"
    with open(map_save_path, "w", encoding="utf-8") as f:
        safe_map = {str(k): str(v) for k, v in mapping_dict.items()}
        json.dump(safe_map, f, ensure_ascii=False, indent=2)

    # ── 2. 数据预处理 ──
    texts = df_train[text_col].astype(str).tolist()
    raw_labels = df_train[label_col].tolist()
    target_names = [mapping_dict.get(lbl, str(lbl)) for lbl in raw_labels]

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(target_names)
    num_classes = len(encoder.classes_)

    if progress_cb:
        progress_cb(10, f"数据预处理完毕，共 {len(texts)} 条，{num_classes} 个类别")

    # ── 3. 模型路径 & 超参数 ──
    embed_path = CURRENT_DIR.parent / "core_model" / "Roberta-wwm-ext-large"
    model_path_str = str(embed_path)

    BATCH_SIZE = 16
    MAX_LENGTH = 256
    EPOCHS = 50
    WARMUP_RATIO = 0.1
    ROBERTA_LR = 1e-5
    MLP_LR = 1e-3
    UNFREEZE_LAYERS = 4
    HIDDEN_DIM = 512

    # ── 4. 创建端到端模型 ──
    if progress_cb:
        progress_cb(15, "正在加载 RoBERTa 预训练权重...")

    model = RoBERTaMLPClassifier(
        model_path_str, num_classes,
        hidden_dim=HIDDEN_DIM, dropout=0.1
    )
    model.unfreeze_last_n_layers(UNFREEZE_LAYERS)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

    # ══════════════════════════════════════════════
    # ★ Tokenize + 拼接（逐批，不再最后 torch.cat）
    # ══════════════════════════════════════════════
    if progress_cb:
        progress_cb(20, "正在 Tokenize 文本...")

    tokenizer = AutoTokenizer.from_pretrained(model_path_str)

    def batch_tokenize_and_concat(text_list, batch_size=512, desc="Tokenize"):
        """
        ★ 逐批 tokenize 并立即拼接，避免最后一次性 torch.cat
        """
        all_input_ids = None
        all_attention_masks = None
        total = len(text_list)

        for i in range(0, total, batch_size):
            batch = text_list[i:i + batch_size]
            encoded = tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            if all_input_ids is None:
                all_input_ids = encoded['input_ids']
                all_attention_masks = encoded['attention_mask']
            else:
                all_input_ids = torch.cat([all_input_ids, encoded['input_ids']], dim=0)
                all_attention_masks = torch.cat([all_attention_masks, encoded['attention_mask']], dim=0)

            # ★ 每批都更新进度
            if progress_cb:
                done = min(i + batch_size, total)
                pct = 20 + int(8 * (done / total))
                progress_cb(pct, f"Tokenize: {done}/{total}")

        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks,
        }

    all_encodings = batch_tokenize_and_concat(texts, batch_size=512)
    all_labels_tensor = torch.LongTensor(encoded_labels)

    if progress_cb:
        progress_cb(28, f"Tokenize 完毕，张量形状: {list(all_encodings['input_ids'].shape)}")

    # ══════════════════════════════════════════════
    # ★ 划分数据集（带进度）
    # ══════════════════════════════════════════════
    if progress_cb:
        progress_cb(29, "正在划分训练集 / 验证集...")

    indices = np.arange(len(texts))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    if progress_cb:
        progress_cb(30, "正在创建训练集张量...")

    train_dataset = TensorDataset(
        all_encodings['input_ids'][train_idx],
        all_encodings['attention_mask'][train_idx],
        all_labels_tensor[train_idx]
    )

    # ══════════════════════════════════════════════
    # ★ 验证集转移 GPU（带进度）
    # ══════════════════════════════════════════════
    if progress_cb:
        progress_cb(31, "正在将验证集转移到 GPU...")

    val_input_ids = all_encodings['input_ids'][val_idx].to(device)
    val_attention_mask = all_encodings['attention_mask'][val_idx].to(device)
    val_labels = all_labels_tensor[val_idx].numpy()

    if progress_cb:
        progress_cb(33, "验证集转移完毕，正在创建 DataLoader...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=False
    )

    # ══════════════════════════════════════════════
    # ★ 优化器 / 调度器（带进度）
    # ══════════════════════════════════════════════
    if progress_cb:
        progress_cb(35, "正在配置优化器和学习率调度器...")

    roberta_params = []
    mlp_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('roberta.'):
            roberta_params.append(param)
        else:
            mlp_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': roberta_params, 'lr': ROBERTA_LR, 'weight_decay': 0.05},
        {'params': mlp_params,     'lr': MLP_LR,     'weight_decay': 1e-3},
    ])

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    if progress_cb:
        progress_cb(38, f"配置完毕 | AMP: {'启用' if use_amp else '禁用'} | "
                       f"RoBERTa LR: {ROBERTA_LR} | MLP LR: {MLP_LR} | "
                       f"总步数: {total_steps} | Warmup: {warmup_steps}")

    # ════════════════════════════════════════
    # 10. 训练循环
    # ════════════════════════════════════════
    best_val_acc = 0.0
    best_model_state = None
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_input_ids, batch_mask, batch_labels in train_loader:
            batch_input_ids = batch_input_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_input_ids, batch_mask)
                    loss = criterion(outputs, batch_labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_input_ids, batch_mask)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0.0

        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    val_outputs = model(val_input_ids, val_attention_mask)
            else:
                val_outputs = model(val_input_ids, val_attention_mask)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_pred_np = val_predicted.cpu().numpy()

        val_acc = accuracy_score(val_labels, val_pred_np)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch % 2 == 0) or (epoch == EPOCHS - 1):
            log_msg = (
                f"  Epoch [{epoch + 1:3d}/{EPOCHS}] | "
                f"Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            print(log_msg)
            if progress_cb:
                prog = 38 + int(57 * ((epoch + 1) / EPOCHS))
                progress_cb(prog, log_msg)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n  ✅ 训练完成！最佳验证准确率: {best_val_acc:.4f}，已恢复最佳参数")

    # ── 11. 保存 ──
    if progress_cb:
        progress_cb(97, "正在保存端到端微调模型...")

    model_save_path = CURRENT_DIR / "mlp_model.pth"
    encoder_save_path = CURRENT_DIR / "mlp_encoder.pkl"

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'roberta_mlp_e2e',
        'model_path': model_path_str,
        'input_dim': 1024,
        'num_classes': num_classes,
        'hidden_dim': HIDDEN_DIM,
        'max_length': MAX_LENGTH,
    }, str(model_save_path))
    joblib.dump(encoder, str(encoder_save_path))

    _MODEL_CACHE.clear()

    if progress_cb:
        progress_cb(100, "🎉 端到端微调完成！模型已自动部署。")
    return True


def api_predict_categories(texts, batch_size=16):
    """
    预测 API（自动适配端到端微调模型 / 冻结+MLP 降级方案）
    """
    if not texts:
        return []

    device = get_compute_device()

    if "model" not in _MODEL_CACHE:
        model_path = CURRENT_DIR / "mlp_model.pth"
        encoder_path = CURRENT_DIR / "mlp_encoder.pkl"

        if not model_path.exists() or not encoder_path.exists():
            raise FileNotFoundError("未找到训练好的模型，请先执行模型训练！")

        checkpoint = torch.load(str(model_path), map_location=device)
        encoder = joblib.load(str(encoder_path))

        model_type = checkpoint.get('model_type', 'mlp_only')

        if model_type == 'roberta_mlp_e2e':
            # ── 端到端微调模型 ──
            model = RoBERTaMLPClassifier(
                checkpoint['model_path'],
                checkpoint['num_classes'],
                checkpoint['hidden_dim']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(checkpoint['model_path'])
            _MODEL_CACHE["tokenizer"] = tokenizer
            _MODEL_CACHE["max_length"] = checkpoint.get('max_length', 256)

        else:
            # ── 降级：冻结 RoBERTa + 独立 MLP ──
            model = MLPClassifier(
                checkpoint['input_dim'],
                checkpoint['num_classes'],
                checkpoint['hidden_dim']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            embed_path = CURRENT_DIR.parent / "core_model" / "Roberta-wwm-ext-large"
            _ = get_roberta_embeddings(["初始化"], str(embed_path), device,
                                       batch_size=1, show_progress=False)

        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["encoder"] = encoder
        _MODEL_CACHE["model_type"] = model_type

    model = _MODEL_CACHE["model"]
    encoder = _MODEL_CACHE["encoder"]
    model_type = _MODEL_CACHE["model_type"]

    if model_type == 'roberta_mlp_e2e':
        # ── 端到端推理 ──
        tokenizer = _MODEL_CACHE["tokenizer"]
        max_length = _MODEL_CACHE["max_length"]

        all_predictions = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_texts = [str(t) if t is not None else "" for t in batch_texts]

            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(encoded['input_ids'], encoded['attention_mask'])
                predicted_ids = torch.argmax(outputs, dim=1).cpu().numpy()

            all_predictions.extend(predicted_ids.tolist())

        predicted_labels = encoder.inverse_transform(all_predictions)
    else:
        # ── 降级推理：冻结向量 + MLP ──
        embed_path = CURRENT_DIR.parent / "core_model" / "Roberta-wwm-ext-large"
        embeddings = get_roberta_embeddings(
            texts, str(embed_path), device,
            batch_size=batch_size, show_progress=False
        )
        with torch.no_grad():
            tensor_emb = torch.FloatTensor(embeddings).to(device)
            outputs = model(tensor_emb)
            predicted_ids = torch.argmax(outputs, dim=1).cpu().numpy()
        predicted_labels = encoder.inverse_transform(predicted_ids)

    return predicted_labels.tolist()


# ──────────────────────────────────────────────
# 终端脚本区：独立测试（已改为端到端微调）
# ──────────────────────────────────────────────
def main():
    DATA_FILE = "../dataset/text_true.txt"
    device = get_compute_device()

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

    X_train, X_test, y_train, y_test = train_test_split(
        texts, label_ids, test_size=0.2, random_state=42
    )

    # ══════════════════════════════════════════════
    # ★ 第4步：端到端微调（替代原来的冻结向量提取）
    # ══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"【第4步】创建端到端微调模型 (设备: {device.upper()})...")

    model_path = r"../core_model/Roberta-wwm-ext-large"
    MAX_LENGTH = 256

    model = RoBERTaMLPClassifier(
        model_path, num_classes, hidden_dim=512, dropout=0.1
    )
    trainable, frozen = model.unfreeze_last_n_layers(n=2)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型总参数: {total_params:,}")

    # ── Tokenize ──
    print("\n" + "=" * 60)
    print("【第5步】Tokenize 全部文本...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize(text_list):
        return tokenizer(
            text_list, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )

    train_encodings = tokenize(X_train)
    test_encodings = tokenize(X_test)

    X_train_ids = train_encodings['input_ids']
    X_train_mask = train_encodings['attention_mask']
    y_train_tensor = torch.LongTensor(y_train)

    X_test_ids = test_encodings['input_ids'].to(device)
    X_test_mask = test_encodings['attention_mask'].to(device)

    train_dataset = TensorDataset(X_train_ids, X_train_mask, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, drop_last=False
    )

    # ── 优化器 & 调度器 ──
    print("\n" + "=" * 60)
    print("【第6步】配置差异化学习率...")

    roberta_params = []
    mlp_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('roberta.'):
            roberta_params.append(param)
        else:
            mlp_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': roberta_params, 'lr': 2e-5,  'weight_decay': 0.01},
        {'params': mlp_params,     'lr': 2e-3,  'weight_decay': 1e-4},
    ])

    EPOCHS = 50
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    criterion = nn.CrossEntropyLoss()

    # ── 训练循环 ──
    print("\n" + "=" * 60)
    print(f"【第7步】开始端到端微调 | AMP: {'启用' if use_amp else '禁用'}...")

    best_acc = 0.0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_ids, batch_mask, batch_labels in train_loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_ids, batch_mask)
                    loss = criterion(outputs, batch_labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_ids, batch_mask)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total

        # 验证
        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    val_outputs = model(X_test_ids, X_test_mask)
            else:
                val_outputs = model(X_test_ids, X_test_mask)
            _, y_pred = torch.max(val_outputs.data, 1)
            y_pred_np = y_pred.cpu().numpy()

        val_acc = accuracy_score(y_test, y_pred_np)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch + 1:3d}/{EPOCHS}] | "
                  f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

    print(f"\n  ✅ 训练完成！最佳验证准确率: {best_acc:.4f}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ── 最终评估 ──
    print("\n" + "=" * 60)
    print("【第8步】最终评估...")

    model.eval()
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(X_test_ids, X_test_mask)
        else:
            outputs = model(X_test_ids, X_test_mask)
        _, y_pred = torch.max(outputs.data, 1)
        y_pred = y_pred.cpu().numpy()

    class_names = [category_names.get(i, f"类别{i}") for i in range(num_classes)]
    all_labels = list(range(min(17, num_classes)))
    print("\n📌 端到端 RoBERTa+MLP 分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names,
                                zero_division=0, labels=all_labels))

    # ── 保存 ──
    print("\n" + "=" * 60)
    print("【第9步】保存模型...")

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'roberta_mlp_e2e',
        'model_path': model_path,
        'input_dim': 1024,
        'num_classes': num_classes,
        'hidden_dim': 512,
        'max_length': MAX_LENGTH,
    }, "mlp_model.pth")
    joblib.dump(label_encoder, "mlp_encoder.pkl")
    print("  ✅ 模型已保存: mlp_model.pth + mlp_encoder.pkl")

    # ── 预测示例 ──
    print("\n" + "=" * 60)
    print("【第10步】预测示例...")

    def predict(text):
        model.eval()
        enc = tokenizer([text], padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(enc['input_ids'], enc['attention_mask'])
            pred = torch.argmax(output, dim=1).item()
        return label_encoder.inverse_transform([pred])[0]

    test_texts = [
        "工厂排放废水污染了河流",
        "商家销售过期食品",
        "未成年人遭受校园欺凌",
        "等一排只剩一两家不修，村中间的又修一间，是搞特殊吗？",
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
