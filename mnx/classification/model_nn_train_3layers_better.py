import os
import time
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sentence_transformers import SentenceTransformer
import joblib
from pathlib import Path
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from openai import OpenAI

# ──────────────────────────────────────────────
# 核心计算设备智能探针
# ──────────────────────────────────────────────
def get_compute_device():
    """全局统一的设备检测，支持双模式无缝切换"""
    return "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# 定义 Focal Loss 以应对数据极度不平衡
# ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.02):  # ✅ 降低 label_smoothing 从 0.05→0.02
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        log_pt = self.ce(inputs, targets)
        pt = torch.exp(-log_pt)
        focal_loss = ((1 - pt) ** self.gamma) * log_pt
        return focal_loss.mean()

# ──────────────────────────────────────────────
# 全局：定义 MLP 神经网络结构 (优化版)
# ──────────────────────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),        # 使用 GELU 提升 NLP 表现
            nn.Dropout(0.3),  # 降低 Dropout 比例防欠拟合
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ──────────────────────────────────────────────
# API 客户端配置（用于不确定样本的二次分类）
# ──────────────────────────────────────────────
_API_BASE_URL = os.getenv(
    "CLASSIFICATION_API_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
_API_MODEL = os.getenv("CLASSIFICATION_API_MODEL", "qwen-plus-2025-07-28")


def _build_api_client():
    """按需创建二次分类客户端；未配置密钥时允许本地模型继续运行。"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(base_url=_API_BASE_URL, api_key=api_key, timeout=60.0)

_api_extra_body = {
    "enable_thinking": False,
}

# API 分类的系统提示词
API_SYSTEM_PROMPT = """你是一个针对12345投诉文本进行智能分类的专家，需要你筛选是否属于我们检察院负责的范围（检察院是法律监督机关，通过行使检察权，维护国家安全和社会秩序，维护个人和组织的合法权益等）， 当前有以下10种类别：
1、生态环境和资源保护
2、食品药品安全
3、国有财产保护
4、国有土地使用权出让
6、未成年人保护
7、安全生产
8、个人信息保护
13、妇女权益保障
14、无障碍环境建设
15、文物和文化遗产保护

每一行投诉文本是一条数据，格式为："ID.文本内容"
请按照以下格式输出分类结果：
ID:类别编号:c

其中：
- ID 是输入中的编号
- 类别编号必须是上述10个类别对应的数字（1,2,3,4,6,7,8,13,14,15）
- 如果文本不属于以上任何类别，类别编号填 0
- c 表示分类结果（固定值）

示例：
输入：
1.工厂排放废水污染了河流
2.商家销售过期食品
3.今天天气不错

输出：
1:1:c
2:2:c
3:0:c

重要要求：
1. 每行一个结果，不要有其他解释文字
2. 类别编号必须是数字，不能是字母或其他字符
3. 即使不确定，也要选择一个最接近的类别编号
"""

# ──────────────────────────────────────────────
# API 区：供 Streamlit (app.py) 调用的动态训练与推理接口
# ──────────────────────────────────────────────
_MODEL_CACHE = {}
CURRENT_DIR = Path(__file__).resolve().parent

def api_train_model(df_train, df_mapping, text_col, label_col, map_id_col, map_name_col, progress_cb=None):
    """
    训练新的分类模型
    
    ⚠️ 重要：训练前会自动清理旧的模型文件和映射文件，确保使用全新的配置
    """
    device = get_compute_device()
    
    # ✅ 步骤0：清理旧的模型文件和映射文件
    if progress_cb:
        progress_cb(2, "🧹 正在清理旧的模型文件...")
    
    files_to_clean = [
        CURRENT_DIR / "category_map.json",      # 类别映射文件
        CURRENT_DIR / "mlp_model.pth",           # 模型权重文件
        CURRENT_DIR / "mlp_encoder.pkl",         # 标签编码器
    ]
    
    cleaned_files = []
    for file_path in files_to_clean:
        if file_path.exists():
            try:
                file_path.unlink()
                cleaned_files.append(file_path.name)
                print(f"  🗑️  已删除: {file_path.name}")
            except Exception as e:
                print(f"  ⚠️  删除失败 {file_path.name}: {e}")
    
    if cleaned_files:
        print(f"✅ 已清理 {len(cleaned_files)} 个旧文件: {', '.join(cleaned_files)}")
    else:
        print("ℹ️  没有需要清理的旧文件")
    
    # 清空模型缓存
    _MODEL_CACHE.clear()
    
    if progress_cb:
        progress_cb(5, f"正在读取映射表与清洗数据... (当前设备: {device.upper()})")

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

    if progress_cb: progress_cb(15, "训练阶段：正在加载 Embedding 模型...")
    embed_path = CURRENT_DIR.parent / "core_model" / "bge-m3"
    embedding_model = SentenceTransformer(str(embed_path), device=device)
    input_dim = embedding_model.get_sentence_embedding_dimension()


    # if progress_cb: progress_cb(30, "正在生成高维特征向量 (耗时较长，请耐心等待)...")
    # full_texts = [f"query:{t}" for t in texts]
    # embeddings = embedding_model.encode(full_texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

    # if progress_cb: progress_cb(60, "向量生成完毕，开始训练 3层 MLP 神经网络...")

    if progress_cb: progress_cb(30, "正在生成高维特征向量 (耗时较长，请耐心等待)...")
    full_texts = [f"query:{t}" for t in texts]
    
    total = len(full_texts)
    batch_size = 16
    all_embeddings = []
    for i in range(0, total, batch_size):
        batch = full_texts[i:i + batch_size]
        batch_emb = embedding_model.encode(batch, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        all_embeddings.append(batch_emb)
        if progress_cb and total > 0:
            prog = 30 + int(30 * ((i + len(batch)) / total))
            progress_cb(prog, f"正在生成高维特征向量... ({i + len(batch)}/{total})")
    embeddings = np.vstack(all_embeddings)

    if progress_cb: progress_cb(60, "向量生成完毕，开始训练 3层 MLP 神经网络...")





    X_all = np.array(embeddings)
    y_all = np.array(encoded_labels)

    X_train_emb, X_val_emb, y_train_arr, y_val_arr = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.FloatTensor(X_train_emb)
    y_train_tensor = torch.LongTensor(y_train_arr)
    X_val_tensor = torch.FloatTensor(X_val_emb)
    y_val_tensor = torch.LongTensor(y_val_arr)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = MLPClassifier(input_dim, num_classes, 512).to(device)
    
    # 计算类别权重 - 确保权重数量与 num_classes 一致
    unique_classes = np.unique(y_train_arr)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_arr)
    
    # 如果类别数不匹配，手动补齐权重（处理训练集中某些类别缺失的情况）
    if len(class_weights) < num_classes:
        print(f"  ⚠️  训练集中只有 {len(class_weights)} 个类别，补齐到 {num_classes} 个类别")
        full_weights = np.ones(num_classes)  # 缺失的类别使用权重 1.0
        for i, cls in enumerate(unique_classes):
            full_weights[int(cls)] = class_weights[i]
        class_weights = full_weights
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 100

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
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

        train_loss = total_loss / len(dataloader)
        train_acc = correct / total if total > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']

        model.eval()
        with torch.no_grad():
            X_val_gpu = X_val_tensor.to(device)
            val_outputs = model(X_val_gpu)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_pred_cpu = val_predicted.cpu().numpy()
        val_acc = accuracy_score(y_val_arr, val_pred_cpu)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        scheduler.step()

        if (epoch % 5 == 0 or epoch == epochs - 1):
            prog = 60 + int(35 * ((epoch + 1) / epochs))
            log_msg = (
                f"  Epoch [{epoch + 1:3d}/{epochs}] | "
                f"Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            print(log_msg)
            if progress_cb:
                progress_cb(prog, log_msg)

    print(f"\n  ✅ 训练完成！最佳验证准确率: {best_val_acc:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("  ✓ 已恢复最佳模型参数")

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


# def api_predict_categories(texts, batch_size=16):
#     # """前端一键真实的AI大模型预测 API"""
#     # valid_texts = [str(t) for t in texts if t and str(t).strip()]
#     # if not valid_texts: return []
    
#     # device = get_compute_device()

#     # if "model" not in _MODEL_CACHE:
#     #     model_path = CURRENT_DIR / "mlp_model.pth"
#     #     encoder_path = CURRENT_DIR / "mlp_encoder.pkl"
#     #     if not model_path.exists() or not encoder_path.exists():
#     #         raise FileNotFoundError("未找到训练好的神经网络，请先在左侧执行模型训练！")

#     #     checkpoint = torch.load(str(model_path), map_location=device)
#     #     encoder = joblib.load(str(encoder_path))

#     #     model = MLPClassifier(checkpoint['input_dim'], checkpoint['num_classes'], checkpoint['hidden_dim'])
#     #     model.load_state_dict(checkpoint['model_state_dict'])
#     #     model.to(device)
#     #     model.eval()

#     #     embed_path = CURRENT_DIR.parent / "core_model" / "bge-m3"
#     #     embedding_model = SentenceTransformer(str(embed_path), device=device)

#     #     _MODEL_CACHE["model"] = model
#     #     _MODEL_CACHE["encoder"] = encoder
#     #     _MODEL_CACHE["embedding_model"] = embedding_model

#     # model = _MODEL_CACHE["model"]
#     # encoder = _MODEL_CACHE["encoder"]
#     # embedding_model = _MODEL_CACHE["embedding_model"]

#     # full_texts = [f"query:{t}" for t in valid_texts]
    
#     # embeddings = embedding_model.encode(full_texts, batch_size=batch_size, show_progress_bar=True,
#     #                                     normalize_embeddings=True)
#     # print("stop here!2")

#     """前端一键真实的AI大模型预测 API"""
#     valid_texts = [str(t) for t in texts if t and str(t).strip()]
#     if not valid_texts: return []
    
#     device = get_compute_device()

#     if "model" not in _MODEL_CACHE:
#         model_path = CURRENT_DIR / "mlp_model.pth"
#         encoder_path = CURRENT_DIR / "mlp_encoder.pkl"
#         if not model_path.exists() or not encoder_path.exists():
#             raise FileNotFoundError("未找到训练好的神经网络，请先在左侧执行模型训练！")

#         checkpoint = torch.load(str(model_path), map_location=device)
#         encoder = joblib.load(str(encoder_path))

#         model = MLPClassifier(checkpoint['input_dim'], checkpoint['num_classes'], checkpoint['hidden_dim'])
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.to(device)
#         model.eval()

#         embed_path = CURRENT_DIR.parent / "core_model" / "bge-m3"
#         embedding_model = SentenceTransformer(str(embed_path), device=device)

#         _MODEL_CACHE["model"] = model
#         _MODEL_CACHE["encoder"] = encoder
#         _MODEL_CACHE["embedding_model"] = embedding_model

#     model = _MODEL_CACHE["model"]
#     encoder = _MODEL_CACHE["encoder"]
#     embedding_model = _MODEL_CACHE["embedding_model"]

#     full_texts = [f"query:{t}" for t in valid_texts]
    
#     total = len(full_texts)
#     all_embeddings = []
#     for i in range(0, total, batch_size):
#         batch = full_texts[i:i + batch_size]
#         batch_emb = embedding_model.encode(batch, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
#         all_embeddings.append(batch_emb)
#         if progress_cb and total > 0:
#             progress_cb(int(100 * ((i + len(batch)) / total)), f"正在生成特征向量... ({i + len(batch)}/{total})")
#     embeddings = np.vstack(all_embeddings)


















    
#     with torch.no_grad():
#         tensor_emb = torch.FloatTensor(embeddings).to(device)
#         outputs = model(tensor_emb)

#         # ====== 计算 Softmax 概率分布 ======
#         probs = torch.softmax(outputs, dim=1)  # 形状: [batch_size, num_classes]

#         # 取 Top-2 置信度及其对应类别索引
#         top2_probs, top2_indices = torch.topk(probs, k=2, dim=1)  # 各 [batch_size, 2]

#         top1_probs = top2_probs[:, 0].cpu().numpy()   # Top-1 置信度
#         top2_probs_val = top2_probs[:, 1].cpu().numpy()  # Top-2 置信度
#         confidence_gap = top1_probs - top2_probs_val    # Top1 - Top2 差值

#         # 判断，只有第一类准确率达到90%以上，我们才会认为本地模型可信，否则交给API处理
#         is_confident = top1_probs > 0.95

#         need_api_count = int((~is_confident).sum())     # 需要API处理的行数
#         print(f"[推理统计] 总行数={len(texts)}, 本地模型可信={is_confident.sum()}行, "
#               f"需转API处理={need_api_count}行")

#         # 最终预测标签（仍使用 Top-1 类别）
#         predicted_ids = top2_indices[:, 0].cpu().numpy()

#         # ====== 写入置信度到 txt 文件 ======
#         result_save_path = CURRENT_DIR / "prediction_confidence.txt"
#         with open(result_save_path, "w", encoding="utf-8") as fout:
#             for i in range(len(texts)):
#                 fout.write(f"{top1_probs[i]:.6f}, {top2_probs_val[i]:.6f}\n")
#         print(f"[置信度日志] 已将 {len(texts)} 条预测的 Top-1/Top-2 置信度写入: {result_save_path}")

#     predicted_labels = encoder.inverse_transform(predicted_ids)
#     return predicted_labels.tolist()

def _call_api_for_classification(texts_with_ids, progress_cb=None):
    """
    调用 API 对不确定的文本进行分类
    
    Args:
        texts_with_ids: list of (display_id, text) 元组
        progress_cb: 进度回调函数
    
    Returns:
        dict: {display_id: (category, group_type)}
    """
    if not texts_with_ids:
        return {}

    api_client = _build_api_client()
    if api_client is None:
        print("⚠️ 未配置 DASHSCOPE_API_KEY，低置信度样本将使用本地模型结果。")
        return {}
    
    BATCH_SIZE = 50
    all_results = {}
    total_batches = (len(texts_with_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(texts_with_ids))
        batch = texts_with_ids[start:end]
        
        # 构建发送给 API 的文本块
        batch_text_lines = []
        for display_id, text in batch:
            # 截断过长的文本（避免超 token 限制）
            truncated = text[:800] if len(text) > 800 else text
            batch_text_lines.append(f"{display_id}.{truncated}")
        
        batch_text = "\n".join(batch_text_lines)
        
        # 调用 API
        try:
            response = api_client.chat.completions.create(
                model=_API_MODEL,
                messages=[
                    {'role': 'system', 'content': API_SYSTEM_PROMPT},
                    {'role': 'user', 'content': batch_text}
                ],
                stream=False,
                extra_body=_api_extra_body
            )
            
            # 提取返回内容
            reply_content = response.choices[0].message.content.strip()
            print(f"\n--- 批次 {batch_idx + 1}/{total_batches} API 返回结果 ---")
            print(reply_content[:500])  # 只打印前500字符，避免过长
            
            # 解析返回结果
            parsed_count = 0
            for line in reply_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # 尝试匹配标准格式：ID:类别编号:c
                match = re.match(r'(\d+)[:：](\d+)[:：](.+)', line)
                if match:
                    result_id = int(match.group(1))
                    category = int(match.group(2))
                    group_type = match.group(3).strip()
                    all_results[result_id] = (category, group_type)
                    parsed_count += 1
                else:
                    # 尝试匹配 API 返回 "c" 作为类别的情况（表示无法分类）
                    match_fallback = re.match(r'(\d+)[:：]c[:：]c', line, re.IGNORECASE)
                    if match_fallback:
                        result_id = int(match_fallback.group(1))
                        # 对于无法分类的样本，标记为类别 0（非公益诉讼）
                        all_results[result_id] = (0, "c")
                        parsed_count += 1
                        print(f"  ⚠️  ID {result_id} API 无法分类，标记为类别 0")
                    else:
                        print(f"  ⚠️  无法解析的行: {line[:100]}")
            
            print(f"  ✅ 本批次成功解析 {parsed_count}/{len(batch)} 条结果")
            
            if progress_cb:
                progress = int(100 * (batch_idx + 1) / total_batches)
                progress_cb(progress, f"API 分类进度: {batch_idx + 1}/{total_batches} 批次")
                
        except Exception as e:
            print(f"\n批次 {batch_idx + 1} 调用出错: {e}")
            # 出错时该批次标记为默认值 0
            for display_id, _ in batch:
                all_results[display_id] = (0, "c")
    
    return all_results


def api_predict_categories(texts, batch_size=16, progress_cb=None):
    """
    前端一键真实的AI大模型预测 API
    集成本地模型和 API 混合推理：
    1. 使用本地 MLP 模型进行初步分类
    2. 读取置信度文件，对低置信度样本调用 API 进行二次分类
    3. 合并所有结果返回
    """
    valid_texts = [str(t) for t in texts if t and str(t).strip()]
    if not valid_texts: 
        return []
    
    device = get_compute_device()

    # 加载模型（如果未缓存）
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

        embed_path = CURRENT_DIR.parent / "core_model" / "bge-m3"
        embedding_model = SentenceTransformer(str(embed_path), device=device)

        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["encoder"] = encoder
        _MODEL_CACHE["embedding_model"] = embedding_model

    model = _MODEL_CACHE["model"]
    encoder = _MODEL_CACHE["encoder"]
    embedding_model = _MODEL_CACHE["embedding_model"]

    # 生成文本向量
    full_texts = [f"query:{t}" for t in valid_texts]
    
    total = len(full_texts)
    all_embeddings = []
    for i in range(0, total, batch_size):
        batch = full_texts[i:i + batch_size]
        batch_emb = embedding_model.encode(batch, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        all_embeddings.append(batch_emb)
        if progress_cb and total > 0:
            progress_cb(int(50 * ((i + len(batch)) / total)), f"正在生成特征向量... ({i + len(batch)}/{total})")
    embeddings = np.vstack(all_embeddings)
    
    # 使用本地模型进行预测
    with torch.no_grad():
        tensor_emb = torch.FloatTensor(embeddings).to(device)
        outputs = model(tensor_emb)

        # 计算 Softmax 概率分布
        probs = torch.softmax(outputs, dim=1)

        # 取 Top-2 置信度及其对应类别索引
        top2_probs, top2_indices = torch.topk(probs, k=2, dim=1)

        top1_probs = top2_probs[:, 0].cpu().numpy()   # Top-1 置信度
        top2_probs_val = top2_probs[:, 1].cpu().numpy()  # Top-2 置信度
        confidence_gap = top1_probs - top2_probs_val    # Top1 - Top2 差值

        # 判断：只有第一类准确率达到90%以上，我们才会认为本地模型可信，否则交给API处理
        is_confident = top1_probs > 0.90

        need_api_count = int((~is_confident).sum())     # 需要API处理的行数
        confident_count = int(is_confident.sum())       # 本地模型可信的行数
        print(f"[推理统计] 总行数={len(texts)}, 本地模型可信={confident_count}行, "
              f"需转API处理={need_api_count}行")

        # 最终预测标签（先使用 Top-1 类别作为默认值）
        predicted_ids = top2_indices[:, 0].cpu().numpy()

        # 写入置信度到 txt 文件
        result_save_path = CURRENT_DIR / "prediction_confidence.txt"
        with open(result_save_path, "w", encoding="utf-8") as fout:
            for i in range(len(texts)):
                fout.write(f"{top1_probs[i]:.6f}, {top2_probs_val[i]:.6f}\n")
        print(f"[置信度日志] 已将 {len(texts)} 条预测的 Top-1/Top-2 置信度写入: {result_save_path}")

    # 将本地模型的预测结果转换为标签
    predicted_labels = encoder.inverse_transform(predicted_ids)
    local_predictions = predicted_labels.tolist()
    
    # 如果有不确定的样本，调用 API 进行二次分类
    if need_api_count > 0:
        print(f"\n[API 调用] 发现 {need_api_count} 条低置信度样本，开始调用 API 进行分类...")
        
        # 收集需要 API 处理的样本（带原始索引）
        uncertain_samples = []
        for idx, (text, is_conf) in enumerate(zip(valid_texts, is_confident)):
            if not is_conf:
                # display_id 从 1 开始
                uncertain_samples.append((idx + 1, text))
        
        # 调用 API 进行分类
        def api_progress_cb(percent, msg):
            if progress_cb:
                # API 进度占总进度的后 50%
                overall_progress = 50 + int(50 * percent / 100)
                progress_cb(overall_progress, msg)
        
        api_results = _call_api_for_classification(uncertain_samples, progress_cb=api_progress_cb)
        
        # 将 API 的结果映射回原来的位置
        # 使用统一的类别映射管理器，确保与 category_map.json 完全一致
        from mnx.classification.category_mapper import get_api_category_map
        category_map = get_api_category_map()
        
        print(f"[API 映射] 加载了 {len(category_map)} 个类别映射")
        
        # 替换不确定样本的预测结果
        api_used_count = 0
        unmapped_categories = []  # 记录未映射的类别
        
        for display_id, (category, group_type) in api_results.items():
            # display_id 是从 1 开始的，转换为数组索引
            array_idx = display_id - 1
            if 0 <= array_idx < len(local_predictions):
                # 将 API 返回的类别编号转换为类别名称
                category_name = category_map.get(category, None)
                
                if category_name is None:
                    # 类别编号不在映射表中，记录警告
                    unmapped_categories.append(category)
                    print(f"  ⚠️  API 返回的类别编号 {category} 不在 category_map.json 中")
                    # 使用默认类别 0（非公益诉讼）
                    category_name = category_map.get(0, "非公益诉讼")
                
                local_predictions[array_idx] = category_name
                api_used_count += 1
        
        # 如果有未映射的类别，输出警告
        if unmapped_categories:
            unique_unmapped = list(set(unmapped_categories))
            print(f"\n[API 警告] 发现 {len(unmapped_categories)} 条记录的类别编号无法映射:")
            print(f"  未映射的类别编号: {unique_unmapped}")
            print(f"  请检查 API 返回的类别编号是否在 category_map.json 中定义")
        
        print(f"[API 完成] 成功使用 API 重新分类 {api_used_count} 条样本")
    
    if progress_cb:
        progress_cb(100, "✅ 推理完成！")
    
    return local_predictions

# ──────────────────────────────────────────────
# 终端脚本区：保留原程序的完整独立测试功能
# ──────────────────────────────────────────────
def main():
    DATA_FILE = "../dataset/text_true.txt"
    EMBEDDING_SAVE_PATH = "./embeddings_10w21.npy"
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
                # ✅ 修复 Bug：强转为 int，避免 LabelEncoder 字典序排错
                labels.append(int(parts[1].strip()))

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
        17: "控烟", 18: "预付式消费", 19: "拖欠工资", 20: "行政检查"
    }

    print("\n" + "=" * 60)
    print("【第3步】划分训练集 / 测试集...")

    X_train, X_test, y_train, y_test = train_test_split(texts, label_ids, test_size=0.2, random_state=42)

    print("\n" + "=" * 60)
    print(f"【第4步】使用 Qwen3-Embedding-0.6B 生成向量 (设备: {device.upper()})...")

    if os.path.exists(EMBEDDING_SAVE_PATH):
        print(f"  检测到已有缓存文件 {EMBEDDING_SAVE_PATH}")
        all_embeddings = np.load(EMBEDDING_SAVE_PATH)
        model_path = r"../core_model/Qwen3-Embedding-0.6B"
        encoder_model = SentenceTransformer(model_path, device=device)
    else:
        model_path = r"../core_model/Qwen3-Embedding-0.6B"
        encoder_model = SentenceTransformer(model_path, device=device)

        all_texts = X_train + X_test
        start_time = time.time()
        instruction = "query:"
        full_texts = [f"{instruction}{t}" for t in all_texts]

        all_embeddings = encoder_model.encode(full_texts, batch_size=16, show_progress_bar=True,
                                              normalize_embeddings=True)

        elapsed = time.time() - start_time
        print(f"  [OK] 向量生成完成！耗时：{elapsed:.1f} 秒")
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
    print("【第 7 步/第 8 步】配置参数并开始训练...")

    # 计算类别权重 - 只计算训练集中实际存在的类别
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    
    # 如果类别数不匹配，手动补齐权重（处理测试集划分时某些类别未被分到的情况）
    if len(class_weights) < num_classes:
        print(f"  ⚠️  训练集中只有 {len(class_weights)} 个类别，补齐到 {num_classes} 个类别")
        full_weights = np.ones(num_classes)
        for i, cls in enumerate(np.unique(y_train)):
            full_weights[int(cls)] = class_weights[i]
        class_weights = full_weights
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
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
    all_labels = list(range(num_classes))
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0, labels=all_labels))

    print("\n" + "=" * 60)
    print("【混淆矩阵分析】")
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    
    class_accuracy = {}
    for i, label in enumerate(all_labels):
        if cm[i].sum() > 0:
            acc = cm[i][i] / cm[i].sum()
            class_accuracy[label] = acc
    
    sorted_classes = sorted(class_accuracy.items(), key=lambda x: x[1])
    
    print("\n❌ 表现最差的 5 个类别:")
    for label, acc in sorted_classes[:5]:
        class_name = category_names.get(label, f"类别{label}")
        print(f"  {class_name} (标签{label}): 准确率 {acc:.2%}")
        
        # ✅ 致命 Bug 已修复：这里之前是 cm[i]，现在正确改成了 cm[label]
        misclassified = cm[label]
        
        top3_indices = np.argsort(misclassified)[-3:][::-1]
        top3_indices = [idx for idx in top3_indices if idx != label]
        
        if len(top3_indices) > 0:
            print(f"    主要误判为:")
            for idx in top3_indices:
                if misclassified[idx] > 0:
                    print(f"      → {category_names.get(idx, f'类别{idx}')}: {misclassified[idx]} 条")
    
    # 📌 新增：打印所有错判样本详情，方便人工检查
    print("\n" + "=" * 60)
    print("【错判样本详情分析】")
    print("=" * 60)
    
    # 找出所有被误判的样本
    misclassified_indices = np.where(y_test != y_pred)[0]
    print(f"\n共发现 {len(misclassified_indices)} 条被误判的样本\n")
    
    # 按真实标签分组统计
    error_details = {}
    for idx in misclassified_indices:
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        if true_label not in error_details:
            error_details[true_label] = []
        error_details[true_label].append((idx, pred_label))
    
    # 📌 将所有误判样本保存到 txt 文件
    print("\n" + "=" * 60)
    print("【正在保存误判样本到文件...】")
    print("=" * 60)
    
    output_file = "misclassified_samples.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("误判样本完整列表\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"测试集总数：{len(y_test)}\n")
        f.write(f"误判样本数：{len(misclassified_indices)}\n")
        f.write(f"误判率：{len(misclassified_indices)/len(y_test)*100:.2f}%\n")
        f.write(f"整体准确率：{accuracy_score(y_test, y_pred)*100:.2f}%\n\n")
        
        # 按标签排序输出
        for label in sorted(error_details.keys()):
            if label in error_details:
                class_name = category_names.get(label, f"类别{label}")
                f.write(f"\n{'='*80}\n")
                f.write(f"【{class_name} (标签{label})】- 共误判 {len(error_details[label])} 条:\n")
                f.write(f"{'='*80}\n\n")
                
                # 打印该类别所有误判样本
                for i, (idx, pred_label) in enumerate(error_details[label], 1):
                    text = X_test[idx]
                    true_name = category_names.get(label, f"类别{label}")
                    pred_name = category_names.get(pred_label, f"类别{pred_label}")
                    
                    f.write(f"{i}. 真实：{true_name} | 预测：{pred_name}\n")
                    f.write(f"   文本：{text}\n")
                    f.write(f"   索引：{idx}\n\n")
    
    print(f"\n✅ 误判样本已保存到：{output_file}")
    print(f"   共保存 {len(misclassified_indices)} 条误判样本\n")
    


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
