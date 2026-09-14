import os
import time
import numpy as np
import joblib
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from collections import Counter

# ==============================================================
# 核心 API 区：支持 Web 平台静默调用与终端高并发调用
# ==============================================================
_MODEL_CACHE = {}


def get_models():
    """单例模式加载模型，确保大模型只在内存中加载一次"""
    if "lr_model" in _MODEL_CACHE:
        return _MODEL_CACHE["lr_model"], _MODEL_CACHE["label_encoder"], _MODEL_CACHE["embedding_model"]

    # 绝对路径寻址，再也不怕相对路径找错文件
    current_dir = Path(__file__).resolve().parent
    lr_path = current_dir / "lr_model.pkl"
    le_path = current_dir / "label_encoder.pkl"
    embed_path = current_dir.parent / "qwen_model" / "Qwen3-Embedding-0.6B"

    if not lr_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {lr_path}")
    if not embed_path.exists():
        raise FileNotFoundError(f"找不到 Embedding 模型文件夹: {embed_path}")

    lr_model = joblib.load(str(lr_path))
    label_encoder = joblib.load(str(le_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(str(embed_path), device=device)

    _MODEL_CACHE["lr_model"] = lr_model
    _MODEL_CACHE["label_encoder"] = label_encoder
    _MODEL_CACHE["embedding_model"] = embedding_model

    return lr_model, label_encoder, embedding_model


def api_predict_categories(texts, batch_size=256, show_progress_bar=False):
    """
    接收文本列表，返回预测的分类名称列表。
    默认参数 (256, False) 专为 app.py (Web 端) 设计，防止炸显存和刷屏日志。
    但在独立运行时，可以通过传参开启 1024 满血性能和进度条。
    """
    if not texts:
        return []

    lr_model, label_encoder, embedding_model = get_models()

    # 批量生成向量
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True
    )

    # 进行逻辑回归预测并转回中文标签
    predicted_ids = lr_model.predict(embeddings)
    predicted_labels = label_encoder.inverse_transform(predicted_ids)

    return predicted_labels.tolist()


# ==============================================================
# 原脚本区：完全还原你原本的终端独立运行逻辑与体验
# ==============================================================
def main():
    # 使用 pathlib 计算输入输出的绝对路径，确保你在任何终端目录下运行都不会报 FileNotFoundError
    current_dir = Path(__file__).resolve().parent
    INPUT_FILE = current_dir.parent / "dataset" / "source_dataset.txt"
    OUTPUT_FILE = current_dir.parent / "dataset" / "classified_dataset.txt"

    print("=" * 60)
    print("【第1步/第2步】加载模型...")
    try:
        # 预热并获取设备信息
        _, _, embedding_model = get_models()
        print(f"  ✅ 逻辑回归模型加载成功")
        print(f"  ✅ 标签编码器加载成功")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  🌟 当前运行设备: {device.upper()}")
        print(f"  ✅ Embedding 模型加载成功")
    except Exception as e:
        print(f"初始化模型失败: {e}")
        return

    print("\n" + "=" * 60)
    print("【第3步】读取待分类文本...")
    if not INPUT_FILE.exists():
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    texts = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        start_idx = 1 if (lines and lines[0].strip().lower() == "content") else 0
        for i, line in enumerate(lines[start_idx:], start=start_idx + 1):
            line = line.strip()
            if line: texts.append(line)

    print(f"  共读取 {len(texts)} 条待分类文本")
    print(f"  示例文本:")
    for i, text in enumerate(texts[:3]):
        print(f"    [{i + 1}] {text[:50]}..." if len(text) > 50 else f"    [{i + 1}] {text}")

    print("\n" + "=" * 60)
    print("【第4步/第5步】生成文本向量并进行预测...")
    start_time = time.time()

    # 🎯 核心精髓：如果是终端独立运行，直接拉满性能配置（还原你原来的代码设置）
    predicted_labels = api_predict_categories(texts, batch_size=1024, show_progress_bar=True)

    elapsed = time.time() - start_time
    print(f"  ✅ 预测完成！耗时: {elapsed:.2f} 秒")
    print(f"  预测结果示例:")
    for i in range(min(5, len(texts))):
        print(f"    [{i + 1}] {predicted_labels[i]}\t{texts[i][:40]}...")

    label_counts = Counter(predicted_labels)
    print(f"\n  预测类别分布:")
    for label, count in label_counts.most_common():
        print(f"    {label}: {count} 条 ({count / len(texts) * 100:.2f}%)")

    print("\n" + "=" * 60)
    print("【第6步】保存分类结果...")

    # 确保保存路径的文件夹存在
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("label\tcontent\n")
        for label, text in zip(predicted_labels, texts):
            f.write(f"{label}\t{text}\n")

    print(f"  ✅ 结果已保存到: {OUTPUT_FILE}")
    print(f"  共保存 {len(texts)} 条分类结果")

    print("\n" + "=" * 60)
    print("【总结】")
    print("=" * 60)
    print(f"  输入文件: {INPUT_FILE}")
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"  处理文本数: {len(texts)}")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  输出格式: label\\tcontent")
    print("\n🎉 全部完成！")
    print("=" * 60)


# ==============================================================
# 这个防护盾保证了被 app.py `import` 时，上述测试代码绝对不会触发
# ==============================================================
if __name__ == "__main__":
    main()