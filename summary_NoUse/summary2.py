import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import pandas as pd
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import gc

# ==================== 设备与精度配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = True  # 8G GPU 必开，显存占用减半

print(f"使用设备: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # print(f"总显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# ==================== 模型加载 ====================
model_path = "./model/Heack-Zhsum"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

model = model.to(DEVICE)

# ★ 关键优化1: FP16 推理，显存占用减半
if USE_FP16 and DEVICE.type == "cuda":
    model = model.half()  # FP32 → FP16

model.eval()  # ★ 关键优化2: 关闭 dropout/BN 训练模式

# ★ 关键优化3: 禁用梯度计算，节省大量显存
torch.set_grad_enabled(False)


# ==================== 文本分块（保留原逻辑） ====================
def _split_text(text, length):
    chunks = []
    start = 0
    while start < len(text):
        if len(text) - start > length:
            pos_forward = start + length
            pos_backward = start + length
            pos = start + length
            while (pos_forward < len(text)) and (pos_backward >= 0) \
                    and (pos_forward < 20 + pos) and (pos_backward + 20 > pos) \
                    and text[pos_forward] not in {'.', '。', '，', ','} \
                    and text[pos_backward] not in {'.', '。', '，', ','}:
                pos_forward += 1
                pos_backward -= 1
            if pos_forward - pos >= 20 and pos_backward <= pos - 20:
                pos = start + length
            elif text[pos_backward] in {'.', '。', '，', ','}:
                pos = pos_backward
            else:
                pos = pos_forward
            chunks.append(text[start:pos + 1])
            start = pos + 1
        else:
            chunks.append(text[start:])
            break
    if len(chunks) > 1 and len(chunks[-1]) < 100:
        chunks[-2] += chunks[-1]
        chunks.pop()
    return chunks


# ==================== ★ 核心优化：批量推理 ====================
def batch_generate_summaries(chunks, batch_size=8, max_input_len=512, max_output_len=150):
    """
    将所有 chunk 按 batch_size 分批推理，而非逐条推理
    8G GPU 建议 batch_size=4~12（取决于模型大小和输入长度）
    """
    all_summaries = []

    for i in range(0, len(chunks), batch_size):
        batch_texts = chunks[i:i + batch_size]
        prefixed = ["summarize: " + t for t in batch_texts]

        # ★ 批量 tokenize
        inputs = tokenizer(
            prefixed,
            return_tensors="pt",
            max_length=max_input_len,
            truncation=True,
            padding=True,        # ★ 关键：padding 到同一长度
        ).to(DEVICE)

        # ★ 批量 generate
        with torch.cuda.amp.autocast(enabled=USE_FP16):  # ★ AMP 自动混合精度
            summary_ids = model.generate(
                **inputs,
                max_length=max_output_len,
                num_beams=4,
                length_penalty=1.5,
                no_repeat_ngram_size=2,
            )

        # ★ 批量 decode
        batch_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        all_summaries.extend(batch_summaries)

        # ★ 定期清理显存碎片
        del inputs, summary_ids
        if DEVICE.type == "cuda" and (i // batch_size) % 20 == 0:
            torch.cuda.empty_cache()

    return all_summaries


# ==================== 单条文本摘要 ====================
def get_summary(text, each_summary_length=150):
    chunks = _split_text(text, 300)
    if not chunks:
        return ""

    # ★ 单条文本的多个 chunk 也批量处理
    summaries = batch_generate_summaries(
        chunks,
        batch_size=len(chunks),  # 单条内所有 chunk 一起推理
        max_output_len=each_summary_length,
    )
    return " ".join(summaries)


# ==================== ★ CSV 批量处理（两层批处理） ====================
def batch_summarize_csv(csv_path, output_path, row_batch_size=16):
    """
    两层批处理：
      - 外层：按行分批读取 CSV（控制内存）
      - 内层：每行的 chunks 也批量推理（控制显存）
    """
    df = pd.read_csv(csv_path)
    results = []

    # ★ 先收集所有需要处理的 chunk，按全局 batch 推理
    all_row_chunks = []    # [(row_idx, [chunk1, chunk2, ...]), ...]
    for idx, row in df.iterrows():
        text = str(row["诉求描述"]).strip()
        if text and text != "nan":
            chunks = _split_text(text, 300)
            all_row_chunks.append((idx, text, chunks))
        else:
            results.append({"原文本": "", "摘要": ""})

    # ★ 全局批量推理：把所有 chunks 扁平化，一起 batch
    total_chunks = sum(len(rc[2]) for rc in all_row_chunks)
    print(f"共 {len(all_row_chunks)} 条文本，{total_chunks} 个分块待处理")

    # 方案A：逐行处理但 chunk 内批量（显存更可控）
    for idx, text, chunks in tqdm(all_row_chunks, desc="摘要进度"):
        summaries = batch_generate_summaries(
            chunks,
            batch_size=8,      # ★ 8G GPU 推荐值，可调
            max_output_len=150,
        )
        summary = " ".join(summaries)
        results.append({
            "原文本": text[:50] + "...",
            "摘要": summary
        })

        # ★ 每处理 N 条清理一次显存
        if idx % 50 == 0 and DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已保存到 {output_path}")


# ==================== ★ 显存监控工具 ====================
def print_gpu_memory():
    if DEVICE.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 显存 — 已分配: {allocated:.2f} GB | 已预留: {reserved:.2f} GB")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 查看初始显存
    print_gpu_memory()

    # 单条测试
    chunk = "道路不平，很多小汽车路过去时有噪声，影响附近居民睡觉"
    summary = get_summary(chunk, each_summary_length=150)
    print("\n生成的摘要:")
    print(summary)
    print("=" * 60)
    print_gpu_memory()

    # 批量处理
    batch_summarize_csv("./deal1.csv", "./output.csv")
    print("\ndeal1.csv 处理完成!")
    print_gpu_memory()
