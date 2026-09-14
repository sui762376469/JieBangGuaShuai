from transformers import MT5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from tqdm import tqdm
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
model_heack = MT5ForConditionalGeneration.from_pretrained("./model/Heack-Zhsum").to(device)
model_heack.eval()
tokenizer_heack = T5Tokenizer.from_pretrained("./model/Heack-Zhsum")

def _split_text(text, length):
    chunks = []
    start = 0
    while start < len(text):
        if len(text) - start > length:
            pos_forward = start + length
            pos_backward = start + length
            pos = start + length
            while (pos_forward < len(text)) and (pos_backward >= 0) and (pos_forward < 20 + pos) and  (pos_backward + 20 > pos) and text[pos_forward] not in {'.', '。','，',','} and text[pos_backward] not in {'.', '。','，',','}:
                pos_forward += 1
                pos_backward -= 1
            if pos_forward - pos >= 20 and pos_backward <= pos - 20:
                pos = start + length
            elif text[pos_backward] in {'.', '。','，',','}:
                pos = pos_backward
            else:
                pos = pos_forward
            chunks.append(text[start:pos+1])
            start = pos + 1
        else:
            chunks.append(text[start:])
            break
    # Combine last chunk with previous one if it's too short
    if len(chunks) > 1 and len(chunks[-1]) < 100:
        chunks[-2] += chunks[-1]
        chunks.pop()
    return chunks

def get_summary_heack(text, each_summary_length=150):
    """单条文本摘要（保留兼容）"""
    chunks = _split_text(text, 300)
    summaries = []
    for chunk in chunks:
        inputs = tokenizer_heack.encode("summarize: " + chunk, return_tensors='pt', 
                                        max_length=512, truncation=True).to(device)
        summary_ids = model_heack.generate(inputs, max_length=each_summary_length,
                                           num_beams=8, length_penalty=2, no_repeat_ngram_size=2)
        summary = tokenizer_heack.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return " ".join(summaries)

# ==================== 【新增】高效批量摘要函数 ====================
def batch_get_summaries(texts, each_summary_length=150, batch_size=8):
    """
    高效批量摘要处理：先收集所有分段 → 批量编码 → 批量生成 → 还原
    
    Args:
        texts: list[str] — 待摘要的文本列表
        each_summary_length: int — 每段摘要的最大长度
        batch_size: int — 同时送入模型的批大小（受显存限制）
    
    Returns:
        list[str] — 每个文本对应的摘要字符串列表
    """
    # ---------- 第1步：将所有文本分段 ----------
    print("正在分段...")
    all_segments = []   # 所有分段文本
    seg_to_text_idx = [] # 每个分段属于原始文本的第几个 (映射关系)
    
    for text_idx, text in enumerate(texts):
        text = str(text).strip()
        if not text or text == "nan":
            seg_to_text_idx.append((text_idx, None))  # 标记空文本
            continue
        
        chunks = _split_text(text, 300)
        start_seg = len(all_segments)  # 记录这个文本的分段从哪开始
        for chunk in chunks:
            all_segments.append("summarize: " + chunk)
        
        seg_to_text_idx.append((text_idx, (start_seg, len(chunks))))
    
    total_segs = len(all_segments)
    print(f"共 {len(texts)} 条文本 → 分割成 {total_segs} 个段落")
    
    if total_segs == 0:
        return [""] * len(texts)
    
    # ---------- 第2步：批量编码所有分段 ----------
    print("正在批量编码...")
    encoded = tokenizer_heack(
        all_segments,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # ---------- 第3步：分批生成摘要 ----------
    print("正在批量生成摘要...")
    all_seg_summaries = []
    
    total_batches = (total_segs + batch_size - 1) // batch_size
    
    with tqdm(total=total_batches, desc="生成进度") as pbar:
        for i in range(0, total_segs, batch_size):
            # 取出一个 batch 的输入
            batch_input_ids = encoded["input_ids"][i : i + batch_size]
            batch_attention_mask = encoded["attention_mask"][i : i + batch_size]
            
            # 关键优化！一次性对整个batch进行生成
            with torch.no_grad():
                summary_ids = model_heack.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_length=each_summary_length,
                    num_beams=4,
                    length_penalty=1.5,
                    no_repeat_ngram_size=2,
                )
            
            # 批量解码
            batch_summaries = tokenizer_heack.batch_decode(
                summary_ids, skip_special_tokens=True
            )
            all_seg_summaries.extend(batch_summaries)
            pbar.update(1)
    
    # ---------- 第4步：还原为每个原始文本的摘要 ----------
    print("正在合并结果...")
    results = []
    for text_idx, seg_info in seg_to_text_idx:
        if seg_info is None:
            results.append("")  # 空文本
        else:
            start, count = seg_info
            text_summary = " ".join(all_seg_summaries[start : start + count])
            results.append(text_summary)
    
    return results

def batch_summarize_csv(csv_path, output_path):
    """高效版：批量读取 CSV → 批量摘要 → 保存"""
    df = pd.read_csv(csv_path)
    
    # 收集所有有效文本
    texts = df["诉求描述"].astype(str).tolist()
    total = len(texts)
    print(f"共 {total} 条数据待处理")
    
    # 【核心改动】一次性批量处理所有文本
    summaries = batch_get_summaries(texts, each_summary_length=150, batch_size=8)
    
    # 组装结果
    # results = []
    # for idx, (text, summary) in enumerate(zip(texts, summaries)):
    #     short_preview = text.strip()[:50] + "..." if len(text.strip()) > 50 else text.strip()
    #     results.append({
    #         "序号": idx + 1,
    #         "原文本预览": short_preview,
    #         "摘要": summary
    #     })
    
    # out_df = pd.DataFrame(results)
    df["诉求描述"] = summaries
    out_df = df
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

# ==================== 测试/主程序入口 ====================
if __name__ == "__main__":
    # 单条测试
    chunk = """
道路不平，很多小汽车路过去时有噪声，影响附近居民睡觉
"""
    # 调用摘要函数
    summary = get_summary_heack(chunk, each_summary_length=150)

    print("\n生成的摘要:")
    print(summary)
    print("=" * 60)

    # 批量处理
    batch_summarize_csv("./dataset/上海投诉汇总-标签4.csv", "./output.csv")
    print("\ndeal1.csv处理完成!")
