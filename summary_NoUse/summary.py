from transformers import MT5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from tqdm import tqdm

model_heack = MT5ForConditionalGeneration.from_pretrained("./model/Heack-Zhsum")
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
    chunks = _split_text(text, 300)
    summaries = []
    for chunk in chunks:
        inputs = tokenizer_heack.encode("summarize: " + chunk, return_tensors='pt', max_length=512, truncation=True)
        summary_ids = model_heack.generate(inputs, max_length=each_summary_length, num_beams=4, length_penalty=1.5, no_repeat_ngram_size=2)
        summary = tokenizer_heack.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return " ".join(summaries)

def batch_summarize_csv(csv_path, output_path):
    df = pd.read_csv(csv_path)
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="摘要进度"):
        text = str(row["诉求描述"]).strip()
        if text and text != "nan":
            summary = get_summary_heack(text, each_summary_length=150)
        else:
            summary = ""
        results.append({"原文本": text[:50] + "...", "摘要": summary})
    
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已保存到 {output_path}")

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
    batch_summarize_csv("./deal1.csv", "./output.csv")
    print("\ndeal1.csv处理完成!")
