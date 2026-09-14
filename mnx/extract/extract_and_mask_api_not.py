# This file used Gpu to parallel processing and use Ernie-ner model
import json
from re import S
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path
import torch

# ============ 全局只加载一次模型 ============
device = 0 if torch.cuda.is_available() else -1
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model" / "Ernie-ner"

# ============ 全局只加载一次模型 ============
ner_pipeline = pipeline(
    "token-classification",
    model=str(MODEL_DIR),
    aggregation_strategy="simple",
    device=device,
)

# 关心的实体类型（小写，用于过滤）
INTERESTED_TYPES = {"name", "organization", "address", "email", "qq", "wx"}


# 实体类型 -> 内部统一类型
ENTITY_TYPE_MAP = {
    "name": "PER",
    "organization": "ORG",
    "address": "LOC",
    "email": "EMAIL",
    "qq": "NUMBER",
    "wx": "NUMBER",
}

# 统一类型 -> 脱敏标签
TYPE_TO_LABEL = {
    "PER": "[人名]",
    "ORG": "[机构名]",
    "LOC": "[位置]",
    "EMAIL": "[邮箱]",
    "NUMBER": "[号码]",
}


def split_text_by_punctuation(text, max_len=512):
    if len(text) <= max_len:
        return [text]
    segments = []
    remaining = text

    while len(remaining) > max_len:
        chunk = remaining[:max_len]
        
        last_split_pos = -1
        for delimiter in ['。', '，', ',', '.']:
            pos = chunk.rfind(delimiter)
            if pos != -1 and pos > last_split_pos:
                last_split_pos = pos
        
        if last_split_pos == -1:
            last_split_pos = max_len
        else:
            last_split_pos += 1
        
        segments.append(remaining[:last_split_pos])
        remaining = remaining[last_split_pos:]

    if remaining:
        segments.append(remaining)

    return segments


from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

def extract_entities_batch(texts, batch_size=32, progress_callback=None):
    """
    使用 Dataset API 批量提取实体，真正利用 GPU 并行推理。
    返回 list[dict]，每个 dict 对应一条原始文本的实体。
    """
    total = len(texts)
    
    # ===== 第一步：展平所有文本段落，记录归属关系 =====
    all_segments = []
    segment_owner = []
    
    for text_idx, text in enumerate(texts):
        segments = split_text_by_punctuation(text.strip(), max_len=512)
        for seg in segments:
            all_segments.append(seg)
            segment_owner.append(text_idx)

    print(f"共 {total} 条原始文本 → 切分为 {len(all_segments)} 个片段，开始批量推理...")
    
    # 构造 HuggingFace Dataset
    ds = Dataset.from_dict({"text": all_segments, "owner_idx": segment_owner})

    # ===== 第二步：使用 KeyDataset 批量推理 + 统一进度条 =====
    all_entities = [
        {"PER": [], "LOC": [], "ORG": [], "EMAIL": [], "NUMBER": []}
        for _ in range(total)
    ]

    # ✅ 关键改进：用 tqdm 包装 pipeline 生成器，实时显示推理+整理一体化进度
    ner_iter = ner_pipeline(
        KeyDataset(ds, "text"),
        batch_size=batch_size,
    )

    with tqdm(total=len(all_segments), desc="实体提取进度", unit="段") as pbar:
        for seg_idx, results in enumerate(ner_iter):
            owner = segment_owner[seg_idx]
            entities = all_entities[owner]

            if isinstance(results, list):
                for entity in results:
                    entity_group = entity.get("entity_group", "").lower()
                    if entity_group not in INTERESTED_TYPES:
                        continue

                    word = entity.get("word", "").replace(" ", "")
                    if not word:
                        continue

                    unified_type = ENTITY_TYPE_MAP[entity_group]
                    if word not in entities[unified_type]:
                        entities[unified_type].append(word)

            pbar.update(1)
            
            if progress_callback:
                progress_callback(1)

    return all_entities


def mask_text_with_type(text, entities):
    """
    按实体类型替换为对应标签：
      PER -> [人名]
      LOC -> [位置]
      ORG -> [机构名]
      EMAIL -> [邮箱]
      NUMBER -> [号码]
    """
    all_items = []
    for entity_type, spans in entities.items():
        label = TYPE_TO_LABEL.get(entity_type)
        if label:
            for span in spans:
                all_items.append((span, label))
    all_items.sort(key=lambda x: len(x[0]), reverse=True)

    for span, label in tqdm(all_items, desc="类型替换进度", unit="实体"):
        text = text.replace(span, label)

    return text


def api_extract_and_mask(df, text_column="content", progress_callback=None):
    """
    对DataFrame中的文本列进行实体提取和脱敏替换。
    返回脱敏后的DataFrame，并保存result.json和deal_labeled.csv到当前目录。
    """
    if text_column not in df.columns:
        raise ValueError(f"DataFrame中未找到'{text_column}'列")

    valid_indices = []
    valid_texts = []
    for idx, row in df.iterrows():
        text = str(row[text_column])
        if pd.isna(row[text_column]) or text == "nan" or text.strip() == "":
            continue
        valid_indices.append(idx)
        valid_texts.append(text)

    print(f"共 {len(valid_texts)} 条有效文本，开始推理...")

    all_entities = extract_entities_batch(valid_texts, batch_size=32, progress_callback=progress_callback)

    result_dict = {}
    for idx, entities in zip(valid_indices, all_entities):
        result_dict[f"l{idx + 1}"] = entities
    with open(str(BASE_DIR / "result.json"), "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    print("结果已保存到" + str(BASE_DIR / "result.json"))

    labeled = df.copy()
    for idx, entities in zip(valid_indices, all_entities):
        labeled.at[idx, text_column] = mask_text_with_type(str(labeled.at[idx, text_column]), entities)
    labeled.to_csv(str(BASE_DIR / "deal_labeled.csv"), index=False, encoding="utf-8-sig")
    print("结果已保存到" + str(BASE_DIR / "deal_labeled.csv"))
    print("\n所有处理完成!")

    return labeled


def main():
    excel_file = "./dataset/测试专用1 - 副本.xlsx_address.csv"
    df = pd.read_csv(excel_file, encoding='utf-8')
    api_extract_and_mask(df, text_column="content")


if __name__ == "__main__":
    main()