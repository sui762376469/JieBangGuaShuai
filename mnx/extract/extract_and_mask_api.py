# This file use Ernie-ner model to process the text
import json
from re import S
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path
import warnings


BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model" / "Ernie-ner"

# ============ 延迟加载模型 ============
_ner_pipeline = None

def get_ner_pipeline():
    """
    延迟加载 NER 模型，只在首次调用时加载
    如果模型文件损坏，返回 None 并给出明确提示
    """
    global _ner_pipeline
    
    if _ner_pipeline is not None:
        return _ner_pipeline
    
    # 检查模型目录是否存在且包含必要文件
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"NER 模型目录不存在: {MODEL_DIR}\n"
            f"请确保已下载 Ernie-ner 模型到该目录"
        )
    
    config_file = MODEL_DIR / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"NER 模型配置文件不存在: {config_file}\n"
            f"模型文件可能损坏或未正确下载，请重新下载模型"
        )
    
    try:
        print(f"[NER 模型] 正在加载模型: {MODEL_DIR}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ner_pipeline = pipeline(
                "token-classification",
                model=str(MODEL_DIR),
                aggregation_strategy="simple"
            )
        print(f"[NER 模型] ✅ 模型加载成功")
        return _ner_pipeline
    except Exception as e:
        raise RuntimeError(
            f"NER 模型加载失败: {e}\n"
            f"模型文件可能损坏，请检查 {MODEL_DIR} 目录\n"
            f"建议：重新下载完整的 Ernie-ner 模型文件"
        ) from e

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
    """
    将长文本按标点符号智能分割，每段不超过 max_len 字符。
    在 max_len 之前的最后一个逗号(，)或句号(。)处进行分割。
    """
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


def extract_entities_batch(texts, batch_size=32, progress_callback=None):
    """批量提取实体，返回 list[dict] + 进度条"""
    # 延迟加载模型
    ner_pipeline = get_ner_pipeline()
    
    all_entities = []
    total = len(texts)

    with tqdm(total=total, desc="实体提取进度", unit="条") as pbar:
        for i in range(0, total, batch_size):
            batch_texts = texts[i: i + batch_size]

            for text in batch_texts:
                # 统一使用内部类型作为key
                entities = {"PER": [], "LOC": [], "ORG": [], "EMAIL": [], "NUMBER": []}
                
                segments = split_text_by_punctuation(text.strip(), max_len=500)
                
                for segment in segments:
                    results = ner_pipeline(segment)

                    if isinstance(results, list):
                        for entity in results:
                            entity_group = entity.get("entity_group", "").lower()
                            if entity_group not in INTERESTED_TYPES:
                                continue

                            # 去除word中的空格（Ernie-ner输出带空格）
                            word = entity.get("word", "").replace(" ", "")
                            if not word:
                                continue

                            unified_type = ENTITY_TYPE_MAP[entity_group]
                            if word not in entities[unified_type]:
                                entities[unified_type].append(word)

                all_entities.append(entities)
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