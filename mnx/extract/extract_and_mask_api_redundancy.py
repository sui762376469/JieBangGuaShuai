import json
import pandas as pd
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm  # 加这一行

# ============ 全局只加载一次模型 ============
ner_pipeline = pipeline(
    task=Tasks.named_entity_recognition,
    model="./model/raner_news"
)

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
        # 取前 max_len 个字符，寻找最后一个逗号或句号
        chunk = remaining[:max_len]
        
        # 从后向前查找逗号或句号
        last_split_pos = -1
        for delimiter in ['。', '，', ',', '.']:
            pos = chunk.rfind(delimiter)
            if pos != -1 and pos > last_split_pos:
                last_split_pos = pos
        
        if last_split_pos == -1:
            # 没找到标点符号，强制在 max_len 处截断
            last_split_pos = max_len
        else:
            # 包含标点符号本身
            last_split_pos += 1
        
        segments.append(remaining[:last_split_pos])
        remaining = remaining[last_split_pos:]

    # 添加最后剩余的部分
    if remaining:
        segments.append(remaining)

    return segments

def extract_entities_batch(texts, batch_size=32):
    """批量提取实体，返回 list[dict] + 进度条"""
    all_entities = []
    total = len(texts)

    # 主进度条
    with tqdm(total=total, desc="实体提取进度", unit="条") as pbar:
        for i in range(0, total, batch_size):
            batch_texts = texts[i: i + batch_size]

            for text in batch_texts:
                entities = {"PER": [], "LOC": [], "ORG": []}
                
                # 对长文本进行智能分段
                segments = split_text_by_punctuation(text.strip(), max_len=500)
                
                for segment in segments:
                    result = ner_pipeline(segment)

                    if isinstance(result, dict) and "output" in result:
                        for entity in result["output"]:
                            entity_type = entity["type"]
                            span = entity["span"]
                            if entity_type in entities:
                                if span not in entities[entity_type]:
                                    entities[entity_type].append(span)

                all_entities.append(entities)
                pbar.update(1)  # 进度条 +1


    return all_entities

def mask_text(text, entities):
    """将实体替换为等长X + 进度条"""
    all_spans = []
    for spans in entities.values():
        all_spans.extend(spans)
    all_spans.sort(key=len, reverse=True)

    # 替换进度条
    for span in tqdm(all_spans, desc="脱敏替换进度", unit="实体"):
        text = text.replace(span, "X" * len(span))
    return text

def remove_text(text, entities):
    """删除实体 + 进度条"""
    all_spans = []
    for spans in entities.values():
        all_spans.extend(spans)
    all_spans.sort(key=len, reverse=True)

    # 删除进度条
    for span in tqdm(all_spans, desc="删除实体进度", unit="实体"):
        text = text.replace(span, "")
    return text

def mask_text_with_type(text, entities):
    """
    按实体类型替换为对应标签：
      PER -> [人名]
      LOC -> [位置]
      ORG -> [机构名]
    """
    type_to_label = {
        "PER": "[人名]",
        "LOC": "[位置]",
        "ORG": "[机构名]"
    }

    # 收集所有 (span, label)，按长度降序排列，防止短串误匹配
    all_items = []
    for entity_type, spans in entities.items():
        label = type_to_label.get(entity_type)
        if label:
            for span in spans:
                all_items.append((span, label))
    all_items.sort(key=lambda x: len(x[0]), reverse=True)

    for span, label in tqdm(all_items, desc="类型替换进度", unit="实体"):
        text = text.replace(span, label)

    return text

def main():
    # excel_file = "./dataset/上海投诉汇总-标签5.xlsx"
    # df = pd.read_excel(excel_file)
    excel_file = "./dataset/测试专用1 - 副本.xlsx_address.csv"
    df = pd.read_csv(excel_file, encoding='utf-8')


    line = "content"
    if line not in df.columns:
        raise ValueError("Excel文件中未找到'{}'列".format(line))

    # ---------- 预处理：收集有效文本及其行号 ----------
    valid_indices = []
    valid_texts = []
    for idx, row in df.iterrows():
        text = str(row[line])
        if pd.isna(row[line]) or text == "nan" or text.strip() == "":
            continue
        valid_indices.append(idx)
        valid_texts.append(text)

    print(f"共 {len(valid_texts)} 条有效文本，开始推理...")

    # ---------- 一次推理，拿到所有实体 ----------
    all_entities = extract_entities_batch(valid_texts, batch_size=32)

    # ---------- 生成三种输出 ----------

    # 1) JSON — 实体提取结果
    result_dict = {}
    for idx, entities in zip(valid_indices, all_entities):
        result_dict[f"l{idx + 1}"] = entities
    with open("./result.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    print("结果已保存到 ./result.json")

    # 2) CSV — 实体替换为 X
    masked = df.copy()
    for idx, entities in zip(valid_indices, all_entities):
        masked.at[idx, line] = mask_text(str(masked.at[idx, line]), entities)
    masked.to_csv("./deal1.csv", index=False, encoding="utf-8-sig")
    print("结果已保存到 ./deal1.csv")

    # 3) CSV — 删除实体
    removed = df.copy()
    for idx, entities in zip(valid_indices, all_entities):
        removed.at[idx, line] = remove_text(str(removed.at[idx, line]), entities)
    removed.to_csv("./deal2.csv", index=False, encoding="utf-8-sig")
    print("结果已保存到 ./deal2.csv")

    # 4) CSV — 实体替换为类型标签 [人名]/[位置]/[机构名]
    labeled = df.copy()
    for idx, entities in zip(valid_indices, all_entities):
        labeled.at[idx, line] = mask_text_with_type(str(labeled.at[idx, line]), entities)
    labeled.to_csv("./deal3_labeled.csv", index=False, encoding="utf-8-sig")
    print("结果已保存到 ./deal3_labeled.csv")
    print("\n所有处理完成!")


if __name__ == "__main__":
    main()
