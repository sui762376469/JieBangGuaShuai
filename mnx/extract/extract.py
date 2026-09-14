import json
import pandas as pd
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm  # 加这一行
# ============ Presidio 敏感信息提取 ============
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer import RecognitionResult
# ============ 全局只加载一次模型 ============
ner_pipeline = pipeline(
    task=Tasks.named_entity_recognition,
    model="./model/raner_news",
)



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
                result = ner_pipeline(text.strip()[:510])

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

def main():
    excel_file = "./dataset/上海投诉汇总-标签4.xlsx"
    df = pd.read_excel(excel_file)

    if "诉求描述" not in df.columns:
        raise ValueError("Excel文件中未找到'诉求描述'列")

    # ---------- 预处理：收集有效文本及其行号 ----------
    valid_indices = []
    valid_texts = []
    for idx, row in df.iterrows():
        text = str(row["诉求描述"])
        if pd.isna(row["诉求描述"]) or text == "nan" or text.strip() == "":
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
        masked.at[idx, "诉求描述"] = mask_text(str(masked.at[idx, "诉求描述"]), entities)
    masked.to_csv("./deal1.csv", index=False, encoding="utf-8-sig")
    print("结果已保存到 ./deal1.csv")

    # 3) CSV — 删除实体
    removed = df.copy()
    for idx, entities in zip(valid_indices, all_entities):
        removed.at[idx, "诉求描述"] = remove_text(str(removed.at[idx, "诉求描述"]), entities)
    removed.to_csv("./deal2.csv", index=False, encoding="utf-8-sig")
    print("结果已保存到 ./deal2.csv")

    print("\n所有处理完成!")


if __name__ == "__main__":
    main()
