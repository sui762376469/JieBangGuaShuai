import pandas as pd
import numpy as np
import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 初始化NER模型
ner_pipeline = pipeline(
    task=Tasks.named_entity_recognition,
    model="./model/raner_news",
)


def extract_entities(text):
    """从文本中提取人名(PER)、地名(LOC)、机构名(ORG)"""
    text = text.strip()[:510]
    result = ner_pipeline(text)
    entities = {"PER": [], "LOC": [], "ORG": []}

    if isinstance(result, dict) and "output" in result:
        for entity in result["output"]:
            entity_type = entity["type"]
            span = entity["span"]
            if entity_type in entities:
                if span not in entities[entity_type]:
                    entities[entity_type].append(span)

    return entities


def process_excel_to_json(excel_path, output_json_path):
    """读取Excel文件，提取实体，保存为JSON格式"""
    df = pd.read_excel(excel_path)

    if "诉求描述" not in df.columns:
        raise ValueError("Excel文件中未找到'诉求描述'列")

    result_dict = {}

    for idx, row in df.iterrows():
        text = str(row["诉求描述"])
        if pd.isna(text) or text == "nan" or text.strip() == "":
            continue

        line_key = f"l{idx + 1}"
        entities = extract_entities(text)
        result_dict[line_key] = entities

        print(f"处理第 {idx + 1} 行完成")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    print(f"结果已保存到 {output_json_path}")
    return result_dict


def mask_entities_with_x(excel_path, output_csv_path):
    """将实体名称替换为X号"""
    df = pd.read_excel(excel_path)

    if "诉求描述" not in df.columns:
        raise ValueError("Excel文件中未找到'诉求描述'列")

    processed_texts = []

    for idx, row in df.iterrows():
        text = str(row["诉求描述"])
        if pd.isna(text) or text == "nan" or text.strip() == "":
            processed_texts.append(text)
            continue

        entities = extract_entities(text)
        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)

        all_entities.sort(key=len, reverse=True)

        masked_text = text
        for entity in all_entities:
            masked_text = masked_text.replace(entity, "X" * len(entity))

        processed_texts.append(masked_text)

    output_df = df.copy()
    output_df["诉求描述"] = processed_texts
    output_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"替换为X的结果已保存到 {output_csv_path}")


def remove_entities(excel_path, output_csv_path):
    """删除实体名称"""
    df = pd.read_excel(excel_path)

    if "诉求描述" not in df.columns:
        raise ValueError("Excel文件中未找到'诉求描述'列")

    processed_texts = []

    for idx, row in df.iterrows():
        text = str(row["诉求描述"])
        if pd.isna(text) or text == "nan" or text.strip() == "":
            processed_texts.append(text)
            continue

        entities = extract_entities(text)
        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)

        all_entities.sort(key=len, reverse=True)

        cleaned_text = text
        for entity in all_entities:
            cleaned_text = cleaned_text.replace(entity, "")

        processed_texts.append(cleaned_text)

    output_df = df.copy()
    output_df["诉求描述"] = processed_texts
    output_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"删除实体的结果已保存到 {output_csv_path}")


if __name__ == "__main__":
    excel_file = "./dataset/上海投诉汇总-标签4.xlsx"

    print("=" * 50)
    print("步骤1: 提取实体并保存为JSON")
    print("=" * 50)
    result_json = process_excel_to_json(excel_file, "./result.json")

    print("\n" + "=" * 50)
    print("步骤2: 将实体替换为X号")
    print("=" * 50)
    mask_entities_with_x(excel_file, "./deal1.csv")

    print("\n" + "=" * 50)
    print("步骤3: 删除实体名称")
    print("=" * 50)
    remove_entities(excel_file, "./deal2.csv")

    print("\n所有处理完成!")
