import pandas as pd
import sys
import re
from .tuomin import mask_text_simple
from tqdm import tqdm


def api_desensitize_df(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    专门为 Streamlit 或其他 Web 框架提供的 API 接口。
    直接在内存中处理 DataFrame，不涉及本地文件系统的读写和终端 input。
    """
    # 拷贝一份数据，避免修改原数据
    df_copy = df.copy()

    if text_column not in df_copy.columns:
        raise ValueError(f"数据中找不到指定的脱敏列: '{text_column}'")

    masked_texts = []

    # 执行脱敏处理
    for text in df_copy[text_column]:
        if pd.isna(text):
            masked_texts.append(text)
        else:
            try:
                text_str = str(text).strip()
                if text_str:
                    masked_text = mask_text_simple(text_str)
                    masked_texts.append(masked_text)
                else:
                    masked_texts.append(text)
            except Exception:
                masked_texts.append(text)  # 出错时保留原文

    # 替换原列为脱敏后的数据
    df_copy[text_column] = masked_texts

    # 清洗所有文本列：去除首尾空格和多余的回车换行
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].apply(lambda x: re.sub(r'\s+', ' ', str(x).strip()) if pd.notna(x) else x)

    return df_copy


def process_excel_file(input_file: str, output_file: str, text_column: str = None):
    """
    （保留原有的本地脚本处理逻辑，供你在线下终端测试时使用）
    """
    # ... 保留你原有的代码 ...
    pass


if __name__ == "__main__":
    # 保留你原有的本地测试代码
    pass