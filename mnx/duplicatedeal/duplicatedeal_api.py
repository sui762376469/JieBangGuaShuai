import pandas as pd
import os

def load_data(file_path):
    """自动识别 .csv 或 .xlsx/.xls 并读取"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif ext == '.csv':
        return pd.read_csv(file_path, encoding='utf-8-sig')
    else:
        raise ValueError(f"不支持的文件格式：{ext}，请使用 .csv 或 .xlsx")

def api_duplicatedeal_api(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    对 DataFrame 按指定文本列进行去重
    
    Args:
        df: 输入的原始 DataFrame
        text_column: 用于去重的文本列名
        
    Returns:
        去重后的 DataFrame
    """
    if df.empty or text_column not in df.columns:
        return df

    original_count = len(df)
    
    # 取前29个字符作为内容指纹进行去重
    df['内容指纹'] = df[text_column].astype(str).str[:29]
    final_df = df.drop_duplicates(subset=['内容指纹'], keep='first')
    final_df = final_df.drop(columns=['内容指纹'])
    final_df = final_df.reset_index(drop=True)
    
    removed_count = original_count - len(final_df)
    print(f"\n✅ 去重完成！原始 {original_count} 条 → 去重后 {len(final_df)} 条 (去除 {removed_count} 条重复)")
    
    return final_df, removed_count


if __name__ == "__main__":
    # 本地测试用
    input_file = './上海投诉汇总-标签4.xlsx'
    df = load_data(input_file)
    result_df, removed = api_duplicatedeal_api(df, '诉求描述')
    result_df.to_csv('去重结果.csv', index=False, encoding='utf-8-sig')