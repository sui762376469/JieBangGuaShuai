"""
基于大模型的公益/私益诉讼分类模块
使用OpenAI兼容API进行智能分类
"""
import os
import pandas as pd
import json
from typing import List, Tuple
from openai import OpenAI
from tqdm import tqdm
import time

# API配置 - 使用NVIDIA API
base_url = os.getenv("CLASSIFICATION_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
OPENAI_API_KEY = os.getenv("CLASSIFICATION_API_KEY", "")
OPENAI_MODEL = os.getenv("CLASSIFICATION_API_MODEL", "meta/llama-3.2-1b-instruct")

# 初始化OpenAI客户端（添加超时配置）
client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url, timeout=30.0) if OPENAI_API_KEY else None

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的法律文本分类助手，专门用于区分公益诉讼和私益诉讼。

## 分类标准

### 公益诉讼（涉及公共利益）
- 环境污染：油烟、噪音、污染、异味、废气、污水、垃圾等
- 公共安全：安全隐患、消防、危险、道路破损、路灯、红绿灯等
- 公共设施：公园、广场、公共厕所、停车场、公交站、供水供电等
- 教育管理：学校、幼儿园、招生、学位、教学质量等
- 医疗卫生：医院、卫生院、医疗、卫生、防疫、医保等
- 城市规划：违建、规划、拆迁、绿化、市容、占道经营等
- 社会保障：养老金、社保、低保、救助、残疾人权益等
- 其他影响多数人利益的事项

### 私益诉讼（涉及个人利益）
- 合同纠纷：合同、协议、违约、退款、赔偿、欠款等
- 个人财产：房产、物业、车位、装修、漏水、产权等
- 劳动纠纷：工资、加班、辞退、工伤、劳动合同等
- 消费纠纷：商品质量、售后服务、退货、消费者等
- 邻里纠纷：邻居、楼上楼下、私人、个人、我家等
- 婚姻家庭：离婚、抚养、赡养、继承、家庭暴力等
- 其他仅涉及个人或少数人利益的事项

## 输出格式
请严格以JSON格式输出，包含以下字段：
{
    "category": "公益" 或 "私益",
    "confidence": 0.0-1.0之间的浮点数,
    "reason": "简短的分类理由（不超过50字）"
}

## 注意事项
1. 仔细分析文本的核心诉求
2. 判断影响范围是公众还是个人
3. 置信度根据判断的确定性给出
4. 如果难以判断，倾向于归类为"公益"
"""

def classify_with_llm(text: str, max_retries: int = 3) -> Tuple[str, float, str]:
    """
    使用大模型对单条文本进行分类
    
    Args:
        text: 投诉文本
        max_retries: 最大重试次数
        
    Returns:
        Tuple[str, float, str]: (分类结果, 置信度, 理由)
    """
    if not text or pd.isna(text):
        return "未知", 0.0, "文本为空"

    if client is None:
        return "未知", 0.0, "未配置 CLASSIFICATION_API_KEY"
    
    # 限制文本长度，避免超出token限制
    if len(text) > 1000:
        text = text[:1000] + "..."
    
    user_prompt = f"""请对以下投诉内容进行分类：

投诉内容：
{text}

请按照要求输出JSON格式的分类结果。"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # 较低温度以获得更稳定的结果
                max_tokens=200,
                timeout=30,  # 30秒超时
                response_format={"type": "json_object"}  # 强制JSON输出
            )
            
            result_text = response.choices[0].message.content
            
            # 解析JSON结果
            result = json.loads(result_text)
            
            category = result.get("category", "未知")
            confidence = result.get("confidence", 0.5)
            reason = result.get("reason", "")
            
            # 验证结果有效性
            if category not in ["公益", "私益"]:
                category = "未知"
            
            if not (0.0 <= confidence <= 1.0):
                confidence = 0.5
            
            return category, float(confidence), reason
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # 重试前等待1秒
                continue
            else:
                print(f"分类失败: {str(e)}")
                return "未知", 0.0, f"分类出错: {str(e)}"


def api_classify_public_private_llm(df: pd.DataFrame, 
                                     text_column: str = "content",
                                     num_samples: int = 200,
                                     batch_size: int = 10,
                                     delay_between_batches: float = 1.0) -> pd.DataFrame:
    """
    使用大模型对DataFrame中的前num_samples条数据进行公益/私益分类
    
    Args:
        df: 输入DataFrame
        text_column: 文本列名
        num_samples: 需要分类的样本数量（默认前200条）
        batch_size: 每批处理的数量
        delay_between_batches: 批次间延迟（秒），避免API限流
        
    Returns:
        添加了'诉讼类型'、'置信度'和'分类理由'列的DataFrame
    """
    if text_column not in df.columns:
        raise ValueError(f"DataFrame中未找到'{text_column}'列")
    
    # 创建副本避免修改原数据
    result_df = df.copy()
    
    # 初始化新列
    result_df['诉讼类型'] = '待分类'
    result_df['置信度'] = 0.0
    result_df['分类理由'] = ''
    
    # 只对前num_samples条数据进行分类
    actual_samples = min(num_samples, len(result_df))
    
    print(f"正在使用大模型对前 {actual_samples} 条数据进行公益/私益分类...")
    print(f"批次大小: {batch_size}, 预计调用API次数: {actual_samples}")
    print(f"预计耗时: 约 {actual_samples * 1:.0f}-{actual_samples * 2:.0f} 秒\n")
    
    with tqdm(total=actual_samples, desc="LLM分类进度", unit="条") as pbar:
        for i in range(0, actual_samples, batch_size):
            batch_end = min(i + batch_size, actual_samples)
            batch_indices = range(i, batch_end)
            
            # 显示当前批次信息
            current_batch = i // batch_size + 1
            total_batches = (actual_samples + batch_size - 1) // batch_size
            print(f"\n📦 处理批次 {current_batch}/{total_batches} (第{i+1}-{batch_end}条)")
            
            for idx in batch_indices:
                text = str(result_df.iloc[idx][text_column])
                
                try:
                    # 调用大模型分类
                    category, confidence, reason = classify_with_llm(text)
                    
                    # 更新DataFrame
                    result_df.at[result_df.index[idx], '诉讼类型'] = category
                    result_df.at[result_df.index[idx], '置信度'] = round(confidence, 2)
                    result_df.at[result_df.index[idx], '分类理由'] = reason
                    
                except Exception as e:
                    print(f"\n⚠️  第{idx+1}条分类失败: {str(e)[:50]}")
                    # 失败时标记为未知
                    result_df.at[result_df.index[idx], '诉讼类型'] = '未知'
                    result_df.at[result_df.index[idx], '置信度'] = 0.0
                    result_df.at[result_df.index[idx], '分类理由'] = f'分类失败: {str(e)[:30]}'
                
                pbar.update(1)
            
            # 批次间延迟，避免API限流
            if batch_end < actual_samples:
                time.sleep(delay_between_batches)
    
    print(f"\n分类完成！共分类 {actual_samples} 条数据")
    
    # 统计分类结果
    public_count = (result_df.iloc[:actual_samples]['诉讼类型'] == '公益').sum()
    private_count = (result_df.iloc[:actual_samples]['诉讼类型'] == '私益').sum()
    unknown_count = (result_df.iloc[:actual_samples]['诉讼类型'] == '未知').sum()
    
    print(f"  - 公益诉讼: {public_count} 条 ({public_count/actual_samples*100:.1f}%)")
    print(f"  - 私益诉讼: {private_count} 条 ({private_count/actual_samples*100:.1f}%)")
    print(f"  - 未知类型: {unknown_count} 条 ({unknown_count/actual_samples*100:.1f}%)")
    
    return result_df


def filter_public_interest_llm(df: pd.DataFrame, 
                               num_samples: int = 200) -> pd.DataFrame:
    """
    过滤出公益诉讼数据（仅针对前num_samples条）
    
    Args:
        df: 已分类的DataFrame（需包含'诉讼类型'列）
        num_samples: 样本数量
        
    Returns:
        只包含公益诉讼的DataFrame
    """
    if '诉讼类型' not in df.columns:
        raise ValueError("DataFrame中未找到'诉讼类型'列，请先进行分类")
    
    # 创建副本
    result_df = df.copy()
    
    # 标记前num_samples条中的私益数据
    for idx in range(min(num_samples, len(result_df))):
        if result_df.iloc[idx]['诉讼类型'] == '私益':
            result_df.at[result_df.index[idx], '_过滤标记'] = '已过滤-私益'
        elif result_df.iloc[idx]['诉讼类型'] == '公益':
            result_df.at[result_df.index[idx], '_过滤标记'] = '保留-公益'
        else:
            result_df.at[result_df.index[idx], '_过滤标记'] = '待确认'
    
    # num_samples之后的数据标记为未处理
    for idx in range(num_samples, len(result_df)):
        result_df.at[result_df.index[idx], '_过滤标记'] = '未处理'
    
    # 过滤掉私益数据
    filtered_df = result_df[result_df['_过滤标记'] != '已过滤-私益'].copy()
    
    # 删除临时标记列
    if '_过滤标记' in filtered_df.columns:
        filtered_df = filtered_df.drop(columns=['_过滤标记'])
    
    print(f"过滤完成！从 {len(df)} 条数据中保留了 {len(filtered_df)} 条公益诉讼数据")
    
    return filtered_df


if __name__ == "__main__":
    # 测试代码
    test_texts = [
        "小区附近工厂排放废气，严重影响居民健康",
        "我与物业公司签订的合同存在纠纷，要求退款",
        "道路坑洼不平，存在安全隐患",
        "楼上邻居装修噪音扰民",
        "公园设施破损，无人维护"
    ]
    
    print("=" * 80)
    print("大模型公益/私益分类测试")
    print("=" * 80)
    print()
    
    for text in test_texts:
        category, confidence, reason = classify_with_llm(text)
        print(f"文本: {text[:40]}...")
        print(f"分类: {category}")
        print(f"置信度: {confidence}")
        print(f"理由: {reason}")
        print("-" * 80)
        print()
