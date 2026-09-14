"""
公益/私益诉讼分类模块
基于关键词规则进行初步分类
"""
import pandas as pd
import re
from typing import List, Tuple


# 公益诉讼特征关键词（涉及公共利益、公共安全、环境保护等）
PUBLIC_INTEREST_KEYWORDS = [
    # 环境污染类
    "油烟", "噪音", "污染", "异味", "臭味", "废气", "污水", "垃圾",
    "扬尘", "粉尘", "辐射", "焚烧", "排放", "环保",
    
    # 公共安全类
    "安全隐患", "消防", "危险", "坍塌", "坑洼", "破损",
    "路灯", "红绿灯", "交通", "道路", "桥梁", "隧道",
    
    # 公共设施类
    "公园", "广场", "公共厕所", "停车场", "公交站", "地铁",
    "供水", "供电", "供气", "供暖", "网络", "通信",
    
    # 教育管理类
    "学校", "幼儿园", "教育", "招生", "学位", "补课",
    "教师", "教学质量", "校园安全",
    
    # 医疗卫生类
    "医院", "卫生院", "医疗", "卫生", "防疫", "疫苗",
    "医保", "药品", "公共卫生",
    
    # 城市规划类
    "违建", "规划", "拆迁", "征地", "绿化", "市容",
    "城管", "占道经营", "乱摆卖",
    
    # 社会保障类
    "养老金", "社保", "低保", "救助", "福利", "残疾人",
    "老年人", "儿童", "妇女权益",
    
    # 其他公共利益
    "公共利益", "公共安全", "环境影响", "群体性",
    "多数人", "周边居民", "附近住户", "小区业主"
]

# 私益诉讼特征关键词（个人财产、合同纠纷、个人权益等）
PRIVATE_INTEREST_KEYWORDS = [
    # 合同纠纷类
    "合同", "协议", "违约", "退款", "退费", "赔偿",
    "欠款", "债务", "借贷", "担保",
    
    # 个人财产类
    "房产", "物业", "车位", "装修", "漏水", "裂缝",
    "产权", "房产证", "过户", "买卖",
    
    # 劳动纠纷类
    "工资", "加班", "辞退", "解雇", "工伤", "劳动合同",
    "社保缴纳", "公积金",
    
    # 消费纠纷类
    "商品质量", "售后服务", "虚假宣传", "欺诈",
    "退货", "换货", "消费者",
    
    # 邻里纠纷类
    "邻居", "楼上楼下", "相邻权", "采光", "通风",
    "私人", "个人", "我家", "我本人",
    
    # 婚姻家庭类
    "离婚", "抚养", "赡养", "继承", "婚姻",
    "家庭暴力", "财产分割"
]


def classify_public_private(text: str) -> Tuple[str, float]:
    """
    分类单个文本为公益或私益诉讼
    
    Args:
        text: 投诉文本
        
    Returns:
        Tuple[str, float]: (分类结果, 置信度)
        分类结果: "公益" 或 "私益"
        置信度: 0-1之间的浮点数
    """
    if not text or pd.isna(text):
        return "未知", 0.0
    
    text_lower = text.lower()
    
    # 计算公益关键词匹配数
    public_matches = sum(1 for keyword in PUBLIC_INTEREST_KEYWORDS 
                        if keyword in text_lower)
    
    # 计算私益关键词匹配数
    private_matches = sum(1 for keyword in PRIVATE_INTEREST_KEYWORDS 
                         if keyword in text_lower)
    
    # 计算总匹配数
    total_matches = public_matches + private_matches
    
    if total_matches == 0:
        # 如果没有匹配到任何关键词，默认为公益（因为12345多为公共事务）
        return "公益", 0.5
    
    # 计算置信度
    if public_matches > private_matches:
        confidence = public_matches / total_matches
        return "公益", min(confidence, 1.0)
    elif private_matches > public_matches:
        confidence = private_matches / total_matches
        return "私益", min(confidence, 1.0)
    else:
        # 相等时，根据文本长度和其他特征判断
        # 较长的文本通常更可能是公益（涉及面广）
        if len(text) > 200:
            return "公益", 0.6
        else:
            return "私益", 0.6


def api_classify_public_private(df: pd.DataFrame, 
                                 text_column: str = "content",
                                 num_samples: int = None) -> pd.DataFrame:
    """
    对DataFrame中的数据进行公益/私益分类
    
    Args:
        df: 输入DataFrame
        text_column: 文本列名
        num_samples: 需要分类的样本数量（默认None表示处理全部数据）
        
    Returns:
        添加了'诉讼类型'和'置信度'列的DataFrame
    """
    if text_column not in df.columns:
        raise ValueError(f"DataFrame中未找到'{text_column}'列")
    
    # 创建副本避免修改原数据
    result_df = df.copy()
    
    # 初始化新列
    result_df['诉讼类型'] = '待分类'
    result_df['置信度'] = 0.0
    
    # 确定处理的样本数量
    if num_samples is None:
        actual_samples = len(result_df)
    else:
        actual_samples = min(num_samples, len(result_df))
    
    print(f"正在对前 {actual_samples} 条数据进行公益/私益分类...")
    
    for idx in range(actual_samples):
        text = str(result_df.iloc[idx][text_column])
        category, confidence = classify_public_private(text)
        result_df.at[result_df.index[idx], '诉讼类型'] = category
        result_df.at[result_df.index[idx], '置信度'] = round(confidence, 2)
    
    print(f"分类完成！共分类 {actual_samples} 条数据")
    
    # 统计分类结果
    public_count = (result_df.iloc[:actual_samples]['诉讼类型'] == '公益').sum()
    private_count = (result_df.iloc[:actual_samples]['诉讼类型'] == '私益').sum()
    unknown_count = (result_df.iloc[:actual_samples]['诉讼类型'] == '未知').sum()
    
    print(f"  - 公益诉讼: {public_count} 条")
    print(f"  - 私益诉讼: {private_count} 条")
    print(f"  - 未知类型: {unknown_count} 条")
    
    return result_df


def filter_public_interest(df: pd.DataFrame, 
                           num_samples: int = None) -> pd.DataFrame:
    """
    过滤出公益诉讼数据
    
    Args:
        df: 已分类的DataFrame（需包含'诉讼类型'列）
        num_samples: 样本数量（默认None表示处理全部数据）
        
    Returns:
        只包含公益诉讼的DataFrame
    """
    if '诉讼类型' not in df.columns:
        raise ValueError("DataFrame中未找到'诉讼类型'列，请先进行分类")
    
    # 创建副本
    result_df = df.copy()
    
    # 确定处理的样本数量
    if num_samples is None:
        actual_samples = len(result_df)
    else:
        actual_samples = min(num_samples, len(result_df))
    
    # 标记前actual_samples条中的私益数据
    for idx in range(actual_samples):
        if result_df.iloc[idx]['诉讼类型'] == '私益':
            result_df.at[result_df.index[idx], '_过滤标记'] = '已过滤-私益'
        elif result_df.iloc[idx]['诉讼类型'] == '公益':
            result_df.at[result_df.index[idx], '_过滤标记'] = '保留-公益'
        else:
            result_df.at[result_df.index[idx], '_过滤标记'] = '待确认'
    
    # actual_samples之后的数据标记为未处理
    for idx in range(actual_samples, len(result_df)):
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
    
    for text in test_texts:
        category, confidence = classify_public_private(text)
        print(f"文本: {text[:30]}...")
        print(f"分类: {category}, 置信度: {confidence}\n")
