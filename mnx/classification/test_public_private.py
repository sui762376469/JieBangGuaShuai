"""
公益/私益分类功能测试脚本
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mnx.classification.public_private_classifier import classify_public_private

# 测试用例
test_cases = [
    # 公益诉讼示例
    ("小区附近工厂排放废气，严重影响居民健康", "公益"),
    ("道路坑洼不平，存在安全隐患，容易发生交通事故", "公益"),
    ("公园设施破损多年，无人维护，影响市民休闲", "公益"),
    ("学校周边交通拥堵，建议增设红绿灯", "公益"),
    ("河边垃圾堆积，臭气熏天，影响周边环境", "公益"),
    ("路灯损坏一个月，夜间出行不安全", "公益"),
    
    # 私益诉讼示例
    ("我与物业公司签订的合同存在纠纷，要求退还物业费", "私益"),
    ("楼上邻居装修噪音扰民，多次沟通无效", "私益"),
    ("购买的商品存在质量问题，商家拒绝退款", "私益"),
    ("与前配偶就子女抚养权产生争议", "私益"),
    ("老板拖欠工资三个月，要求支付", "私益"),
    ("房产过户手续办理遇到问题", "私益"),
]

print("=" * 80)
print("公益/私益诉讼分类功能测试")
print("=" * 80)
print()

correct = 0
total = len(test_cases)

for text, expected in test_cases:
    category, confidence = classify_public_private(text)
    is_correct = category == expected
    if is_correct:
        correct += 1
    
    status = "✅" if is_correct else "❌"
    print(f"{status} 预期: {expected:4s} | 实际: {category:4s} | 置信度: {confidence:.2f}")
    print(f"   文本: {text[:50]}...")
    print()

print("=" * 80)
print(f"测试结果: {correct}/{total} 正确 ({correct/total*100:.1f}%)")
print("=" * 80)
