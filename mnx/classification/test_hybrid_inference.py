"""
测试混合推理功能（本地模型 + API）
"""
import sys
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from mnx.classification.model_nn_train_3layers_better import api_predict_categories

def test_hybrid_inference():
    """测试混合推理功能"""
    
    # 测试文本
    test_texts = [
        "工厂排放废水污染了河流，影响周边居民生活",
        "商家销售过期食品，消费者食用后身体不适",
        "未成年人遭受校园欺凌，需要法律保护",
        "某公司拖欠工资，工人无法拿到应得报酬",
        "个人信息被泄露，收到大量骚扰电话"
    ]
    
    print("=" * 60)
    print("测试混合推理功能（本地模型 + API）")
    print("=" * 60)
    print(f"\n测试样本数: {len(test_texts)}\n")
    
    # 定义进度回调
    def progress_cb(percent, msg):
        print(f"[{percent:3d}%] {msg}")
    
    try:
        # 调用预测函数
        results = api_predict_categories(test_texts, batch_size=16, progress_cb=progress_cb)
        
        print("\n" + "=" * 60)
        print("预测结果:")
        print("=" * 60)
        for i, (text, pred) in enumerate(zip(test_texts, results), 1):
            print(f"\n{i}. 文本: {text[:50]}...")
            print(f"   预测类别: {pred}")
        
        print("\n" + "=" * 60)
        print("✅ 测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hybrid_inference()
