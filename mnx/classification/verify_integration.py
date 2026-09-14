"""
验证混合推理功能集成是否成功
"""
import sys
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

def check_imports():
    """检查导入是否成功"""
    print("=" * 60)
    print("步骤 1: 检查模块导入")
    print("=" * 60)
    
    try:
        from mnx.classification.model_nn_train_3layers_better import (
            api_predict_categories,
            _call_api_for_classification,
            API_SYSTEM_PROMPT,
            _api_client
        )
        print("✅ 成功导入 api_predict_categories")
        print("✅ 成功导入 _call_api_for_classification")
        print("✅ 成功导入 API_SYSTEM_PROMPT")
        print("✅ 成功导入 _api_client")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_api_config():
    """检查API配置"""
    print("\n" + "=" * 60)
    print("步骤 2: 检查 API 配置")
    print("=" * 60)
    
    from mnx.classification.model_nn_train_3layers_better import API_SYSTEM_PROMPT
    
    # 检查类别名称是否正确
    if "文物和文化遗产保护" in API_SYSTEM_PROMPT:
        print("✅ API 提示词中使用正确的类别名称: '文物和文化遗产保护'")
    else:
        print("❌ API 提示词中类别名称不正确")
        return False
    
    # 检查是否包含所有必需的类别
    required_categories = [
        "生态环境和资源保护",
        "食品药品安全",
        "国有财产保护",
        "国有土地使用权出让",
        "未成年人保护",
        "安全生产",
        "个人信息保护",
        "妇女权益保障",
        "无障碍环境建设",
        "文物和文化遗产保护"
    ]
    
    missing = []
    for cat in required_categories:
        if cat not in API_SYSTEM_PROMPT:
            missing.append(cat)
    
    if missing:
        print(f"❌ API 提示词中缺少以下类别: {missing}")
        return False
    else:
        print(f"✅ API 提示词包含所有 {len(required_categories)} 个必需类别")
    
    return True

def check_function_signature():
    """检查函数签名"""
    print("\n" + "=" * 60)
    print("步骤 3: 检查函数签名")
    print("=" * 60)
    
    from mnx.classification.model_nn_train_3layers_better import api_predict_categories
    import inspect
    
    sig = inspect.signature(api_predict_categories)
    params = list(sig.parameters.keys())
    
    expected_params = ['texts', 'batch_size', 'progress_cb']
    
    if params == expected_params:
        print(f"✅ 函数签名正确: {sig}")
        return True
    else:
        print(f"❌ 函数签名不匹配")
        print(f"   期望: {expected_params}")
        print(f"   实际: {params}")
        return False

def check_category_map_consistency():
    """检查类别映射一致性"""
    print("\n" + "=" * 60)
    print("步骤 4: 检查类别映射一致性")
    print("=" * 60)
    
    # 读取 category_map.json（如果存在）
    map_path = project_root / 'mnx' / 'classification' / 'category_map.json'
    
    if map_path.exists():
        import json
        with open(map_path, 'r', encoding='utf-8') as f:
            category_map = json.load(f)
        
        print(f"✅ 找到 category_map.json，包含 {len(category_map)} 个类别")
        
        # 检查类别 15 的名称
        if '15' in category_map:
            cat_15_name = category_map['15']
            print(f"   类别 15 的名称: '{cat_15_name}'")
            
            if cat_15_name == "文物和文化遗产保护":
                print("✅ 类别 15 名称正确")
            else:
                print(f"⚠️  类别 15 名称为 '{cat_15_name}'，可能需要在代码中调整")
    else:
        print("⚠️  未找到 category_map.json，将使用默认映射")
    
    return True

def check_file_structure():
    """检查文件结构"""
    print("\n" + "=" * 60)
    print("步骤 5: 检查文件结构")
    print("=" * 60)
    
    required_files = [
        current_dir / "model_nn_train_3layers_better.py",
        current_dir / "test_hybrid_inference.py",
        current_dir / "HYBRID_INFERENCE_README.md",
        current_dir / "USAGE_EXAMPLES.md",
        current_dir / "MODIFICATION_SUMMARY.md",
    ]
    
    all_exist = True
    for file_path in required_files:
        if file_path.exists():
            print(f"✅ {file_path.name}")
        else:
            print(f"❌ {file_path.name} 不存在")
            all_exist = False
    
    return all_exist

def main():
    """主验证流程"""
    print("\n" + "🔍" * 30)
    print("混合推理功能集成验证")
    print("🔍" * 30 + "\n")
    
    results = []
    
    # 执行各项检查
    results.append(("模块导入", check_imports()))
    results.append(("API 配置", check_api_config()))
    results.append(("函数签名", check_function_signature()))
    results.append(("类别映射", check_category_map_consistency()))
    results.append(("文件结构", check_file_structure()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{check_name:15s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有检查通过！混合推理功能已成功集成。")
        print("\n下一步:")
        print("  1. 运行测试脚本: python test_hybrid_inference.py")
        print("  2. 启动 Streamlit App: streamlit run ../../sy/app.py")
    else:
        print("⚠️  部分检查未通过，请查看上面的详细信息。")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
