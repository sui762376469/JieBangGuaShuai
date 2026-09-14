import pandas as pd
import os

# 设置文件路径（相对于当前脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..', 'mnx', 'classification')
REAL_FILE = os.path.join(BASE_DIR, 'test_set_with_true_labels.xlsx')
PRED_FILE = os.path.join(BASE_DIR, 'test_set_with_pred_labels.xlsx')

# 列名配置
DESCRIPTION_COL = '诉求描述'
REAL_LABEL_COL = '真实标签'
PRED_LABEL_COL = '预测标签'
THRESHOLD = 0.85  # 85% 阈值


def calculate_prediction_accuracy(real_file, pred_file, description_col=DESCRIPTION_COL, 
                                   real_label_col=REAL_LABEL_COL, pred_label_col=PRED_LABEL_COL,
                                   threshold=THRESHOLD):
    """
    计算预测准确率并判断错误率是否超过阈值
    
    参数:
        real_file: 包含真实标签的 Excel 文件路径
        pred_file: 包含预测标签的 Excel 文件路径
        description_col: 诉求描述列名
        real_label_col: 真实标签列名
        pred_label_col: 预测标签列名
        threshold: 准确率阈值要求 (默认 85%)
    
    返回:
        overall_accuracy: 总体准确率
        class_accuracies: 各类别准确率字典
        is_acceptable: 是否满足要求 (所有准确率和平均准确率 >= 85%)
    """
    
    # 读取两个 Excel 文件
    print(f"正在读取真实标签文件：{real_file}")
    df_real = pd.read_excel(real_file)
    
    print(f"正在读取预测标签文件：{pred_file}")
    df_pred = pd.read_excel(pred_file)
    
    # 检查必需的列是否存在
    for df, name in [(df_real, '真实标签文件'), (df_pred, '预测标签文件')]:
        if description_col not in df.columns:
            raise ValueError(f"{name}中缺少列：{description_col}")
    
    if real_label_col not in df_real.columns:
        raise ValueError(f"真实标签文件中缺少列：{real_label_col}")
    
    if pred_label_col not in df_pred.columns:
        raise ValueError(f"预测标签文件中缺少列：{pred_label_col}")
    
    # 根据诉求描述进行匹配
    print("正在根据诉求描述匹配数据...")
    merged_df = pd.merge(df_real, df_pred, on=description_col, how='inner')
    
    if len(merged_df) == 0:
        raise ValueError("无法匹配到任何数据，请检查诉求描述列的内容")
    
    print(f"成功匹配到 {len(merged_df)} 条记录")
    
    # 计算预测正确的数量
    merged_df['is_correct'] = merged_df[real_label_col] == merged_df[pred_label_col]
    
    # 计算总体准确率
    correct_count = merged_df['is_correct'].sum()
    total_count = len(merged_df)
    overall_accuracy = correct_count / total_count
    
    # 计算每一类的准确率
    print("\n正在计算每一类的准确率...")
    class_accuracies = {}
    
    # 使用真实标签列来分组计算每一类的准确率
    for label in merged_df[real_label_col].unique():
        class_data = merged_df[merged_df[real_label_col] == label]
        class_correct = class_data['is_correct'].sum()
        class_total = len(class_data)
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        class_accuracies[label] = {
            'accuracy': class_accuracy,
            'correct': class_correct,
            'total': class_total
        }
    
    # 计算平均准确率（所有类别的准确率的平均值）
    avg_accuracy = sum(acc['accuracy'] for acc in class_accuracies.values()) / len(class_accuracies)
    
    # 判断是否满足要求
    # 1. 总体准确率 >= 85%
    # 2. 平均准确率 >= 85%
    # 3. 每一类的准确率都 >= 85%
    overall_pass = overall_accuracy >= threshold
    avg_pass = avg_accuracy >= threshold
    all_classes_pass = all(acc['accuracy'] >= threshold for acc in class_accuracies.values())
    
    is_acceptable = overall_pass and avg_pass and all_classes_pass
    
    # 输出结果
    print("\n" + "="*60)
    print("预测准确率分析报告")
    print("="*60)
    print(f"总样本数：{total_count}")
    print(f"预测正确数：{correct_count}")
    print(f"预测错误数：{total_count - correct_count}")
    print(f"总体准确率：{overall_accuracy:.2%}")
    print(f"平均准确率：{avg_accuracy:.2%}")
    print(f"阈值要求：准确率 >= {threshold:.0%}")
    print("-"*60)
    
    print("\n各类别准确率详情:")
    print("-"*60)
    print(f"{'类别':<20} {'样本数':>8} {'正确数':>8} {'准确率':>10} {'状态':>8}")
    print("-"*60)
    
    for label, stats in sorted(class_accuracies.items()):
        status = "✓" if stats['accuracy'] >= threshold else "✗"
        print(f"{str(label):<20} {stats['total']:>8} {stats['correct']:>8} {stats['accuracy']:>10.2%} {status:>8}")
    
    print("-"*60)
    
    # 总结
    print("\n" + "="*60)
    print("评估结果:")
    print("="*60)
    print(f"总体准确率：{'✓ 达标' if overall_pass else '✗ 未达标'} ({overall_accuracy:.2%} >= {threshold:.0%})")
    print(f"平均准确率：{'✓ 达标' if avg_pass else '✗ 未达标'} ({avg_accuracy:.2%} >= {threshold:.0%})")
    print(f"所有类别：{'✓ 全部达标' if all_classes_pass else '✗ 存在未达标类别'}")
    print("="*60)
    
    if is_acceptable:
        print("\n✓✓✓ 结果：所有指标均达到要求！")
    else:
        print("\n✗✗✗ 结果：存在未达标项，需要优化模型")
        if not overall_pass:
            print("  - 总体准确率未达标")
        if not avg_pass:
            print("  - 平均准确率未达标")
        if not all_classes_pass:
            print("  - 部分类别准确率未达标")
    
    print("="*60)
    
    return overall_accuracy, class_accuracies, is_acceptable, avg_accuracy


def main():
    """主函数"""
    print("="*60)
    print("预测准确率计算工具")
    print("="*60)
    print(f"\n真实标签文件：{REAL_FILE}")
    print(f"预测标签文件：{PRED_FILE}")
    print(f"阈值要求：{THRESHOLD:.0%}")
    print("-"*60)
    
    try:
        overall_accuracy, class_accuracies, is_acceptable, avg_accuracy = calculate_prediction_accuracy(
            REAL_FILE, 
            PRED_FILE,
            DESCRIPTION_COL,
            REAL_LABEL_COL,
            PRED_LABEL_COL,
            THRESHOLD
        )
        
        print(f"\n分析完成！")
        
    except Exception as e:
        print(f"\n错误：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
