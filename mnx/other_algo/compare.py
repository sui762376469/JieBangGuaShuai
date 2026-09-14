# 功能: 比较 classified_dataset.txt 和 processed_dataset.txt 的差异
# 输出: 差异行的位置、差异内容、差异比例

from cgi import test


def compare_files(classified_file, processed_file):
    """
    比较两个文件的差异
    
    Args:
        classified_file: 分类后的文件 (label\tcontent 格式)
        processed_file: 原始处理后的文件 (content 格式)
    """
    print("=" * 80)
    print("开始比较两个文件的差异...")
    print("=" * 80)
    
    # 读取 classified_dataset.txt，提取 content 部分
    classified_contents = []
    with open(classified_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 跳过标题行
        for i, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if '\t' in line:
                parts = line.split('\t', 1)
                content = parts[1] if len(parts) > 1 else parts[0]
                classified_contents.append(content)
            elif line:
                classified_contents.append(line)
    
    # 读取 processed_dataset.txt
    processed_contents = []
    with open(processed_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 跳过标题行
        for i, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if line:
                processed_contents.append(line)
    
    print(f"\n文件统计:")
    print(f"  classified_dataset.txt: {len(classified_contents)} 行内容")
    print(f"  test_dataset.txt: {len(processed_contents)} 行内容")
    
    # 确定比较的行数（取较小的那个）
    min_lines = min(len(classified_contents), len(processed_contents))
    max_lines = max(len(classified_contents), len(processed_contents))
    
    if len(classified_contents) != len(processed_contents):
        print(f"\n⚠️  警告: 两个文件的行数不一致!")
        print(f"     差异: {abs(len(classified_contents) - len(processed_contents))} 行")
    
    print(f"\n开始逐行比较前 {min_lines} 行...")
    
    # 找出差异
    differences = []
    for i in range(min_lines):
        if classified_contents[i] != processed_contents[i]:
            differences.append({
                'line_number': i + 2,  # +2 因为跳过了标题行，且行号从1开始
                'index': i,
                'classified': classified_contents[i],
                'processed': processed_contents[i]
            })
    
    # 输出结果
    print("\n" + "=" * 80)
    print("比较结果")
    print("=" * 80)
    
    total_compared = min_lines
    diff_count = len(differences)
    same_count = total_compared - diff_count
    diff_ratio = diff_count / total_compared * 100 if total_compared > 0 else 0
    
    print(f"\n📊 统计信息:")
    print(f"  总比较行数: {total_compared}")
    print(f"  相同行数: {same_count}")
    print(f"  差异行数: {diff_count}")
    print(f"  差异比例: {diff_ratio:.2f}%")
    
    if diff_count > 0:
        print(f"\n❌ 发现 {diff_count} 处差异:")
        print("-" * 80)
        
        # 显示所有差异（如果差异太多，只显示前20个）
        display_count = min(20, diff_count)
        for idx, diff in enumerate(differences[:display_count]):
            print(f"\n差异 #{idx + 1} - 第 {diff['line_number']} 行:")
            print(f"  classified_dataset: {diff['classified'][:60]}{'...' if len(diff['classified']) > 60 else ''}")
            print(f"  processed_dataset:  {diff['processed'][:60]}{'...' if len(diff['processed']) > 60 else ''}")
            
            # 高亮显示具体差异位置
            classified_text = diff['classified']
            processed_text = diff['processed']
            
            # 找到第一个不同的字符位置
            min_len = min(len(classified_text), len(processed_text))
            first_diff_pos = -1
            for j in range(min_len):
                if classified_text[j] != processed_text[j]:
                    first_diff_pos = j
                    break
            
            if first_diff_pos == -1 and len(classified_text) != len(processed_text):
                first_diff_pos = min_len
            
            if first_diff_pos >= 0:
                print(f"  ⚠️  第一个差异位置: 第 {first_diff_pos + 1} 个字符")
                print(f"      classified: ...{classified_text[max(0, first_diff_pos-10):first_diff_pos+10]}...")
                print(f"      processed:  ...{processed_text[max(0, first_diff_pos-10):first_diff_pos+10]}...")
        
        if diff_count > 20:
            print(f"\n  ... 还有 {diff_count - 20} 处差异未显示")
        
        # 保存差异详情到文件
        diff_output_file = "dataset/differences_detail.txt"
        with open(diff_output_file, 'w', encoding='utf-8') as f:
            f.write(f"文件比较差异报告\n")
            f.write(f"=" * 80 + "\n")
            f.write(f"classified_dataset.txt: {classified_file}\n")
            f.write(f"processed_dataset.txt: {processed_file}\n")
            f.write(f"=" * 80 + "\n\n")
            f.write(f"总比较行数: {total_compared}\n")
            f.write(f"差异行数: {diff_count}\n")
            f.write(f"差异比例: {diff_ratio:.2f}%\n\n")
            f.write(f"详细差异列表:\n")
            f.write(f"-" * 80 + "\n")
            
            for idx, diff in enumerate(differences):
                f.write(f"\n差异 #{idx + 1} - 第 {diff['line_number']} 行:\n")
                f.write(f"  classified_dataset: {diff['classified']}\n")
                f.write(f"  processed_dataset:  {diff['processed']}\n")
        
        print(f"\n💾 差异详情已保存到: {diff_output_file}")
    
    else:
        print(f"\n✅ 两个文件的 content 部分完全一致！")
    
    # 检查是否有额外的行
    if len(classified_contents) > len(processed_contents):
        extra_lines = len(classified_contents) - len(processed_contents)
        print(f"\n⚠️  classified_dataset.txt 多出 {extra_lines} 行")
    elif len(processed_contents) > len(classified_contents):
        extra_lines = len(processed_contents) - len(classified_contents)
        print(f"\n⚠️  processed_dataset.txt 多出 {extra_lines} 行")
    
    print("\n" + "=" * 80)
    print("比较完成！")
    print("=" * 80)
    
    return {
        'total_lines': total_compared,
        'diff_count': diff_count,
        'same_count': same_count,
        'diff_ratio': diff_ratio,
        'differences': differences
    }


if __name__ == "__main__":
    classified_file = "../dataset/classified_dataset.txt"
    test_file = "../dataset/test_dataset.txt"
    
    result = compare_files(classified_file, test_file)