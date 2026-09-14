import os

def process_txt_file(input_file, output_file):
    """
    处理txt文件，去除每行的标签和制表符，只保留content部分
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            # 读取所有行
            lines = infile.readlines()
            
            if not lines:
                print("输入文件为空")
                return
            
            # 处理第一行（标题行）
            first_line = lines[0].strip()
            if '\t' in first_line:
                # 如果第一行包含制表符，跳过它（不写入输出文件）
                print(f"跳过标题行: {first_line}")
            else:
                # 如果第一行没有制表符，可能是数据行，按正常数据处理
                parts = first_line.split('\t', 1)  # 最多分割一次
                if len(parts) > 1:
                    content = parts[1]  # 取第二部分作为content
                else:
                    content = parts[0]  # 如果没有制表符，整行作为content
                outfile.write(content + '\n')
            
            # 处理剩余的行
            for i, line in enumerate(lines[1:], start=2):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                if '\t' in line:
                    parts = line.split('\t', 1)  # 最多分割一次
                    if len(parts) > 1:
                        content = parts[1]  # 取第二部分作为content
                    else:
                        content = parts[0]  # 如果没有制表符，整行作为content
                else:
                    content = line  # 如果没有制表符，整行作为content
                
                outfile.write(content + '\n')
        
        print(f"处理完成！结果已保存到: {output_file}")
        print(f"共处理了 {len(lines)-1} 行数据")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_file = "../dataset/test_dataset.txt"  # 根据你的实际文件名修改
    output_file = "../dataset/processed_dataset.txt"  # 输出文件名
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 执行处理
    process_txt_file(input_file, output_file)