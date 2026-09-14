from openpyxl import load_workbook

def clean_xls_newlines(file_path):
    wb = load_workbook(file_path)
    for ws in wb.worksheets:  # 遍历所有工作表
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None and isinstance(cell.value, str):
                    # 去除 \n 和 \r
                    cell.value = cell.value.replace("\n", "").replace("\r", "")
    wb.save(file_path.replace(".xlsx", "_cleaned.xls"))  # 保存为新文件
    wb.close()
    print("清理完成！")

# 调用示例（替换为你的文件路径）
clean_xls_newlines("E:\jiebang\my_proj\mnx\dataset\测试.xlsx")