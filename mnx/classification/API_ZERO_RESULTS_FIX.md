# API 返回 0 条结果问题修复说明

## 🐛 问题描述

运行混合推理时，日志显示：
```
[推理统计] 总行数=152, 本地模型可信=3行, 需转API处理=149行
[API 调用] 发现 149 条低置信度样本，开始调用 API 进行分类...
[API 完成] 成功使用 API 重新分类 0 条样本  ← 为什么是 0？
```

## 🔍 根本原因

**API 返回格式与解析正则不匹配！**

### 问题细节

1. **我们期望的格式**（正则表达式要求）：
   ```
   ID:类别编号:分组类型
   例如：1:1:c
   ```

2. **实际 System Prompt 要求的格式**：
   ```
   只输出类别编号
   例如：1
   ```

3. **结果**：
   - API 返回：`1` 或 `生态环境和资源保护`
   - 正则期望：`1:1:c`
   - **匹配失败** → 解析结果为 0 条

### 代码位置

**正则表达式**（第 463 行）：
```python
match = re.match(r'(\d+)[:：](\d+)[:：](.+)', line)
```

这个正则要求必须有**三个部分**，用冒号分隔：
- 第1组：ID（数字）
- 第2组：类别编号（数字）
- 第3组：分组类型（任意文本）

但原来的 `API_SYSTEM_PROMPT` 只要求返回类别编号，导致格式不匹配。

## ✅ 解决方案

### 修改 1：更新 API System Prompt

让 API 按照我们需要的格式返回结果：

```python
API_SYSTEM_PROMPT = """你是一个针对12345投诉文本进行智能分类的专家...

每一行投诉文本是一条数据，格式为："ID.文本内容"
请按照以下格式输出分类结果：
ID:类别编号:c

其中：
- ID 是输入中的编号
- 类别编号是上述10个类别对应的数字
- c 表示分类结果（固定值）

示例：
输入：
1.工厂排放废水污染了河流
2.商家销售过期食品

输出：
1:1:c
2:2:c

要求：每行一个结果，不要有其他解释文字
"""
```

### 修改 2：添加调试日志

在 `_call_api_for_classification` 函数中添加：

```python
# 打印 API 原始返回（前500字符）
print(f"\n--- 批次 {batch_idx + 1}/{total_batches} API 返回结果 ---")
print(reply_content[:500])

# 统计解析成功的数量
parsed_count = 0
for line in reply_content.split('\n'):
    ...
    if match:
        ...
        parsed_count += 1
    else:
        print(f"  ⚠️  无法解析的行: {line[:100]}")

print(f"  ✅ 本批次成功解析 {parsed_count}/{len(batch)} 条结果")
```

## 📊 修复后的效果

### 修复前
```
[API 调用] 发现 149 条低置信度样本，开始调用 API 进行分类...
[API 完成] 成功使用 API 重新分类 0 条样本  ← 全部失败
```

### 修复后（预期）
```
--- 批次 1/3 API 返回结果 ---
1:1:c
2:2:c
3:7:c
...
  ✅ 本批次成功解析 50/50 条结果

--- 批次 2/3 API 返回结果 ---
51:13:c
52:8:c
...
  ✅ 本批次成功解析 50/50 条结果

--- 批次 3/3 API 返回结果 ---
101:14:c
...
  ✅ 本批次成功解析 49/49 条结果

[API 完成] 成功使用 API 重新分类 149 条样本  ← 全部成功！
```

## 🔧 如果仍然返回 0 条

### 排查步骤

1. **查看 API 原始返回**：
   - 控制台会打印 API 返回的前 500 字符
   - 检查格式是否符合 `ID:类别编号:c`

2. **检查无法解析的行**：
   - 控制台会打印 `⚠️ 无法解析的行: ...`
   - 根据实际返回调整正则表达式

3. **常见错误格式**：
   ```
   ❌ "1"                          # 只有类别编号
   ❌ "生态环境和资源保护"          # 中文类别名
   ❌ "1: 生态环境"                # 缺少第三部分
   ❌ "类别1"                      # 格式完全不对
   
   ✅ "1:1:c"                      # 正确格式
   ```

4. **测试 API 调用**：
   ```python
   import os
   from openai import OpenAI
   
   client = OpenAI(
       base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
       api_key=os.environ['DASHSCOPE_API_KEY'],  # 从环境变量读取，勿将密钥写入代码
   )
   
   response = client.chat.completions.create(
       model='qwen-plus-2025-07-28',
       messages=[
           {'role': 'system', 'content': API_SYSTEM_PROMPT},
           {'role': 'user', 'content': '1.工厂排放废水'}
       ]
   )
   
   print(response.choices[0].message.content)
   # 应该输出：1:1:c
   ```

## 📝 相关文件

- **主程序**：`D:\Python\pythonProject\12345s\mnx\classification\model_nn_train_3layers_better.py`
- **参考实现**：`D:\Python\pythonProject\12345s\new_test\test2.py`
- **修复位置**：
  - 第 76-108 行：`API_SYSTEM_PROMPT`
  - 第 455-495 行：`_call_api_for_classification` 函数

## 💡 经验总结

1. **System Prompt 要明确指定输出格式**
   - 不仅要说"输出什么"，还要说"怎么输出"
   - 提供示例是最好的方式

2. **正则表达式要与实际返回严格匹配**
   - 开发时要先测试 API 返回格式
   - 添加调试日志帮助排查

3. **防御性编程**
   - 解析失败时要有 fallback 机制
   - 当前代码会将失败的批次标记为 `(0, "c")`

---

**修复日期**：2026-05-08  
**修复者**：AI Assistant
