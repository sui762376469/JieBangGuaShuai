# 预测类别映射统一化管理说明

## 📋 问题背景

在系统中发现预测类别为空的问题，根本原因是：
1. **类别名称不统一**：不同模块使用了不同的类别映射定义
2. **API托底推理映射不完整**：硬编码的类别映射与 `category_map.json` 不一致
3. **缺少验证机制**：无法及时发现和警告类别映射失败的情况

## ✅ 解决方案

### 1. 创建统一的类别映射管理器

**文件位置**: `D:\Python\pythonProject\12345s\mnx\classification\category_mapper.py`

**核心功能**:
- 单例模式，确保全局使用同一份类别映射
- 从 `category_map.json` 加载类别定义（训练时上传的标签表格）
- 提供正向映射（ID → Name）和反向映射（Name → ID）
- 自动验证类别名称的有效性
- 提供详细的调试日志

**主要接口**:
```python
from mnx.classification.category_mapper import (
    get_category_name,      # ID -> 名称
    get_category_id,        # 名称 -> ID
    get_all_categories,     # 获取所有映射
    get_api_category_map,   # 获取API使用的映射
    validate_and_fix_category_name  # 验证并修正类别名称
)
```

### 2. 修改 API 托底推理部分

**文件**: `D:\Python\pythonProject\12345s\mnx\classification\model_nn_train_3layers_better.py`

**修改内容** (第633-670行):
```python
# 修改前：硬编码的不完整映射
category_map = {
    1: "生态环境和资源保护",
    2: "食品药品安全",
    # ... 只有部分类别
}
category_name = category_map.get(category, f"类别{category}")  # 失败时生成无效名称

# 修改后：使用统一映射管理器
from mnx.classification.category_mapper import get_api_category_map
category_map = get_api_category_map()  # 完整的类别映射

category_name = category_map.get(category, None)
if category_name is None:
    # 记录警告并使用默认类别
    print(f"⚠️ API 返回的类别编号 {category} 不在 category_map.json 中")
    category_name = category_map.get(0, "非公益诉讼")
```

**优势**:
- ✅ 与训练时的类别定义完全一致
- ✅ 包含所有17个类别（0-16）
- ✅ 自动检测并警告未映射的类别编号
- ✅ 失败时使用合理的默认值

### 3. 修改前端预测结果处理

**文件**: `D:\Python\pythonProject\12345s\sy\app.py`

**修改内容** (第681-709行):
```python
# 修改前：直接使用 REVERSE_CATEGORY_MAP
cat_id = REVERSE_CATEGORY_MAP.get(label_str, 0)

# 修改后：使用统一的验证函数
from mnx.classification.category_mapper import validate_and_fix_category_name
cat_id, validated_name = validate_and_fix_category_name(label_str, default_id=0)

if cat_id == 0 and label_str != "非公益诉讼":
    unmapped_labels.append(label_str)  # 记录映射失败的标签
```

**新增功能**:
- 📊 显示未映射标签的详细列表
- 📊 显示预测类别分布统计
- ⚠️ 提供明确的错误提示和修复建议

## 🔧 使用指南

### 场景1：模型训练后

训练完成后，`category_map.json` 会自动更新。无需额外操作，`CategoryMapper` 会在下次使用时自动重新加载。

### 场景2：手动更新类别映射

如果需要手动修改类别定义：

1. 编辑 `D:\Python\pythonProject\12345s\mnx\classification\category_map.json`
2. 确保格式正确：
```json
{
  "0": "非公益诉讼",
  "1": "生态环境和资源保护",
  ...
}
```
3. 重启应用或调用 `CategoryMapper().reload()`

### 场景3：调试类别映射问题

运行测试脚本：
```bash
cd D:\Python\pythonProject\12345s\mnx\classification
python category_mapper.py
```

输出示例：
```
============================================================
类别映射管理器测试
============================================================

所有类别:
  0: 非公益诉讼
  1: 生态环境和资源保护
  2: 食品药品安全
  ...

测试正向映射 (ID -> Name):
  0 -> 非公益诉讼
  1 -> 生态环境和资源保护
  99 -> 未知类别(99)

测试反向映射 (Name -> ID):
  '生态环境和资源保护' -> 1
  '不存在的类别' -> None
```

## 📊 当前类别定义

根据 `category_map.json`，系统支持以下17个类别：

| ID | 类别名称 |
|----|---------|
| 0 | 非公益诉讼 |
| 1 | 生态环境和资源保护 |
| 2 | 食品药品安全 |
| 3 | 国有财产保护 |
| 4 | 国有土地使用权出让 |
| 5 | 英雄烈士保护 |
| 6 | 未成年人保护 |
| 7 | 安全生产 |
| 8 | 个人信息保护 |
| 9 | 军人地位和权益保障 |
| 10 | 反垄断和反不正当竞争 |
| 11 | 反电信网络诈骗 |
| 12 | 农产品质量安全 |
| 13 | 妇女权益保障 |
| 14 | 无障碍环境建设 |
| 15 | 文物和文化遗产保护 |
| 16 | 国防和军事利益保护 |

## ⚠️ 注意事项

1. **严格遵循训练时的类别定义**
   - 所有涉及预测类别的地方都必须使用 `category_mapper` 模块
   - 不要硬编码类别名称或映射关系

2. **API 返回的类别编号必须在映射表中定义**
   - 如果 API 返回了新的类别编号，需要先更新 `category_map.json`
   - 系统会自动检测并警告未映射的类别

3. **类别名称必须完全匹配**
   - "文物和文化遗产保护" ≠ "文物和文化遗产"
   - 注意空格、标点符号的差异

4. **训练后自动更新**
   - 每次重新训练模型时，`category_map.json` 会被覆盖
   - 确保训练数据中的标签与预期一致

## 🔍 故障排查

### 问题1：预测类别显示为"非公益诉讼"（ID=0）

**可能原因**:
- 模型预测的类别名称不在 `category_map.json` 中
- API 返回的类别编号未定义

**解决方法**:
1. 查看控制台输出的"未识别的标签"警告
2. 检查 `category_map.json` 是否包含该类别
3. 如果缺少，添加对应的类别定义

### 问题2：台账查看界面类别为空

**可能原因**:
- 数据库中存储的 `pred_category` 为 0 或 NULL
- `CATEGORY_MAP` 加载失败

**解决方法**:
1. 检查数据库中的实际值：
```sql
SELECT DISTINCT pred_category FROM ledger;
```
2. 确认 `category_map.json` 文件存在且格式正确
3. 查看应用启动日志是否有加载错误

### 问题3：API 托底推理返回未知类别

**可能原因**:
- API 返回的类别编号超出范围
- `category_map.json` 定义不完整

**解决方法**:
1. 查看控制台输出的"[API 警告]"信息
2. 确认 API 使用的类别编号体系
3. 更新 `category_map.json` 以包含所有可能的类别

## 📝 修改历史

- **2026-05-14**: 创建统一的类别映射管理器
- **2026-05-14**: 修复 API 托底推理的类别映射问题
- **2026-05-14**: 增强前端预测结果的验证和调试功能

## 🎯 最佳实践

1. **始终使用统一接口**
   ```python
   # ✅ 正确
   from mnx.classification.category_mapper import get_category_id
   cat_id = get_category_id("生态环境和资源保护")
   
   # ❌ 错误
   cat_id = REVERSE_CATEGORY_MAP.get("生态环境和资源保护", 0)
   ```

2. **定期验证映射完整性**
   ```python
   from mnx.classification.category_mapper import CategoryMapper
   mapper = CategoryMapper()
   
   # 检查所有类别是否有效
   for cat_id, cat_name in mapper.get_all_categories().items():
       assert mapper.validate_category_name(cat_name)
   ```

3. **保留调试日志**
   - 生产环境中保留类别映射的警告日志
   - 定期检查是否有新的未映射类别出现

---

**维护者**: AI Assistant  
**最后更新**: 2026-05-14
