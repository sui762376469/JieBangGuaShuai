# 12345 涉检线索智能筛查系统

面向 12345 政务投诉数据的线索筛查与辅助研判平台，提供数据导入、重复投诉检测、公益/私益筛选、敏感信息脱敏、智能分类、拖欠工资要素提取、多级预警、台账管理和地图可视化等功能。
<img width="1611" height="929" alt="屏幕截图 2026-09-14 170745" src="https://github.com/user-attachments/assets/c3208af0-c722-49c4-b55b-1f9906f4ab7d" />

## 技术栈

- Python、Streamlit、Pandas、SQLite
- jieba、BM25、Sentence-Transformers/BGE-M3
- PyTorch 三层 MLP、Focal Loss、GPU/CPU 自动切换
- Transformers NER 脱敏、OpenAI 兼容大模型接口

## 环境安装

```powershell
conda create -n 12345s python=3.11 -y
conda activate 12345s
cd D:\Python\pythonProject\12345s
python -m pip install -r requirements.txt
```

`requirements.txt` 中只保留第三方依赖；`sqlite3`、`pathlib`、`re` 等属于 Python 标准库，不需要安装。

## 模型与数据

模型权重、真实投诉数据、数据库、生成结果和 API Key 不提交到 GitHub。请根据实际部署环境，将模型放入以下目录，或在代码中配置新的模型路径：

```text
mnx/core_model/bge-m3/
mnx/extract/model/Ernie-ner/
```

训练完成后，分类权重和编码器会生成在 `mnx/classification/`。大模型二次分类使用环境变量 `DASHSCOPE_API_KEY`；可选变量包括 `CLASSIFICATION_API_BASE_URL` 和 `CLASSIFICATION_API_MODEL`。参考 `.env.example` 配置，严禁把真实密钥写入代码。

## 启动

```powershell
cd D:\Python\pythonProject\12345s\sy
conda activate 12345s
python -m streamlit run main.py
```

浏览器打开终端显示的地址，默认通常为 `http://localhost:8501`。测试账号由 `sy/users.json` 管理，生产环境必须修改默认账号并使用安全的密码哈希方案。

## 使用流程

1. 登录系统。
2. 进入“数据导入与预测”，上传 CSV 或 Excel。
3. 选择投诉文本列，执行去重和公益/私益筛选。
4. 执行专业脱敏。
5. 在“模型重训”中准备训练集和类别映射表，训练或部署分类模型。
6. 执行分类预测并写入台账。
7. 在“台账查看”中进行人工复核、查询和导出。

## 安全说明

本项目仅用于内部演示和辅助研判。上传 GitHub 前请确认代码中没有真实 API Key，`.gitignore` 已排除模型、数据库和数据文件，历史提交中也没有密钥。此前已经暴露的密钥必须先在服务商平台撤销并重新生成。
