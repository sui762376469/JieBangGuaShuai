from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 加载本地模型（关键：把 model_id 换成你本地的文件夹路径）
ner_pipeline = pipeline(
    task=Tasks.named_entity_recognition,
    model="./model/raner_news",  # 这里填你下载的模型本地路径
)

# 测试识别
text = "大众点评商家：兄弟龙虾烧烤(凌江路店)；经营者：上海市浦东新区高桥镇超男餐饮店（个体工商户）；许可证编号：JY23101155540292；"
result = ner_pipeline(text)

# 输出结果
print(result)


# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from seqeval.metrics.sequence_labeling import get_entities


# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained("./model/bert4chinese")
# model = AutoModelForTokenClassification.from_pretrained("./model/bert4chinese")
# label_list = ['I-ORG', 'B-LOC', 'O', 'B-ORG', 'I-LOC', 'I-PER', 'B-TIME', 'I-TIME', 'B-PER']

# sentence = "依据国家鼓励生育、建设生育友好型社会政策及广东省“长幼随学”指导意见，英德市区城西片区楼盘密集、人口众多，公办幼儿园紧缺，多孩家庭接送不便、就近入园难。恳请尽快实施长幼随读，支持多孩子女同园同校、就近就便入园入学，方便家长接送；同时加快新增公办幼儿园，补齐学位缺口，琉璃河镇，落实普惠托育，切实减轻家庭育儿负担，请尽快落实。"


# def get_entity(sentence):
#     tokens = tokenizer.tokenize(sentence)
#     inputs = tokenizer.encode(sentence, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(inputs).logits
#     predictions = torch.argmax(outputs, dim=2)
#     char_tags = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())][1:-1]
#     print(sentence)
#     print(char_tags)

#     pred_labels = [i[1] for i in char_tags]
#     entities = []
#     line_entities = get_entities(pred_labels)
#     for i in line_entities:
#         word = sentence[i[1]: i[2] + 1]
#         entity_type = i[0]
#         entities.append((word, entity_type))

#     print("Sentence entity:")
#     print(entities)


# get_entity(sentence)
from transformers import pipeline
ner = pipeline("token-classification", model="./model/Ernie-ner", aggregation_strategy="simple")
print(ner("大众点评商家：兄弟龙虾烧烤(凌江路店)；18772387213，邮箱2108815080@qq.com,微信号wweqas123,qq号是213214123，联系方式：15232453213,经营者：上海市浦东新区高桥镇超男餐饮店（个体工商户）；许可证编号：JY23101155540292；"))