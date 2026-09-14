from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

task = Tasks.sentence_similarity
model = './model/mego_similar'
inputs = inputs = ('西安市鄠邑区余下镇仪表厂十字路口厂蒲洼乡', '西安市鄠邑区余下镇仪表厂十字路口河北镇')
# v1.2.0和v1.1.2均可尝试使用
pipeline_ins = pipeline(
    task=task, model=model, model_revision='master')
print(pipeline_ins(input=inputs))
# 输出
# {'scores': [0.06451419740915298, 0.9217355251312256, 0.013750356622040272], 'labels': ['partial_match', 'exact_match', 'not_match']}