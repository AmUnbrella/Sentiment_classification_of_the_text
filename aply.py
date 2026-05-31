import jieba
import os
import joblib
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_dir = os.path.join(script_dir,"sentiment_pipline.pkl") #这里是先路径，后文件名
pipeline = joblib.load(pkl_dir)

text = input("请输入文本：")

text = re.sub(r'http\S+', '', text) #去URL
text = re.sub(r'@\w+', '', text) #去@用户
text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '',text) #仅保留中英文数字
words = jieba.lcut(text)
text = ' '.join(words)

proba = pipeline.predict_proba([text])[0]
confidence = max(proba)
print("置信度：{}".format(confidence))

pred = pipeline.predict([text])[0]
if pred == 0:
    opt = "负面评价"
else:
    opt = "正面评价"
print("结果：{}".format(opt))