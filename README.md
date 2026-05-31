# 对评价类文本的情感判断

基于逻辑回归对文本进行二分类，并利用paddlepaddle构建深度学习模型对比评估模型性能

项目文件夹主要包括以下文件：
ReadMe.txt
sd.py
aply.py
paddle_model.py
test.csv
train.csv
paddle_predictions.csv
文本分类的数学思想.md
分析报告.md
对比分析报告.md

**声明：paddle_model.py 和 对比分析报告.md 均由Claude Code（API：Deepseek v4flash）生成**

模型训练数据来自 GitCode https://gitcode.com/open-source-toolkit/78ecd/blob/main/chnsenticorp.zip
包括train.csv和test.csv

sd.py 是模型训练程序，包括：
1.获取数据及数据清洗
2.jieba 分词
3.TF-IDF获取词频数据
4.逻辑回归
5.混淆矩阵
6.网格化搜索

aply.py 运行后可以直接输入一句话，输出情感评估和置信度

paddle_model.py 是Claude Code本地部署PaddlePaddle生成的深度学习模型

sd模型训练数据：
The report:

              precision    recall  f1-score   support

           0       0.88      0.90      0.89       592
           1       0.90      0.88      0.89       608

    accuracy                           0.89      1200
   macro avg       0.89      0.89      0.89      1200
weighted avg       0.89      0.89      0.89      1200


 The confusion matrix:

[[535  57]
 [ 73 535]]

**注意：本包不包括 pip 的安装，如果想要在本地运行要下载 numpy库 sklearn库 jieba库 pandas库 joblib库
如果想测试paddle_mode.py请根据py文件内容自行下载pip，因为是ClaudeCode给我做的所以我也不知道要下载什么库**
