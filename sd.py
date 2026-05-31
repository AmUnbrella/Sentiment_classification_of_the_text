import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import jieba
import os
import re
import pandas
import joblib

#获取数据及数据清洗
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_train_dir = os.path.join(script_dir,"train.csv") #这里是先路径，后文件名
csv_test_dir = os.path.join(script_dir,"test.csv")
df_train = pandas.read_csv(csv_train_dir,sep = '\t',header = None, names = ['text','label'])
df_test = pandas.read_csv(csv_test_dir,sep = '\t',header = None, names = ['text','label'])
#print(df.head())
#print(df['label'].value_counts())
def clean_text(text):
    text = re.sub(r'http\S+', '', text) #去URL
    text = re.sub(r'@\w+', '', text) #去@用户
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '',text) #仅保留中英文数字
    return text.strip()
df_train['clean_text'] = df_train['text'].astype(str).apply(clean_text)
df_test['clean_text'] = df_test['text'].astype(str).apply(clean_text)
#print(df.head())

#jieba 分词
def tokenize_and_join(text):
    return ' '.join(jieba.lcut(text))

df_train['token'] = df_train['clean_text'].apply(tokenize_and_join)
df_test['token'] = df_test['clean_text'].apply(tokenize_and_join)
#print(df['token'].head())
#print(df_test['token'].head())

y_train = df_train['label']
y_test = df_test['label']

#打包到pipeline
pipeline = Pipeline([
    ('tfidf',TfidfVectorizer(max_features=5000,min_df = 2,ngram_range=(1,2))),
    ('clf',linear_model.LogisticRegression(C = 10,solver = 'lbfgs',max_iter = 1000,random_state = 42))
])

'''
#设置参数测试范围
param_grid = {
    'tfidf__max_features':[3000, 5000],
    'tfidf__ngram_range' : [(1,1), (1,2)],
    'tfidf__min_df' : [1,2],
    'clf__C' : [0.1, 1, 10],
    'clf__solver' : ['lbfgs']
}

#网格搜索 + 5折交叉认证
grid_search = GridSearchCV(pipeline, param_grid, cv = 5, scoring = 'f1_macro', n_jobs = -1, verbose = 1)
grid_search.fit(df_train['token'],y_train)

print("The best varieties", grid_search.best_params_)
print("The best exchange examine F1(macro) : {:.4f}".format(grid_search.best_score_))


best_model = grid_search.best_estimator_
y_pred = best_model.predict(df_test['token'])
'''
print("The train begins")
X_train = df_train['token']
pipeline.fit(X_train, y_train)
print("The train ends")
X_test = df_test['token']
y_pred = pipeline.predict(X_test)

print("\nThe report:\n")
print(metrics.classification_report(y_test, y_pred))
print("\n The confusion matrix:\n")
print(metrics.confusion_matrix(y_test, y_pred))
joblib.dump(pipeline,'sentiment_pipline.pkl')
'''

#TF - IDF获取词频数据

vectorizer = TfidfVectorizer(max_features = 5000, ngram_range = (1,2))
X_train = vectorizer.fit_transform(df_train['token'])
X_test = vectorizer.transform(df_test['token'])

#逻辑回归进行模型训练
clf = linear_model.LogisticRegression(max_iter = 1000,random_state = 42)
clf.fit(X_train,df_train['label'])

#混淆矩阵评估
y_pred = clf.predict(X_test)
print("\nThe answer:\n")
print(metrics.classification_report(df_test['label'], y_pred))
print("\n The confusion matrix:\n")
print(metrics.confusion_matrix(df_test['label'], y_pred))
'''