import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import tree
import pydotplus
from io import StringIO
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import export_graphviz
from IPython.display import Image

 
 
#导入数据
data = pd.read_csv('titanic_train.csv')
 

#数据清洗
data["Age"] = data["Age"].fillna(data["Age"].median())
data.loc[data["Sex"] == "male", "Sex"] = 0
data.loc[data["Sex"] == "female", "Sex"] = 1
data["Embarked"] = data["Embarked"].fillna("S")
data.loc[data["Embarked"] == "S", "Embarked"] = 0
data.loc[data["Embarked"] == "C", "Embarked"] = 1
data.loc[data["Embarked"] == "Q", "Embarked"] = 2
data["FamilySize"] = data["SibSp"] + data["Parch"]
data["NameLength"] = data["Name"].apply(lambda x: len(x))


# 提取名字信息
import re
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
titles = data["Name"].apply(get_title)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Mlle": 7, "Major": 8, "Col": 9,
                 "Ms": 10, "Mme": 11, "Lady": 12, "Sir": 13, "Capt": 14, "Don": 15, "Jonkheer": 16, "Countess": 17}
for k, v in title_mapping.items():
    titles[titles == k] = v
data["Title"] = titles
data= data.drop('Name', 1)
data= data.drop('Ticket', 1)
data= data.drop('Cabin', 1)
data= data.drop('PassengerId', 1)
print(data)








'''

#网格搜索

X = data.drop(["Survived"], axis=1)
y = data["Survived"]
 
#训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)
 
 
#构建网格参数
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 50],
    'max_features': [len((X.columns))],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [4, 8],
    'n_estimators': [5, 10, 50]
}
 
#初始化模型
forest = RandomForestClassifier()
#初始化网格搜索
grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, cv=10,
                           n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
 
#查看最好的参数选择
print(grid_search.best_params_)
 
#使用网格搜索得到的最好的参数选择进行模型训练
best_forest = grid_search.best_estimator_
best_forest.fit(X_train, y_train)
 
# 预测
pred_train = best_forest.predict(X_train)
pred_test = best_forest.predict(X_test)
 
#准确率
train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print ("训练集准确率: {0:.4f}, 测试集准确率: {1:.4f}".format(train_acc, test_acc))

#其他模型评估指标
precision, recall, F1, _ = precision_recall_fscore_support(y_test, pred_test, average="binary")
print ("precision: {0:.4f}. recall: {1:.4f}, F1: {2:.4f}".format(precision, recall, F1))
 
#特征重要度
features = list(X_test.columns)
importances = best_forest.feature_importances_
indices = np.argsort(importances)[::-1]
num_features = len(importances)
'''

'''
#将特征重要度以柱状图展示
plt.figure()
plt.title("Feature importances")
plt.bar(range(num_features), importances[indices], color="g", align="center")
plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
plt.xlim([-1, num_features])
plt.show()
'''
'''
#输出各个特征的重要度
for i in indices:
    print ("{0} - {1:.3f}".format(features[i], importances[i]))

'''




'''

训练集准确率: 0.8937, 测试集准确率: 0.8233
训练效果：precision: 0.7720. recall: 0.7321, F1: 0.7519
Title - 0.418
Fare - 0.124
NameLength - 0.099
Age - 0.098
FamilySize - 0.087
Pclass - 0.085
Sex - 0.044
Embarked - 0.029
SibSp - 0.008
Parch - 0.007


'''







#生成决策树
'''
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","Title","NameLength"]
selector=SelectKBest(f_classif,k=5)
selector.fit(data[predictors],data["Survived"])
scores=-np.log10(selector.pvalues_)

plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)),predictors,rotation='vertical')

# 取出所有自变量名称
predictors = data.columns[1:]
X_train, X_test, y_train, y_test = model_selection.train_test_split(data[predictors], data.Survived, test_size = 0.3, random_state = 1234)

# 预设各参数的不同选项值
max_depth = [2,3,4,5,6]
min_samples_split = [2,4,6,8]
min_samples_leaf = [2,4,8,10,12]
# 将各参数值以字典形式组织起来
parameters = {'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}

# 网格搜索法，测试不同的参数值
grid_dtcateg = GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid = parameters, cv=10)
# 模型拟合
grid_dtcateg.fit(X_train, y_train)

# 返回最佳组合的参数值
# 构建分类决策树
CART_Class = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf = 4, min_samples_split=2)
# 模型拟合
decision_tree = CART_Class.fit(X_train, y_train)
# 模型在测试集上的预测
pred = CART_Class.predict(X_test)
# 模型的准确率
print('模型在测试集的预测准确率：\n',metrics.accuracy_score(y_test, pred))
'''


#随机森林
'''
data1 = data[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","Title","NameLength"]]
#0.8789237668161435
#data1 = data[["Pclass", "Sex", "Age","SibSp","Parch","Fare"]]
#0.8385650224215246
target = data["Survived"]


X_train, X_test, y_train, y_test = train_test_split(
    data1, target, test_size=0.25)

# 特征工程，类别
dct = DictVectorizer(sparse=False)
X_train = dct.fit_transform(X_train.to_dict(orient="records"))
X_test = dct.transform(X_test.to_dict(orient="records"))
print(dct.get_feature_names())

# 随机森林预测 超参数调优
rf = RandomForestClassifier()

# 网格搜索与交叉验证
params = {
    "n_estimators": [40, 60, 100, 120, 150, 200],
    "max_depth": [3, 5, 8, 10, 20, 30]
}

gs = GridSearchCV(rf, params, cv=10)
gs.fit(X_train, y_train)
print(gs.score(X_test, y_test))
print(gs.best_params_)

'''


#Adaboosting算法
'''
X_train= data.drop(["Survived"], axis=1)
Y_train = data["Survived"]

DTC = DecisionTreeClassifier()
#设置AdaBoost的参数
adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,20],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}
gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=10, 
                                 scoring="accuracy", n_jobs= 4, verbose = 1)
gsadaDTC.fit(X_train,Y_train)
ada_best = gsadaDTC.best_estimator_

print(ada_best)
ExtC = ExtraTreesClassifier()
# 优化决策树算法参数设置
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, 
                                    scoring="accuracy", n_jobs= 4, verbose = 1)
#应用到训练集上去
gsExtC.fit(X_train,Y_train)
ExtC_best = gsExtC.best_estimator_

'''
'''
结果：
AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',
                                                         splitter='random'),
                   learning_rate=0.1, random_state=7)
'''