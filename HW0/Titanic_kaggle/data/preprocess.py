# package
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender = pd.read_csv("gender_submission.csv")
# train.info()
# 共891個index，沒有滿的就是空值
# train空值：Age, Cabin, Embarked
# test.info()
# 418
# test空值：Age, Fare, Cabin
data = train.append(test)

#查看資料_性別
sns.countplot(data['Sex'], hue=data['Survived'])
plt.show()
# 生存人數:女>男
survived_sex = data[['Sex','Survived']].groupby(['Sex'], as_index=True).mean().round(3)
# 生存比例：女=>0.742，男=>0.189

#查看資料_艙等
sns.countplot(data['Pclass'], hue=data['Survived'])
plt.show()
# 生存人數:1>2>3
survived_pclass = data[['Pclass','Survived']].groupby(['Pclass'], as_index=True).mean().round(3)
# 生存比例：1=>0.63，2=>0.473，3=>0.242

del survived_sex, survived_pclass

# 把性別變成0/1編碼
train['Sex_code'] = train['Sex'].map({'female':int(1), 'male':int(0)})
test['Sex_code'] = test['Sex'].map({'female':int(1), 'male':int(0)})

# x = train.drop(labels=['Survived','PassengerId'],axis=1)
x = train
y = train['Survived']
x1 = test

# 先建base model(只看性別與艙等的模型)
base = ['Sex_code']
# base = ['Sex_code','Pclass']
base_model = RandomForestClassifier(random_state=2, n_estimators=250, min_samples_split=20, oob_score=True)
base_model.fit(x[base],y)
print('base oob score = {}'.format(base_model.oob_score_))
a = base_model.predict(x1[base])
print()