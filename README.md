# data-for-a-week
## 数据集说明
- 这是我们本次实践数据的[下载地址](https://pan.baidu.com/s/1dtHJiV6zMbf_fWPi-dZ95g)
- 说明：这份数据集是金融数据（非原始数据，已经处理过了），我们要做的是预测贷款用户是否会逾期。表格中 "status" 是结果标签：0表示未逾期，1表示逾期。
## 模型构建
### 1.首先是导入需要用到的库
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
```
`the size of X,y: (4754, 84) (4754,)`
### 2.对数据进行处理，分开数据和标签
```python
data=pd.read_csv('data_all.csv')
y=data['status']
X=data.drop(['status'],axis=1)
print('the size of X,y:',X.shape,y.shape)
'''
### 3.划分训练集和测试集
将训练集与测试集7-3分
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)
```
### 4.模型构建及评估
分别构建逻辑回归、SVM、和决策树模型，并对三个模型进行评分，评分标准为sklearn.metrics里的Accuracy_score和AUC两个评价指标。
```python
models=[('LogisticRegression',LogisticRegression(random_state=2018)),('SVM',SVC(gamma='auto',random_state=2018)),('DecisionTreeClassifier',DecisionTreeClassifier(random_state=2018))]
for name,model in models:
    model.fit(X_train,y_train)
    acc=accuracy_score(y_test,model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test,model.predict(X_test))
    model_auc=auc(fpr,tpr)
    print(name,'测试集正确率,AUC_score：',acc,model_auc)
```
最终模型结果如下：

        模型名称       |  accuracy_score  |         AUC
----------------------|------------------|------------------
  LogisticRegression  |0.7484232655921513|0.5
          SVM         |0.7484232655921513|0.5
DecisionTreeClassifier|0.6846531184302733|0.5942367479369453

## 参考文档
- sklearn官方英文文档：https://scikit-learn.org/stable/index.html
- sklearn中文版文档：http://sklearn.apachecn.org/#/
