import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv('data_all.csv')

y=data['status']
X=data.drop(['status'],axis=1)
print('the size of X,y:',X.shape,y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)

models=[('LogisticRegression',LogisticRegression(random_state=2018)),('SVM',SVC(gamma='auto',random_state=2018)),('DecisionTreeClassifier',DecisionTreeClassifier(random_state=2018))]
for name,model in models:
    model.fit(X_train,y_train)
    acc=accuracy_score(y_test,model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test,model.predict(X_test))
    model_auc=auc(fpr,tpr)
    print(name,'测试集正确率,AUC：',acc,model_auc)
