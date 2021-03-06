# Big Data Analytics SecondHomework

本次作業主要目的在於藉由sklearn的GridSearchCV測試怎樣的參數可以在xgboost得到最好的分類效果。

---

## (1) 首先先測試'max_depth'和'min_child_weight'的參數
```python=
param_test1 = {
    'max_depth':list(range(3,10,2)),
    'min_child_weight':list(range(1,6,2))
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
                       param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
```

得到最好的參數為```max_depth = 5，min_child_weight = 1```

## (2) 使用(1)的結果修改參數後，接著測試'gamma'參數
```python=
param_test2 = {
    'gamma':list([i/10.0 for i in range(0,10)])
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
                       param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
```

得到最好的參數為```'gamma' = 0.0```

## (3) 延續(2)的結果，tune'subsample'與'colsample_bytree'參數
```python=
param_test3 = {
     'subsample':list([i/10.0 for i in range(6,10)]),
     'colsample_bytree':list([i/10.0 for i in range(6,10)])
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
                       param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
```

得到最好的```subsample = 0.8，colsample_bytree = 0.8。```

## (4) 最後是'reg_alpha'
```python=
param_test4 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
                       param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
```

得到最好的```reg_alpha = 1e-05。```


經過這一連串的測試後可以使用剛剛產生的參數，可以得到最好的預測率。

### 最後再使用confusion matrix 驗證分析結果
```python=
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


data = pd.read_csv("./dataset/LargeTrain.csv")
label = 'Class'
features = [x for x in data.columns if x != label]
class_names = [ 'Class'+ str(x) for x in range(1,10)] 
X = data[features]
y = data[label]

#cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = XGBClassifier(max_depth=5,min_child_weight=1,gamma=0.0,
subsample=0.8,colsample_bytree=0.8,reg_alpha=1e-05,reg_lambda=0.01)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(20, 20), dpi=200)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()
```
![confusion matrix](./picture/confusion matrix.png)

---

### Reference
1. [XGBoostのパラメータチューニング実践 with Python](http://kamonohashiperry.com/archives/500)
2. [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
3. [Complete Guide to Parameter Tuning in XGBoost (with codes in Python)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
4. [Confusion matrix](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)
