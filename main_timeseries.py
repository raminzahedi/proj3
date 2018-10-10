
import time
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn import metrics

from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def model_run(model, name):
    
    start = time.time()
    model.fit(X_tr, Y_tr)    
    end = time.time()
    print('training time for', name, 'on EEG Eye State dataset :', end - start, 'seconds')
    
    Y_pred = model.predict(X_ts)
    error = (Y_ts != Y_pred).sum()
    print('Misclassified samples: ', error)
    print('Accuracy: ', metrics.accuracy_score(Y_ts, Y_pred), '\n')



data = data = pd.read_csv('EEG Eye State.csv', header=None)
X_tr = data.loc[:10000,0:13]
X_ts = data.loc[10001:,0:13]
Y_tr = data.loc[:10000,14]
Y_ts = data.loc[10001:,14]

model = Perceptron()
model_run(model, 'Perceptron')

model = SGDClassifier(loss='hinge')
model_run(model, 'Linear SVM')

model = SVC(kernel='rbf')
model_run(model, 'Non-linear SVM')

model = DecisionTreeClassifier(criterion='gini')
model_run(model, 'Decision Tree')

model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
model_run(model, 'KNN')

model = LogisticRegression(C=100)
model_run(model, 'Logistic Regression')   
