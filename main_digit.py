
import time
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.model_selection import train_test_split
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
    print('training time for', name, 'on digits dataset :', end - start, 'seconds')
    
    Y_pred = model.predict(X_ts)
    error = (Y_ts != Y_pred).sum()
    print('Misclassified samples: ', error)
    print('Accuracy: ', metrics.accuracy_score(Y_ts, Y_pred), '\n')



data, target = datasets.load_digits(return_X_y=True)
X_tr, X_ts, Y_tr, Y_ts = train_test_split(data, target, test_size=0.3,
                                          random_state=1, stratify=target)

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
