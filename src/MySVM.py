from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

class MySVM:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = StandardScaler().fit_transform(X_train)  # Design Matrix
        self.X_test = StandardScaler().fit_transform(X_test)  # Design Matrix
        self.y_train = y_train  # Label
        self.y_test = y_test  # Label
        
        parameters = {'C': [0.01, 0.1, 1, 10, 100, 1000]}
        self.clf = GridSearchCV(svm.LinearSVC(max_iter=100000, loss='hinge'), param_grid=parameters, n_jobs=-1)
    
    def train(self) -> float:
        self.clf.fit(self.X_train, self.y_train)
        print(f'Best Hyperparameters: {self.clf.best_params_}')
        return self.clf.score(self.X_test, self.y_test)
    
    def get_confusion_matrix(self):
        print(confusion_matrix(self.y_test, self.clf.predict(self.X_test)))
        print(f'f1-score: {f1_score(self.y_test, self.clf.predict(self.X_test))}')