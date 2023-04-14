from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class MyGaussianNaiveBayes:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.clf = GaussianNB()
        self.X_train = X_train  # Design Matrix
        self.X_test = X_test  # Design Matrix
        self.y_train = y_train  # Label
        self.y_test = y_test  # Label
    
    def train(self) -> float:
        self.clf.fit(self.X_train, self.y_train)
        return self.clf.score(self.X_test, self.y_test)

    def get_confusion_matrix(self):
        print(confusion_matrix(self.y_test, self.clf.predict(self.X_test)))
        print(f'f1-score: {f1_score(self.y_test, self.clf.predict(self.X_test))}')