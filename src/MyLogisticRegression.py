from sklearn.linear_model import LogisticRegression

class MyLogisticRegression:
    def __init__(self, X, Y):
        self.reg = LogisticRegression(solver='sag', max_iter=100000)
        self.X = X  # Design Matrix
        self.Y = Y  # Label
    
    def train(self):
        self.reg.fit(self.X, self.Y)
        return self.reg.score(self.X, self.Y)