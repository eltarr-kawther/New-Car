import numpy as np

class LinearRegression:
    def __init__(self, X, y, alpha=0.001):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.B = np.zeros((self.X.shape[1],))

    def cost_function(self, X, y, B):
        m = len(y)
        J = np.sum((X.dot(B) - y) ** 2)/(2 * m)
        return J
    
    def gradient_descent(self, X, y, B, iterations=100000):
        cost_history = [0]*iterations
        m = len(y)
        for iteration in range(iterations):
            # Hypothesis Values
            h = X.dot(B)
            # Difference b/w Hypothesis and Actual Y
            loss = h - y
            # Gradient Calculation
            gradient = X.T.dot(loss) / m
            # Changing Values of B using Gradient
            B = B - self.alpha * gradient
            # New Cost Value
            cost = self.cost_function(X, y, B)
            cost_history[iteration] = cost

        return B, cost_history
    
    def rmse(self, y, y_pred):
        rmse = np.sqrt(sum((y - y_pred) ** 2) / len(y))
        return rmse
    
    def r2_score(self, y, y_pred):
        mean_y = np.mean(y)
        ss_tot = sum((y-mean_y)**2)
        ss_res = sum((y-y_pred)**2)
        r2 = 1 - (ss_res/ss_tot)
        return r2
    
    def fit(self):
        inital_cost = self.cost_function(self.X, self.y, self.B)
        print("Initial Cost")
        print(inital_cost)
        n_B, cost_history = self.gradient_descent(self.X, self.y, self.B, iterations=100000)
        print("New Coefficients")
        print(n_B)
        
        print("Final Cost")
        print(cost_history[-1])
        return n_B
    
    def predict(self, n_B):
        y_pred = self.X.dot(n_B)
        return y_pred
