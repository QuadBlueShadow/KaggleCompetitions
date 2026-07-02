import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics

values = np.random.random_sample(10)

indicies = []
for i in range(len(values)):
    indicies.append(i)

class LinReg:
    def __init__(self):
        self.x = 1
        self.b = 0

        self.lr = 0.005

    def run(self, x):
        return [self.x * val + self.b for val in x]
    
    def train(self, x, y):
        avg_error = None
        past_avg_error = None

        while True:
            predictions = self.run(x)

            if avg_error != None:
                past_avg_error = avg_error

            avg_error = 0

            for i in range(len(y)):
                error = 2 * (predictions[i] - y[i]) / len(y)
                avg_error += (predictions[i] - y[i]) ** 2

                self.x -= error * x[i] * self.lr
                self.b -= error * self.lr

            avg_error = avg_error / len(y)

            if past_avg_error == None:
                past_avg_error = avg_error + 1

            print(avg_error)

            if not (avg_error < past_avg_error):
                break

class PolyReg:
    def __init__(self, degree = 1):
        self.weights = []
        self.b = 0
        self.lam = 0.00001

        for i in range(degree):
            self.weights.append(1)

        self.lr = 0.0000001

    def run(self, x):
        return [(self.run_weights(val) + self.b) for val in x]
    
    def run_weights(self, x):
        val = 0

        for i in range(len(self.weights)):
            val += self.weights[i] * pow(x, (i + 1))

        return val
    
    def clamp(self, x, min, max):
        if x < min:
            return min
        if x > max:
            return max
        return x
    
    def train_weights(self, x, error):
        for i in range(len(self.weights)):
            self.weights[i] -= self.clamp(error * self.lr * pow(x, (i + 1)), -1, 1) + self.lam * self.weights[i]
    
    def train(self, x, y):
        avg_error = None
        past_avg_error = None

        while True:
            predictions = self.run(x)

            if avg_error != None:
                past_avg_error = avg_error
                
            avg_error = 0

            for i in range(len(y)):
                error = 2 * (predictions[i] - y[i]) / len(y)
                avg_error += (predictions[i] - y[i]) ** 2 + sum([self.lam * w for w in self.weights])

                self.train_weights(x[i], error)
                self.b -= error * self.lr + self.lam * self.b

            avg_error = avg_error / len(y)

            if past_avg_error == None:
                past_avg_error = avg_error + 10

            print(avg_error)

            if abs(avg_error - past_avg_error) < 0.0001:
                break

model = PolyReg(2)

model.train(indicies, values.tolist())

print()
print(model.weights, model.b)

plt.scatter(indicies, values)
plt.plot(indicies, model.run(indicies))
plt.show()