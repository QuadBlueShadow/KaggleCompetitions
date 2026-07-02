import numpy as np
import matplotlib.pyplot as plt
import math

import math
import numpy as np
import random

class LogReg:
    def __init__(self, features = 1):
        self.weights = []
        self.b = random.random() * 2 - 1
        self.lr = 0.01
        self.lam = 0

        for i in range(features):
            self.weights.append(random.random() * 2 - 1)

    def sigmoid(self, x):
        return 1 / (1 + pow(math.e, -x))
    
    def run_weights(self, x):
        output = 0

        for i in range(len(x)):
            output += self.weights[i] + x[i]

        return output
    
    def set_class(self, x):
        if x >= 0.5:
            return 1
        return 0
    
    def predict(self, x):
        return [self.set_class(self.sigmoid(self.run_weights(val) + self.b)) for val in x]

    def run(self, x):
        return [self.sigmoid(self.run_weights(val) + self.b) for val in x]
    
    def adjust_weights(self, x, error):
        for i in range(len(self.weights)):
            self.weights[i] -= x[i] * error * self.lr + self.lam * self.weights[i]
    
    def train(self, x, y):
        avg_error = None
        past_avg_error = None

        iterations = 0

        while True:
            # indices = np.random.permutation(len(x))

            # x = x[indices]
            # y = y[indices]

            predictions = self.run(x)

            if avg_error != None:
                past_avg_error = avg_error
                
            avg_error = 0

            for i in range(len(y)):
                error = (predictions[i] - y[i]) / len(y)
                avg_error += -(y[i] * math.log(predictions[i] if predictions[i] != 0 else 0.0001) + (1 - y[i]) * math.log(1 - predictions[i] if predictions[i] != 1 else 0.999))

                self.adjust_weights(x[i], error)
                self.b -= error * self.lr + self.lam * self.b

            avg_error = avg_error / len(y)

            if past_avg_error == None:
                past_avg_error = avg_error + 10

            #print(avg_error)

            iterations += 1

            if abs(avg_error - past_avg_error) < 0.0001 or iterations > 500:
                break
    
model = LogReg()

rng = np.random.default_rng()

values = np.concat((np.full(10, 0), np.full(10, 1)))
rng.shuffle(values)

indicies = []
for i in range(10, 30):
    indicies.append([i])

model.train(np.array(indicies), values)

# model.b = 10
# model.weights[0] = 10
print(model.weights, model.b)

plt.scatter(indicies, values)
plt.plot(indicies, model.run(indicies))
plt.show()