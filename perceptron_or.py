import numpy as np

from perceptron import Perceptron

# construindo o dataset OR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

print("TREINANDO PERCEPTRON...")

p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

print("TESTANDO PERCEPTRON...")

for (x, target) in zip(X, y):
    
    pred = p.predict(x)
    print("[INFO] dado: {}, valor real: {}, predicao do perceptron: {}".format(x, target[0], pred))