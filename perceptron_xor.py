import numpy as np

from perceptron import Perceptron

# construindo o dataset XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("TREINANDO PERCEPTRON...")

p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

print("TESTANDO PERCEPTRON...")

for (x, target) in zip(X, y):
    
    pred = p.predict(x)
    print("[INFO] dado: {}, valor real: {}, predicao do perceptron: {}".format(x, target[0], pred))

# no exemplo XOR, o perceptron nunca vai acertar tudo, pois o perceptron simples 
# como foi concebido pelos autores nao consegue se sair bem em datasets 
# que nao sao linearmente separaveis