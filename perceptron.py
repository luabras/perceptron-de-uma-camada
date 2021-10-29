import numpy as np

class Perceptron:

    # n eh o numero de inputs/features
    # alpha eh a taxa de aprendizado
    def __init__(self, n, alpha=0.1):

        # inicializando a matriz de pesos com n+1 nos (1 eh o bias)/raiz quadrada de n
        # randn gera numeros aleatorios com distribuicao normal
        self.W = np.random.randn(n+1)/np.sqrt(n)
        self.alpha = alpha

    # funcao de ativacao step, retorna 1 se o input for maior que 0, e 0 caso contrario
    def step(self, x):
        return 1 if x > 0 else 0

    # funcao de treinamento
    # X eh o input de features
    # y eh o output alvo de classe
    # epochs eh o numero de epocas a serem treinadas
    def fit(self, X, y, epochs=10):

        # colocando uma coluna de 1's no inicio da matriz de features para representar o bias
        # isso permite que o bias seja um parametro treinavel na matriz de pesos
        X = np.c_[X, np.ones((X.shape[0]))]

        # percorrendo o numero de epocas
        for epoch in np.arange(0, epochs):
            # percorrendo cada dado individualmente
            for (x, target) in zip(X, y):
                # pegando o produto entre os inputs e a matriz de pesos e passando
                # esse valor para a funcao de ativacao para obter o output/predicao/classe
                p = self.step(np.dot(x, self.W))
                # so atualizar o peso se a predicao for diferente da classe alvo
                if p != target:
                    # calculando o erro
                    error = p - target
                    # atualizando os pesos
                    self.W += -self.alpha * error * x

    # funcao de predicao
    # X eh o input de features
    def predict(self, X, addBias=True):
        # verificando que o nosso input eh uma matriz
        X = np.atleast_2d(X)

        # verificando se precisamos adicionar a coluna de bias
        if addBias:
            # adicionando a coluna de bias na ultima entrada da matriz de features
            X = np.c_[X, np.ones((X.shape[0]))]

        # retornando a predicao
        return self.step(np.dot(X, self.W))