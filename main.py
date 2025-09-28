
import numpy as np

# ==============================
# Funções auxiliares
# ==============================
def funcao_ativacao_degrau(valor):
    """Função de ativação do tipo degrau"""
    return 1 if valor >= 0 else 0

def normalizar(valor, minimo, maximo):
    """Normaliza um valor para a faixa [0, 1]"""
    return (valor - minimo) / (maximo - minimo)

def desnormalizar(valor, minimo, maximo):
    """Desnormaliza um valor para a escala original"""
    return valor * (maximo - minimo) + minimo

def mapear_tipo_piso(piso):
    """One-hot encoding para tipo de piso"""
    mapping = {
        'madeira':  [1, 0, 0],
        'ceramica': [0, 1, 0],
        'carpete':  [0, 0, 1]
    }
    if piso not in mapping:
        raise ValueError("Tipo de piso inválido")
    return mapping[piso]

# ==============================
# Classe Perceptron
# ==============================
class Perceptron:
    def __init__(self, num_inputs, taxa_aprendizado=0.1):
        self.weights = np.random.uniform(-1.0, 1.0, size=num_inputs)
        self.bias = np.random.uniform(-1.0, 1.0)
        self.lr = taxa_aprendizado

    def treinar(self, X, y, epochs=100):
        for epoch in range(epochs):
            total_error = 0
            for xi, target in zip(X, y):
                soma = np.dot(xi, self.weights) + self.bias
                pred = funcao_ativacao_degrau(soma)
                erro = target - pred
                self.weights += self.lr * erro * xi
                self.bias += self.lr * erro
                total_error += abs(erro)
            if total_error == 0:
                print(f"Treinamento concluído em {epoch+1} épocas")
                break

    def prever(self, entrada):
        soma = np.dot(entrada, self.weights) + self.bias
        return funcao_ativacao_degrau(soma)

# ==============================
# Dataset de exemplo
# ==============================
inputs = np.array([
    mapear_tipo_piso('madeira') + [normalizar(1, 0, 10), normalizar(5, 0, 5)],
    mapear_tipo_piso('carpete') + [normalizar(9, 0, 10), normalizar(0.5, 0, 5)]
])

outputs_potencia = np.array([
    normalizar(1, 1, 3),  # baixa
    normalizar(3, 1, 3)   # alta
])

outputs_velocidade = np.array([
    normalizar(5, 1, 5),  # alta
    normalizar(1, 1, 5)   # baixa
])

# ==============================
# Treinamento
# ==============================
print("=== Treinando Perceptron de Potência ===")
p_potencia = Perceptron(inputs.shape[1])
p_potencia.treinar(inputs, outputs_potencia)

print("\n=== Treinando Perceptron de Velocidade ===")
p_velocidade = Perceptron(inputs.shape[1])
p_velocidade.treinar(inputs, outputs_velocidade)

# ==============================
# Testes
# ==============================
print("\n=== Testes Finais ===")
for i, entrada in enumerate(inputs):
    pot = desnormalizar(p_potencia.prever(entrada), 1, 3)
    vel = desnormalizar(p_velocidade.prever(entrada), 1, 5)
    print(f"Entrada: {entrada}")
    print(f"Potência prevista: {pot:.2f} | Esperado: {desnormalizar(outputs_potencia[i], 1, 3):.2f}")
    print(f"Velocidade prevista: {vel:.2f} | Esperado: {desnormalizar(outputs_velocidade[i], 1, 5):.2f}\n")

