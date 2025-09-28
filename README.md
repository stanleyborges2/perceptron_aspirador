Stanley Borges papa 12415298

# Perceptron em Python

Este projeto apresenta uma implementação simples do **Perceptron**, considerado o primeiro modelo de neurônio artificial da história da Inteligência Artificial (IA).
O objetivo é demonstrar seu funcionamento básico, as etapas do treinamento e refletir sobre sua importância histórica e aplicações práticas.

---

## 1. Conceito

O **Perceptron** é o modelo que deu origem ao conceito de neurônio artificial. Ele é composto por quatro partes principais:

* **Entradas (inputs):** representam as variáveis de entrada.
* **Conexões (pesos):** cada entrada é multiplicada por um peso que indica sua relevância.
* **Corpo da célula (função soma):** combina as entradas ponderadas e adiciona o bias.
* **Saída:** resultado binário após aplicar a função de ativação.

Sua importância histórica está no fato de ter aberto as portas para o desenvolvimento de **redes neurais artificiais**, que hoje são a base de grande parte dos sistemas de **Machine Learning** e **Deep Learning**.

---

## 2. Funcionamento

O Perceptron é um **classificador linear**. Isso significa que ele consegue separar dados em duas classes distintas (ex.: **Sim/Não**, **Apto/Não Apto**, **Positivo/Negativo**) por meio de uma linha reta (ou hiperplano, em dimensões maiores).

### Limitação

O modelo não consegue resolver problemas que **não são linearmente separáveis**. Um exemplo clássico é o problema do **XOR**, em que não é possível separar as classes com uma linha reta.

---

## 3. Estrutura do Código

O código implementado em Python segue três etapas principais:

1. **Função de entrada (`perceptron_input`)**
   Recebe os valores de entrada, pesos e bias. Calcula a **soma ponderada**:
   [
   \text{Soma} = \sum (entrada \times peso) + bias
   ]

2. **Função de ativação (`perceptron_output`)**
   Converte o valor da soma em uma saída binária:

   * `1` se a soma ≥ 0
   * `0` se a soma < 0

3. **Execução no `main.py`**
   O programa roda os cálculos e exibe a saída do Perceptron, além de mensagens simples como `"Hello World!"` e `"This is the main file"` para reforçar a execução do código principal.

---

## 4. Aplicação Prática

Um exemplo de aplicação real de um Perceptron simples é em **sistemas de triagem inicial de currículos em RH**.

* Entradas podem ser variáveis como nível de proficiência em idiomas (1 = básico, 4 = fluente), ranking da universidade (1 a 5), ou experiência prévia (anos).
* Os **pesos** representariam a importância de cada critério para a vaga.
* O **bias** calibraria o corte mínimo.
* O **resultado binário** poderia indicar:

  * `1` → candidato apto para próxima fase.
  * `0` → candidato não apto.

Isso permitiria filtrar grandes volumes de candidatos de forma automática e eficiente.

---

## 5. Execução do Código

1. Certifique-se de ter o Python instalado (3.10 ou superior).
2. Instale as dependências necessárias (nesse caso apenas o `numpy`):

   ```bash
   pip install numpy
   ```
3. Execute o programa:

   ```bash
   python main.py
   ```

---

## 6. Conclusão

O Perceptron foi o **ponto de partida para a IA moderna**. Apesar de suas limitações como classificador linear, seu conceito serviu de base para arquiteturas muito mais avançadas, como **Redes Neurais Profundas (Deep Learning)**, que hoje alimentam desde sistemas de recomendação até carros autônomos.
