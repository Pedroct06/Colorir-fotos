# Colorização Automática de Imagens com Deep Learning

Este notebook implementa uma **rede neural U-Net** para colorização automática de imagens em preto e branco usando TensorFlow/Keras.

## 📋 Descrição

O projeto utiliza deep learning para aprender a colorizar imagens automaticamente, transformando fotos em escala de cinza em imagens coloridas realistas. A abordagem utiliza o espaço de cores **LAB** (Luminância, A, B) ao invés de RGB para melhor separação entre luminosidade e informação de cor.

## 🎨 Como Funciona

### Espaço de Cores LAB

- **L (Luminância)**: Representa a escala de cinza (0-100)
- **A**: Canal de cor verde-vermelho (-128 a +127)
- **B**: Canal de cor azul-amarelo (-128 a +127)

**Vantagem**: Separar luminosidade (L) da informação de cor (A, B) permite que a rede aprenda apenas a adicionar cor, mantendo a estrutura original da imagem.

## 🛠️ Requisitos

```bash
pip install tensorflow
pip install opencv-python
pip install kagglehub
pip install matplotlib
pip install scikit-learn
```

Bibliotecas utilizadas:
- tensorflow/keras
- numpy
- opencv-python (cv2)
- matplotlib
- scikit-learn
- kagglehub

## 📁 Dataset

**Image Colorization Dataset**
- Fonte: Kaggle (`aayush9753/image-colorization-dataset`)
- Estrutura:
  - `train_black/` - Imagens de treino em preto e branco
  - `train_color/` - Imagens de treino coloridas
  - `test_black/` - Imagens de teste em preto e branco
  - `test_color/` - Imagens de teste coloridas
- Tamanho das imagens: Redimensionadas para 128×128 pixels

## 🔄 Pipeline de Processamento

### 1. Carregamento e Preparação dos Dados

```python
def carregar_imagem(input, esperado):
    # Para cada imagem:
    # 1. Carregar e redimensionar para 128x128
    img_bgr = cv2.imread(caminho)
    img_bgr = cv2.resize(img_bgr, (128, 128))
    
    # 2. Converter de BGR para LAB
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # 3. Separar canais
    l, a, b = cv2.split(img_lab)
    
    # 4. Normalizar canal L (entrada): [0, 255] → [0, 1]
    img_input_l = l / 255.0
    
    # 5. Normalizar canais AB (saída): [0, 255] → [-1, 1]
    img_out_ab = (ab - 128.0) / 128.0
    
    return np.array(Icinza), np.array(Icor)
```

### Normalização Explicada

| Canal | Intervalo Original | Normalização | Intervalo Final |
|-------|-------------------|--------------|-----------------|
| **L** | 0 - 255 | `L / 255.0` | 0.0 - 1.0 |
| **A, B** | 0 - 255 | `(AB - 128) / 128` | -1.0 - 1.0 |

## 🏗️ Arquitetura U-Net

### Estrutura da Rede

A U-Net é uma arquitetura encoder-decoder com conexões skip:

```
Entrada (128×128×1)
    ↓
┌─── Encoder (Contração) ───┐
│   C1: 128×128×64 → 64×64   │
│   C2: 64×64×128 → 32×32    │
│   C3: 32×32×256 → 16×16    │
└────────────────────────────┘
            ↓
    Bottleneck (16×16×512)
            ↓
┌─── Decoder (Expansão) ─────┐
│   U1: 16×16 → 32×32×256    │──┐
│   Concatenate com C2        │  │ Skip Connections
│   U2: 32×32 → 64×64×128    │──┤ (preservam detalhes)
│   Concatenate com C1        │  │
│   U3: 64×64 → 128×128×64   │──┘
└────────────────────────────┘
            ↓
    Saída (128×128×2)
```

### Implementação do Modelo

```python
def rede_neural():
    inputs = Input(shape=(128, 128, 1))
    
    # Encoder - Reduz dimensão espacial, aumenta profundidade
    c1 = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(inputs)
    c2 = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(c1)
    c3 = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(c2)
    
    # Bottleneck - Maior profundidade, menor dimensão
    b = Conv2D(512, (3,3), activation='relu', padding='same')(c3)
    
    # Decoder - Aumenta dimensão espacial, reduz profundidade
    u1 = Conv2DTranspose(256, (3,3), activation='relu', strides=2)(b)
    merge = Concatenate()([u1, c2])  # Skip connection
    c4 = Conv2D(256, (3,3), activation='relu', padding='same')(merge)
    
    u2 = Conv2DTranspose(128, (3,3), activation='relu', strides=2)(c4)
    merge = Concatenate()([u2, c1])  # Skip connection
    c5 = Conv2D(128, (3,3), activation='relu', padding='same')(merge)
    
    u3 = Conv2DTranspose(64, (3,3), activation='relu', strides=2)(c5)
    merge = Concatenate()([u3, inputs])  # Skip connection
    c6 = Conv2D(64, (3,3), activation='relu', padding='same')(merge)
    
    # Saída - 2 canais (A e B)
    outputs = Conv2D(2, (3,3), activation='tanh', padding='same')(c6)
    
    return Model(inputs=[inputs], outputs=[outputs])
```

### Componentes Principais

#### Conv2D (Convolução)
- Extrai características das imagens
- Filtros 3×3 para detectar padrões
- `padding='same'`: mantém dimensões
- `strides=2`: reduz dimensão pela metade

#### Conv2DTranspose (Deconvolução)
- Aumenta resolução espacial
- Reconstrói detalhes da imagem
- `strides=2`: dobra a dimensão

#### Concatenate (Skip Connections)
- Une features do encoder com decoder
- Recupera detalhes espaciais perdidos
- **Essencial para qualidade da imagem**

#### Funções de Ativação
- **ReLU** (`relu`): Camadas intermediárias
  - Introduz não-linearidade
  - Computacionalmente eficiente
- **Tanh** (`tanh`): Camada de saída
  - Retorna valores em [-1, 1]
  - Necessário pois AB foi normalizado neste intervalo

## 🎯 Treinamento

### Configuração

```python
model = rede_neural()
model.compile(
    optimizer='adam',      # Otimizador adaptativo
    loss='mae'            # Mean Absolute Error
)
```

### Escolhas de Design

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| **Otimizador** | Adam | Adaptativo, converge rápido, funciona bem em muitos cenários |
| **Função de Perda** | MAE | Melhor que MSE para esta tarefa, menos sensível a outliers |
| **Épocas** | 100 | Número de vezes que o dataset completo é processado |
| **Batch Size** | 32 | Compromisso entre memória e convergência |

### Execução do Treinamento

```python
model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, Y_test)
)
```

**validation_data**: Previne overfitting ao monitorar performance em dados não vistos

## 📊 Visualização de Resultados

### Função de Visualização

```python
def ver_imagem(i):
    # 1. Prever canais AB
    L_input = X_test[i]
    ab_predict = model.predict(L_input.reshape(1, 128, 128, 1))[0]
    
    # 2. Desnormalizar
    L_ajustado = L_input * 100.0           # [0,1] → [0,100]
    ab_ajustado = (ab_predict * 128.0) * 3  # [-1,1] → [-128,127] com boost
    
    # 3. Recombinar LAB
    LAB = np.concatenate([L_ajustado, ab_ajustado], axis=2)
    
    # 4. Converter LAB → RGB
    RGB = cv2.cvtColor(LAB.astype('float32'), cv2.COLOR_LAB2RGB)
    RGB = np.clip(RGB, 0, 1)
    
    # 5. Plotar lado a lado
    plt.subplot(1, 2, 1)
    plt.imshow(RGB)  # Predição
    
    plt.subplot(1, 2, 2)
    plt.imshow(RGB_real)  # Original
```

### Processo de Reconstrução

1. **Entrada**: Imagem em escala de cinza (canal L)
2. **Predição**: Rede neural gera canais A e B
3. **Desnormalização**: Reverter transformações
4. **Recombinação**: Juntar L + AB → LAB
5. **Conversão**: LAB → RGB para visualização
6. **Clipping**: Garantir valores válidos [0, 1]

## 🚀 Como Usar

### 1. Preparar o Ambiente

```python
# Baixar dataset do Kaggle
dataset_path = kagglehub.dataset_download("aayush9753/image-colorization-dataset")

# Configurar caminhos
treino_cinza = os.path.join(dataset_path, 'data/train_black')
treino_cor = os.path.join(dataset_path, 'data/train_color')
teste_cinza = os.path.join(dataset_path, 'data/test_black')
teste_cor = os.path.join(dataset_path, 'data/test_color')
```

### 2. Carregar Dados

```python
X_train, Y_train = carregar_imagem(treino_cinza, treino_cor)
X_test, Y_test = carregar_imagem(teste_cinza, teste_cor)
```

### 3. Treinar Modelo

```python
model = rede_neural()
model.compile(optimizer='adam', loss='mae')
model.fit(X_train, Y_train, epochs=100, batch_size=32, 
          validation_data=(X_test, Y_test))
```

### 4. Visualizar Resultados

```python
# Ver colorização da primeira imagem de teste
ver_imagem(0)

# Ver outras imagens
ver_imagem(5)
ver_imagem(10)
```

## 💡 Insights Técnicos

### Por que U-Net?

1. **Skip Connections**: Preservam detalhes espaciais durante upsampling
2. **Simetria**: Encoder e decoder balanceados
3. **Comprovada**: Excelente para tarefas de segmentação e colorização

### Por que Espaço LAB?

1. **Separação Natural**: L contém estrutura, AB contém cor
2. **Perceptualmente Uniforme**: Mudanças numéricas correspondem a mudanças visuais
3. **Facilita Aprendizado**: Rede foca apenas em adicionar cor

### Detalhes de Implementação

- **Multiplicador ×3 no AB**: Intensifica cores na reconstrução
- **GPU Recomendada**: Treinamento é intensivo (T4 no Colab)
- **Imagens 128×128**: Compromisso entre qualidade e velocidade

## 📈 Melhorias Possíveis

### Arquitetura
- Aumentar resolução para 256×256 ou 512×512
- Adicionar mais camadas no bottleneck
- Implementar attention mechanisms

### Treinamento
- Data augmentation (rotação, flip, zoom)
- Learning rate scheduling
- Early stopping com checkpoint

### Pós-processamento
- Ajuste fino de saturação/brilho
- Ensemble de múltiplos modelos
- Refinamento com GANs

## ⚠️ Limitações

- **Resolução**: 128×128 pode perder detalhes finos
- **Ambiguidade**: Alguns objetos podem ter múltiplas cores válidas
- **Dataset**: Performance depende da qualidade dos dados de treino
- **Recursos**: Requer GPU para treinamento eficiente

## 🔗 Links Úteis

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [LAB Color Space](https://en.wikipedia.org/wiki/CIELAB_color_space)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Dataset Kaggle](https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset)

## 📄 Licença

Este notebook está disponível no GitHub: [Colorização de fotos](https://github.com/Pedroct06/Coloriza-o-de-fotos)

---

**Nota**: Este é um projeto educacional demonstrando técnicas de deep learning para processamento de imagens. Os resultados podem variar dependendo do dataset e hiperparâmetros utilizados.
