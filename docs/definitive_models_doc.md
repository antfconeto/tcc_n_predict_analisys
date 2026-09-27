> **Documento do modelo v1.** O R² 0,753 descrito aqui depende de um resíduo numérico no canal `rgb` e não se reproduz (≈ 0,59). O modelo atual está em [modelo_app_v2.md](modelo_app_v2.md).

# Documentação Técnica: Modelos Definitivos de Predição de Clorofila (SPAD)

Esta documentação descreve os dois melhores modelos preditivos desenvolvidos para estimar a clorofila Falker (SPAD) a partir de imagens aéreas RGB (dossel). Contém o pipeline detalhado de processamento de imagem, as fórmulas matemáticas e todos os parâmetros e pesos necessários para a replicação completa em qualquer linguagem (como Dart/Flutter, C#, Python ou C++).

---

## 1. Resumo dos Modelos

Através de validação cruzada rigorosa do tipo **GroupKFold por Data** (Out-of-Fold — LODO), com o melhor pré-processamento identificado na busca (`bilateral`, máscara ExG > 0.15, grade 10×10), os modelos apresentaram o seguinte desempenho geral (avaliados com as mesmas 34 imagens/amostras experimentais):

| Métrica de Desempenho | Rede Neural MLP Campeã (rgb28) | Melhor Índice Espectral (Median_r_over_b) |
| :--- | :---: | :---: |
| **R² (Validação Cruzada OOF)** | **0.7531** | 0.6225 |
| **RMSE (Erro Quadrático Médio)** | **2.0992 SPAD** | 2.5958 SPAD |
| **MAE (Erro Médio Absoluto)** | **1.6509 SPAD** | 1.9801 SPAD |

> **Nota:** A rede neural MLP campeã combina 28 estatísticas espaciais e atinge R² OOF ≥ 0.75. O modelo baseado no índice individual `Median_r_over_b` oferece simplicidade computacional, porém com acurácia inferior neste pré-processamento.

---

## 2. Pipeline de Pré-Processamento de Imagem

Para obter predições corretas e idênticas às do ambiente de pesquisa, qualquer imagem de entrada deve passar exatamente pela sequência de pré-processamento a seguir:

1. **Filtro Bilateral (Bilateral Filter):**
   * Aplica suavização preservando bordas na imagem original BGR.
   * Parâmetros OpenCV: `d = 9`, `sigmaColor = 75`, `sigmaSpace = 75`.

2. **Correção Gamma (Gamma = 0.8):**
   * Ajusta o contraste e a iluminação não linear dos canais.
   * Fórmula para cada pixel e canal I (valores de 0 a 255):
     
     I_gamma = 255.0 * ((I_raw / 255.0) ^ 0.8)

3. **Corte da Margem (Crop = 20%):**
   * Remove 20% da largura e altura das bordas externas da imagem. Isso elimina efeitos de borda, sombras externas e foca na área central do dossel.
   * Se a imagem possui dimensões originais W x H:
     * Margem de largura: x_start = 0.20 * W até x_end = 0.80 * W
     * Margem de altura: y_start = 0.20 * H até y_end = 0.80 * H

4. **Subamostragem em Blocos (Grid de 10×10 pixels):**
   * A imagem é dividida em blocos não sobrepostos de 10 × 10 pixels.
   * A média aritmética de cada canal (R, G, B) é calculada individualmente para cada bloco, gerando uma matriz reduzida de intensidades médias do dossel.

5. **Cálculo de Coordenadas Cromáticas e Canais Combinados:**
   Para cada bloco da grade subamostrada, as intensidades médias R, G e B são convertidas em coordenadas cromáticas normalizadas (que isolam a influência de variações na luz solar):
   * Soma Total: S = R + G + B (com proteção S = 1.0 se S == 0)
   * Canais Cromáticos:
     * r = R / S
     * g = G / S
     * b = B / S
   * Canais Combinados:
     * rg = (r + g) / 2.0
     * rb = (r + b) / 2.0
     * gb = (g + b) / 2.0
     * rgb = (r + g + b) / 3.0 (sempre constante em 0.33333)
   * Razões espectrais (com proteção contra divisão por zero):
     * r_over_b = r / b (se b == 0, usar 0)

6. **Máscara de Vegetação (ExG > 0.15):**
   * Para filtrar solo exposto, palha e sombras profundas, calcula-se o índice ExG (Excess Green) para cada bloco da grade:
     
     ExG = 2.0 * g - r - b
     
   * Um bloco é classificado como vegetação se ExG > 0.15. Apenas os blocos classificados como vegetação são mantidos para as predições.

7. **Estatísticas Espaciais:**
   As estatísticas de Mediana, Média, Percentil 75 (P75) e Percentil 90 (P90) são computadas usando os valores de todas as células/blocos válidos de vegetação na imagem.

---

## 3. Modelo 1: Melhor Índice Espectral (Fórmula Linear)

Este modelo baseia-se unicamente em um índice espectral unidimensional. O melhor preditor isolado foi a mediana do índice de reflectância do Vermelho em relação ao Azul (r/b) nos blocos vegetados.

### Parâmetros da Fórmula
*   **Índice:** Median_r_over_b (Mediana do valor r/b dos blocos vegetados).
*   **Equação Linear de Predição:**
    
    Clorofila Total (SPAD) = -10.52711 * Median_r_over_b + 59.77540

### Exemplo de Fluxo em Pseudocódigo (Dart/Flutter)
```dart
double predizerClorofilaIndice(List<double> valoresROverBVegetacao) {
  if (valoresROverBVegetacao.isEmpty) return 0.0;
  
  valoresROverBVegetacao.sort();
  int mid = valoresROverBVegetacao.length ~/ 2;
  double medianROverB = valoresROverBVegetacao.length.isOdd
      ? valoresROverBVegetacao[mid]
      : (valoresROverBVegetacao[mid - 1] + valoresROverBVegetacao[mid]) / 2.0;
  
  return -10.52711 * medianROverB + 59.77540;
}
```

---

## 4. Modelo 2: MLP Campeã (Rede Neural Artificial)

A MLP é uma rede neural multicamada (Multi-Layer Perceptron) do tipo regressora feedforward. Foi treinada no conjunto completo (todas as 34 amostras) utilizando regularização L2 (`alpha = 8.0`) e o otimizador L-BFGS.

### Arquitetura da Rede
*   **Camada de Entrada:** 28 neurônios (estatísticas dos 7 canais cromáticos).
*   **Normalizador de Entrada:** StandardScaler (média e desvio padrão de cada feature).
*   **Camada Oculta:** 1 camada contendo 3 neurônios com função de ativação ReLU (f(x) = max(0, x)).
*   **Camada de Saída:** 1 neurônio linear (Clorofila SPAD).

### Ordem exata dos 28 Canais de Entrada (X[i])
1.  Median_r
2.  Median_g
3.  Median_b
4.  Median_rg
5.  Median_rb
6.  Median_gb
7.  Median_rgb
8.  Mean_r
9.  Mean_g
10. Mean_b
11. Mean_rg
12. Mean_rb
13. Mean_gb
14. Mean_rgb
15. P75_r
16. P75_g
17. P75_b
18. P75_rg
19. P75_rb
20. P75_gb
21. P75_rgb
22. P90_r
23. P90_g
24. P90_b
25. P90_rg
26. P90_rb
27. P90_gb
28. P90_rgb

---

### Parâmetros do Normalizador (StandardScaler)

Para normalizar cada feature antes de alimentar a rede neural, calcula-se:
  
  X_scaled[i] = (X[i] - Mean[i]) / Scale[i]

Parâmetros de escala (Médias e Desvios Padrão), treinados com pré-processamento `bilateral`:

| Feature ID (i) | Nome da Feature | Média (Mean[i]) | Desvio Padrão (Scale[i]) |
| :---: | :--- | :---: | :---: |
| 1 | Median_r | 0.36665135 | 0.01694143 |
| 2 | Median_g | 0.45173120 | 0.01278448 |
| 3 | Median_b | 0.17864957 | 0.02040325 |
| 4 | Median_rg | 0.41067521 | 0.01020163 |
| 5 | Median_rb | 0.27413440 | 0.00639224 |
| 6 | Median_gb | 0.31667432 | 0.00847071 |
| 7 | Median_rgb | 0.33333334 | 1.00000000 |
| 8 | Mean_r | 0.36813524 | 0.01647643 |
| 9 | Mean_g | 0.45590662 | 0.01263341 |
| 10 | Mean_b | 0.17595814 | 0.02054350 |
| 11 | Mean_rg | 0.41202093 | 0.01027175 |
| 12 | Mean_rb | 0.27204669 | 0.00631670 |
| 13 | Mean_gb | 0.31593238 | 0.00823822 |
| 14 | Mean_rgb | 0.33333331 | 0.00000002 |
| 15 | P75_r | 0.38099160 | 0.01813449 |
| 16 | P75_g | 0.48020644 | 0.01602887 |
| 17 | P75_b | 0.20139974 | 0.01692762 |
| 18 | P75_rg | 0.42346079 | 0.01223848 |
| 19 | P75_rb | 0.28680883 | 0.00512370 |
| 20 | P75_gb | 0.32310277 | 0.00778760 |
| 21 | P75_rgb | 0.33333334 | 1.00000000 |
| 22 | P90_r | 0.39591764 | 0.01870839 |
| 23 | P90_g | 0.50977020 | 0.02059253 |
| 24 | P90_b | 0.22015803 | 0.01460095 |
| 25 | P90_rg | 0.43622987 | 0.01410187 |
| 26 | P90_rb | 0.29669243 | 0.00366629 |
| 27 | P90_gb | 0.32871721 | 0.00738461 |
| 28 | P90_rgb | 0.33333334 | 1.00000000 |

---

### Pesos e Biases da Rede Neural

#### 1. Camada Oculta: Pesos W1 (Matriz 28 × 3) e Biases b1 (Vetor de tamanho 3)

Cada entrada normalizada é multiplicada por uma coluna da matriz W1 e somada ao respectivo bias em b1:

```
Neurônio Oculto 1 (H1) Peso:
W1_col1 = [
   -0.09639647, -0.14458600,  0.12483132, -0.12482238,  0.14456806,  0.09640620,
    0.00000149, -0.13875458, -0.01601436,  0.12114071, -0.12114587,  0.01600909,
    0.13877138, -0.76305470, -0.00093734, -0.09384225,  0.14036998, -0.11272782,
    0.19106880,  0.25370607,  0.00000243,  0.02090202,  0.13386178,  0.21596549,
   -0.01433617,  0.22007121,  0.41137551, -0.00000270
]
Bias b1[0] = 12.36054400

Neurônio Oculto 2 (H2) Peso:
W1_col2 = [
   -0.07572270, -0.11351757,  0.09802229, -0.09801682,  0.11352471,  0.07571718,
    0.00000457, -0.10897570, -0.01257010,  0.09513013, -0.09512691,  0.01255649,
    0.10899268, -0.59902728, -0.00079945, -0.07369432,  0.11022557, -0.08854726,
    0.14995913,  0.19920793,  0.00000500,  0.01635177,  0.10508007,  0.16954934,
   -0.01130001,  0.17266516,  0.32299503,  0.00000370
]
Bias b1[1] = 5.52950210

Neurônio Oculto 3 (H3) Peso:
W1_col3 = [
   -0.04636407, -0.06944888,  0.05999801, -0.06000866,  0.06945927,  0.04635849,
   -0.00000245, -0.06670179, -0.00771989,  0.05826084, -0.05824261,  0.00772143,
    0.06670292, -0.36682641, -0.00045350, -0.04511301,  0.06746211, -0.05421466,
    0.09182246,  0.12197136, -0.00000719,  0.01006641,  0.06425774,  0.10382034,
   -0.00696700,  0.10583455,  0.19773186,  0.00000954
]
Bias b1[2] = 8.39661667
```

#### 2. Camada de Saída: Pesos W2 (Matriz 3 × 1) e Bias b2 (Vetor de tamanho 1)

Multiplica as saídas ativadas da camada oculta para produzir o valor final estimado:

```
H1 para Saída Peso: W2[0] = 1.08052629
H2 para Saída Peso: W2[1] = 0.84831688
H3 para Saída Peso: W2[2] = 0.51935323

Bias de Saída b2[0] = 15.21205233
```

---

### Algoritmo Completo de Inferência (Pseudocódigo / Dart)

O bloco a seguir demonstra como implementar a execução da rede neural de ponta a ponta:

```dart
// 1. Dados estatísticos do StandardScaler
final List<double> means = [
  0.36665135, 0.45173120, 0.17864957, 0.41067521, 0.27413440, 0.31667432, 0.33333334,
  0.36813524, 0.45590662, 0.17595814, 0.41202093, 0.27204669, 0.31593238, 0.33333331,
  0.38099160, 0.48020644, 0.20139974, 0.42346079, 0.28680883, 0.32310277, 0.33333334,
  0.39591764, 0.50977020, 0.22015803, 0.43622987, 0.29669243, 0.32871721, 0.33333334
];

final List<double> scales = [
  0.01694143, 0.01278448, 0.02040325, 0.01020163, 0.00639224, 0.00847071, 1.0,
  0.01647643, 0.01263341, 0.02054350, 0.01027175, 0.00631670, 0.00823822, 0.00000002,
  0.01813449, 0.01602887, 0.01692762, 0.01223848, 0.00512370, 0.00778760, 1.0,
  0.01870839, 0.02059253, 0.01460095, 0.01410187, 0.00366629, 0.00738461, 1.0
];

// 2. Pesos das conexões sinápticas
final List<double> w1Col1 = [
  -0.09639647, -0.14458600,  0.12483132, -0.12482238,  0.14456806,  0.09640620,
   0.00000149, -0.13875458, -0.01601436,  0.12114071, -0.12114587,  0.01600909,
   0.13877138, -0.76305470, -0.00093734, -0.09384225,  0.14036998, -0.11272782,
   0.19106880,  0.25370607,  0.00000243,  0.02090202,  0.13386178,  0.21596549,
  -0.01433617,  0.22007121,  0.41137551, -0.00000270
];

final List<double> w1Col2 = [
  -0.07572270, -0.11351757,  0.09802229, -0.09801682,  0.11352471,  0.07571718,
   0.00000457, -0.10897570, -0.01257010,  0.09513013, -0.09512691,  0.01255649,
   0.10899268, -0.59902728, -0.00079945, -0.07369432,  0.11022557, -0.08854726,
   0.14995913,  0.19920793,  0.00000500,  0.01635177,  0.10508007,  0.16954934,
  -0.01130001,  0.17266516,  0.32299503,  0.00000370
];

final List<double> w1Col3 = [
  -0.04636407, -0.06944888,  0.05999801, -0.06000866,  0.06945927,  0.04635849,
  -0.00000245, -0.06670179, -0.00771989,  0.05826084, -0.05824261,  0.00772143,
   0.06670292, -0.36682641, -0.00045350, -0.04511301,  0.06746211, -0.05421466,
   0.09182246,  0.12197136, -0.00000719,  0.01006641,  0.06425774,  0.10382034,
  -0.00696700,  0.10583455,  0.19773186,  0.00000954
];

final double b1_0 = 12.36054400;
final double b1_1 = 5.52950210;
final double b1_2 = 8.39661667;

final double w2_0 = 1.08052629;
final double w2_1 = 0.84831688;
final double w2_2 = 0.51935323;
final double b2_out = 15.21205233;

double relu(double val) => val < 0.0 ? 0.0 : val;

double predizerClorofilaMLP(List<double> rawFeatures) {
  // 1. Escalar entradas
  List<double> scaled = List.filled(28, 0.0);
  for (int i = 0; i < 28; i++) {
    scaled[i] = (rawFeatures[i] - means[i]) / scales[i];
  }
  
  // 2. Camada Oculta (Produto escalar + Bias + Ativação ReLU)
  double z1 = b1_0;
  double z2 = b1_1;
  double z3 = b1_2;
  for (int i = 0; i < 28; i++) {
    z1 += scaled[i] * w1Col1[i];
    z2 += scaled[i] * w1Col2[i];
    z3 += scaled[i] * w1Col3[i];
  }
  
  double h1 = relu(z1);
  double h2 = relu(z2);
  double h3 = relu(z3);
  
  // 3. Camada de Saída (Regressão Linear)
  double output = (h1 * w2_0) + (h2 * w2_1) + (h3 * w2_2) + b2_out;
  
  return output;
}
```
