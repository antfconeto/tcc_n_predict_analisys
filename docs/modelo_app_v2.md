# Modelo de predição de clorofila — app v2

Gerado por `pipeline/03_modelos_clorofila/02_definitive_model.py`. Substitui a versão 1
(`definitive_models_doc.md`), cujas 28 entradas incluíam as colunas `*_rgb`.

## Por que mudou

O canal `rgb` = (r+g+b)/3 vale sempre 1/3. Na versão 1, o resíduo de arredondamento de `Mean_rgb`
(float32) virava uma entrada grande depois da padronização, e o R² de 0,753 dependia dele. No app,
calculado em float64, essa entrada fica constante. A versão 2 remove as 4 colunas `*_rgb` e passa a
descartar pixels de sombra e reflexo.

## Desempenho (validação por data, fotos de datas fora do treino)

| Modelo | R² | RMSE (SPAD) | MAE (SPAD) |
| :--- | :---: | :---: | :---: |
| MLP v2 (24 entradas, máscara HSV) | 0.616 | 2.62 | 1.99 |
| Melhor índice único (`Mean_r_over_b`) | 0.648 | 2.51 | 1.99 |
| MLP v1 sem o artefato (28 → 24 entradas, sem máscara) | 0.589 | – | – |

Para escala: uma única leitura do Falker se afasta em média cerca de 2 SPAD da média da parcela.

## Pré-processamento (tem de ser idêntico no app)

1. Filtro bilateral: `d = 9`, `sigmaColor = 75`, `sigmaSpace = 75` (imagem BGR original).
2. Gamma 0,8: `I = 255 · (I_raw / 255) ^ (1 / 0,8)`, por canal, tabela de 256 valores truncada para inteiro.
3. Recorte de 20% de cada borda.
4. **Máscara HSV (novo)**, em cada pixel da imagem já com gamma (HSV do OpenCV: H 0–179, S e V 0–255):
   o pixel é descartado se `V < 45`, `V > 240`, `S < 25` ou (`V > 240` e `S < 20`).
5. Blocos de 10×10 px: média de B, G e R **só dos pixels válidos**; bloco sem pixel válido é descartado.
6. Por bloco: `S = R + G + B` (1 se 0); `r = R/S`, `g = G/S`, `b = B/S`;
   `rg = (r+g)/2`, `rb = (r+b)/2`, `gb = (g+b)/2`.
7. Vegetação: blocos com `ExG = 2g − r − b > 0,15`.
8. Estatísticas nos blocos de vegetação: mediana, média, percentil 75 e percentil 90
   (percentis com interpolação linear, como `numpy.percentile`).

## Entradas, na ordem

1. `Median_r`
2. `Median_g`
3. `Median_b`
4. `Median_rg`
5. `Median_rb`
6. `Median_gb`
7. `Mean_r`
8. `Mean_g`
9. `Mean_b`
10. `Mean_rg`
11. `Mean_rb`
12. `Mean_gb`
13. `P75_r`
14. `P75_g`
15. `P75_b`
16. `P75_rg`
17. `P75_rb`
18. `P75_gb`
19. `P90_r`
20. `P90_g`
21. `P90_b`
22. `P90_rg`
23. `P90_rb`
24. `P90_gb`

## Rede

`x_pad[i] = (x[i] − means[i]) / scales[i]` → `h_j = max(0, b1[j] + Σ x_pad[i] · W1[i][j])` (3 neurônios)
→ `SPAD = b2 + Σ h_j · W2[j]`.

```dart
final means = <double>[
  0.3650856597, 0.4454965381, 0.185784875, 0.4071075592, 0.2772517336, 0.3174571684,
  0.3668012356, 0.4495767986, 0.183621968, 0.4081890154, 0.2752116033, 0.3165993822,
  0.3785672521, 0.4729445752, 0.2094195305, 0.4199558567, 0.2893372012, 0.3233936376,
  0.3925799359, 0.5005866052, 0.2291475824, 0.4324856734, 0.2984352796, 0.328262791
];

final scales = <double>[
  0.01657801257, 0.01247909278, 0.02258935069, 0.01129467519, 0.006239540115, 0.008289006615,
  0.01611359008, 0.01218015427, 0.02243377329, 0.01121688764, 0.006090080152, 0.008056796841,
  0.018458739, 0.01605415681, 0.01893410574, 0.01331242513, 0.004740942656, 0.007281600071,
  0.01952559031, 0.02017131303, 0.01625211151, 0.01509653982, 0.003276686374, 0.006499319654
];

final w1Col1 = <double>[
  0.03816713343, -0.177288899, 0.0771810843, -0.07867566626, 0.1800429915, -0.04099526146,
  0.04597242545, -0.1739541977, 0.05958084841, -0.05977009606, 0.1723366976, -0.0435360424,
  -0.03221792661, -0.1776304433, 0.03727927738, -0.08433569042, 0.08493349261, -0.1283378689,
  -0.07189387775, -0.1558423558, -0.02926782137, -0.08001746229, -0.04698724432, -0.2299355794
];

final w1Col2 = <double>[
  -0.1651361919, -0.09619872888, 0.1856090252, -0.1863345279, 0.09216204737, 0.1667695676,
  -0.2301466824, -0.1034851751, 0.2216355555, -0.2192214104, 0.1000521861, 0.2317842729,
  -0.1009657291, -0.229158695, 0.1403269575, -0.2700779179, 0.0420945032, 0.3043204282,
  -0.1099963164, -0.2450408192, 0.1630469379, -0.273000839, 0.04210113694, 0.5218993583
];

final w1Col3 = <double>[
  -0.1521721885, 0.1196035622, 0.05464022201, -0.05098836063, -0.1238375553, 0.1538137949,
  -0.1661135959, 0.05459591005, 0.08780248871, -0.09262180339, -0.05823162508, 0.168801313,
  -0.1870712379, 0.02746851113, 0.01232592394, -0.1292465907, -0.1876309583, 0.1326868909,
  -0.21687239, -0.0747080653, -0.04616244275, -0.2008541484, -0.2773612397, 0.09792584705
];

final b1 = <double>[0.1125241633, 26.22978502, -0.4327608693];
final w2 = <double>[-0.5756931009, 1.039143891, -0.6573748782];
const b2 = 11.01915701;
```

Rede treinada com todas as 34 fotos (semente 42); R² de ajuste
nas próprias fotos: 0.740 (otimista — o valor a citar é o da validação por data).
