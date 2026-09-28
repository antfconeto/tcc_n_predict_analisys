# tcc_n_predict_analisys

Predição de clorofila (SPAD) e, por consequência, do estado de nitrogênio em capim Marandu a partir de imagens RGB do dossel. Os resultados alimentam o app Nutrinitro.

- **Experimento:** delineamento em blocos casualizados, 3 blocos × 4 doses de N (0, 50, 75 e 100 kg/ha).
- **Coletas:** 4 datas (18/05, 21/05, 26/05 e 01/06/2026).
- **Referência:** clorofilômetro Falker.

Esta é a versão organizada da análise que antes ficava na pasta `tcc/` (scripts `01`–`17` e `another_scripts/`). Os dados e resultados agora vivem em um **banco SQLite local**, e não mais em CSVs soltos.

## Estrutura

```
run.py                     executor do pipeline e consultas ao banco
tcc_analysis/              código compartilhado
  config.py                caminhos, delineamento (DBC), constantes do modelo
  db.py                    banco: runs, tabelas de resultado, relatórios, gráficos
  data.py                  leitura dos dados (Falker, campo, índices) a partir do banco
  ingest.py                carga dos dados brutos e importação da análise anterior
  preprocess.py            pré-processamento das imagens
  indices.py               índices espectrais (31) e estatísticas por imagem
  modeling.py              MLP campeã + validação GroupKFold por data
  metrics.py               R², RMSE, MAE
  app_model.py             pesos da MLP embarcada no app (Dart) e manifest
pipeline/
  00_configuracao/         cria o banco e importa a análise anterior
  01_extracao/             índices por imagem (inclui a variante bilateral)
  02_clorofila_medida/     ANOVA da dose de N, gráficos, clorofila por imagem
  03_modelos_clorofila/    fórmula linear por índice (r/b) e MLP definitiva
  04_mapas/                mapas de clorofila bloco a bloco
  05_analise_vegetativa/   altura e biomassa × dose e clorofila (26/05)
  06_validacao_app/        validação do modelo embarcado no app Nutrinitro
  exploratorio/            buscas de modelos que não entraram no resultado final
data/raw/                  dados brutos tabulares (Falker, campo, manifest do app)
database/tcc.sqlite        banco local (gerado; fora do git)
outputs/                   gráficos, cópias dos relatórios, exportações (gerado)
legacy/scripts/            cópia fiel dos scripts originais, para referência
docs/                      documentação do modelo definitivo (pesos, pipeline)
site/                      site da análise: server.py (API), queries.py, imaging.py, pages/, static/
```

## Como rodar

```bash
pip install -r requirements.txt
python run.py setup        # cria o banco e importa a análise anterior
python run.py all          # pipeline principal (~6 min)
python run.py list         # todas as etapas
python run.py anova definitive_model   # etapas específicas
```

As imagens de campo não fazem parte do repositório. Por padrão, o código procura as imagens em `data/image/` e, se não achar, em `../data/image/` (a pasta do projeto original). Para outro local, defina `TCC_IMAGE_ROOT=/caminho/para/image`.

## Site da análise

```bash
python run.py site          # abre http://localhost:8000
```

Um servidor local (só biblioteca padrão do Python) entrega as páginas de `site/` e uma API JSON que consulta
o SQLite **ao vivo**, em modo somente leitura. Nenhum dado fica copiado em arquivo: rodar o pipeline de novo e
recarregar a página já mostra os resultados novos.

| Página | Conteúdo |
| :--- | :--- |
| Resumo | Vista aérea, números principais e o que os dados mostram |
| Campo | Mapa interativo dos blocos × tratamentos. Cada parcela mostra as fotos por data, as leituras Falker, as predições (rede, índice, app), o crescimento, as etapas do processamento, o mapa de clorofila e os alertas |
| Análise | 10 figuras numeradas com interpretação: dose, leituras, cor × clorofila, modelo, erro por data, abordagens, cobertura, altura, app |
| Outliers | Critérios fixos e cada caso com a justificativa calculada a partir dos diagnósticos |
| Imagens | Etapas do processamento, comparação entre duas fotos, galeria filtrável e mapas |
| Dados | Explorador de tabelas (com execuções anteriores), método, relatórios, execuções e glossário |

- As miniaturas e as etapas do processamento de cada foto são geradas sob demanda a partir das fotos
  originais, com cache em `outputs/cache/`.
- Os textos dos outliers e os números citados vêm do banco (`site/queries.py`), sem valores escritos à mão.
- A página precisa do servidor: abrir `site/index.html` direto não funciona. Rotas da API: docstring de
  `site/server.py`.

## Banco de dados

Todas as execuções ficam registradas na tabela `runs`. Cada tabela de resultado tem a coluna `run_id`: rodar um script de novo **acrescenta** linhas, não sobrescreve as anteriores. Por padrão, a leitura devolve a execução mais recente.

| Tipo | Tabelas |
| :--- | :--- |
| Dados brutos | `dbc_layout`, `falker_readings`, `campo_altura_peso`, `campo_massa_verde_seca`, `images` |
| Índices por imagem | `image_indices` (colunas `Variant` = default/bilateral e `Block_Size` = 10/5/2) |
| Resultados | `anova_n_dose`, `anova_indices`, `predictive_formulas`, `cv_lobo_results`, `cv_lodo_results`, `lmm_results`, `definitive_model_predictions`, `definitive_model_summary`, `mapping_images_summary`, `regression_summary`, … (`python run.py tables`) |
| Relatórios e gráficos | `reports` (markdown) e `artifacts` (caminho de cada gráfico gerado) |
| Análise anterior | `legacy__*` (um por CSV antigo), `reports` com nome `legacy/...` |

Comandos úteis:

```bash
python run.py runs                                   # histórico de execuções
python run.py tables                                 # tabelas e descrição
python run.py show definitive_model_summary          # ver uma tabela
python run.py export predictive_formulas --format xlsx   # gera outputs/exports/…
python run.py report vegetative_analysis_report      # imprime um relatório
```

Em Python: `from tcc_analysis import db; db.load_df("definitive_model_predictions")`. Para uma execução específica, use `run_id=...`; para todas, `run_id=None`.

## Pipeline principal × exploratório

O pipeline principal (`python run.py all`) contém apenas o que produz os resultados do TCC. Os scripts em `pipeline/exploratorio/` foram buscas de hiperparâmetros ou abordagens que não superaram o modelo final:

| Script | O que buscava | Resultado |
| :--- | :--- | :--- |
| `01_preprocess_search` | 36 variantes de pré-processamento | Escolheu a variante `bilateral`, agora fixa na extração |
| `02_ensemble_ridge_pca_mlp` | Ensemble Ridge + PCA + MLP | R² OOF negativo a 0,38 |
| `03_mlp_rgb` / `04_mlp_rgb_advanced` | Grades de arquitetura e alpha | R² OOF ≤ 0,70 |
| `05_symbolic_regression` / `06_physical_index_search` | Fórmulas de índice | Sobreajuste (treino ≈ 0,84; validação ≈ 0,45) |
| `07_mlp_custom_indices` | MLP com índices descobertos | R² OOF 0,63 |
| `08_mlp_height` | MLP prevendo altura | R² LOOCV −0,78 |
| `09_match_debug_export` | Reidentificar parcelas no JSON do app | Substituído por `06_validacao_app/05_app_vs_falker` |

Eles continuam executáveis (`python run.py exploratorio`), e os resultados da execução original estão nas tabelas `legacy__*`.

## Resultado principal

Validação cruzada deixando uma data de fora (GroupKFold por data), 34 imagens, pré-processamento
`bilateral_hsv` (bilateral + máscara HSV de sombra/estouro/reflexo, blocos 10×10, ExG > 0,15):

| Modelo | R² | RMSE (SPAD) | MAE (SPAD) |
| :--- | :---: | :---: | :---: |
| MLP 24 entradas (Median/Mean/P75/P90 × r, g, b, rg, rb, gb) | 0,616 | 2,62 | 1,99 |
| Índice único `Mean_r_over_b` (escolhido entre 117; valor otimista) | 0,648 | 2,51 | 1,99 |

Na prática os dois empatam. O antigo 0,753 era artefato (ver abaixo). Pesos e passo a passo do modelo novo
para o app: [docs/modelo_app_v2.md](docs/modelo_app_v2.md). O documento antigo
([docs/definitive_models_doc.md](docs/definitive_models_doc.md)) descreve o modelo v1.

## Análise por coleta (dentro de cada data)

`pipeline/03_modelos_clorofila/03_per_date_analysis.py` analisa cada data sozinha: correlação de cada índice com
o SPAD, q de Benjamini-Hochberg, R² de validação deixando uma parcela de fora e ANOVA em blocos. Tabelas
`per_date_*`; página **Coletas** do site.

| Data | Parcelas | Melhor índice | R² validação | Índices com q < 0,05 |
| :--- | :---: | :--- | :---: | :---: |
| 18/05 | 7 | `P90_GLI` | 0,84 | 71 de 117 |
| 21/05 | 12 | `P75_BI` | 0,38 | 47 de 117 |
| 26/05 | 12 | `P75_b_over_r` | 0,71 | 82 de 117 |
| 01/06 | 3 | (sem inferência) | – | 0 |

Dentro da data a foto acompanha bem o SPAD; entre datas o nível se desloca. Em 21/05 e 26/05 os melhores
índices separam as doses na ANOVA com F maior que o do próprio SPAD.

## Altura e biomassa pela foto (26/05, 12 parcelas)

`pipeline/05_analise_vegetativa/04_indices_vs_vegetative.py` (`python run.py vegetative_indices`, ~11 min) compara os
117 índices com altura e massas e busca uma MLP (350 configurações) com validação aninhada. Página **Altura e massa** do site.

| Variável | Melhor índice (R² LOO, escolhido olhando tudo) | Melhor abordagem honesta só com foto | MLP aninhada | SPAD Falker |
| :--- | :---: | :---: | :---: | :---: |
| Altura | 0,76 (`Mean_L`) | Ridge 24 entradas 0,68 | 0,12 | 0,83 |
| Massa verde de folha | 0,74 (`P90_r`) | reta no melhor índice 0,68 | 0,06 | 0,28 |
| Massa seca de folha | 0,68 (`P75_r`) | SPAD previsto pela foto 0,52 | 0,16 | 0,33 |
| Massa verde total | 0,69 (`P90_HI`) | reta no melhor índice 0,63 | 0,62 | 0,12 |
| Massa seca total | 0,68 (`P90_HI`) | MLP aninhada 0,54 | 0,54 | 0,19 |

A melhor MLP da busca chega a 0,64–0,73, mas com validação aninhada cai para 0,06–0,62: com 12 parcelas a busca
decora os dados. Colmo e relação folha/colmo não são previsíveis. Uma data só, com a dose variando junto.

## Classes de clorofila (baixa, média, alta)

`pipeline/03_modelos_clorofila/04_chlorophyll_classes.py` (`python run.py chl_classes`) cria as classes a partir das
próprias medições, pelas faixas de Beaufils (Beaufils, 1973; Guimarães, 2014): média (m) e desvio padrão (s) do SPAD
médio das 36 parcelas × data. Não há nível crítico fixo de SPAD para pastagens (Bacelar et al., 2015, Embrapa
Rondônia Doc. 161), por isso a calibração local. As cinco faixas de Beaufils foram agrupadas em três:

| Classe | Faixa de Beaufils | SPAD (Falker) | Parcelas |
| :--- | :--- | :---: | :---: |
| Baixa | deficiência + tendência à deficiência (< m − 2/3 s) | < 35,6 | 8 |
| Média | suficiente (m ± 2/3 s) | 35,6 – 41,0 | 17 |
| Alta | tendência ao excesso + excesso (> m + 2/3 s) | > 41,0 | 11 |

m = 38,3; s = 4,1; Shapiro-Wilk p = 0,62. A classe acompanha a dose (ρ = 0,70) e, em 26/05, a altura (ρ = 0,89) e a
massa (ρ ≈ 0,70).

**Incerteza do Falker.** As 7 leituras de uma parcela variam ±2,6 SPAD; a média tem IC 95 % de ±2,3 SPAD. Só 11 das
36 parcelas têm classe segura (o intervalo não cruza um limite). Medir a mesma parcela de novo daria a mesma classe em
80 % das vezes (teto; kappa 0,74). Com 27 leituras por parcela o IC cairia para ±1 SPAD e o teto subiria para 89 %.
Com cinco classes (faixas de 2,7 SPAD) o teto seria 73 %; a comparação 5 × 4 × 3 está em `chl_class_scheme_comparison`.

| Pela foto (reta ou rede ajustada em duas datas, testada na terceira) | Classe exata | Kappa | Dentro da incerteza |
| :--- | :---: | :---: | :---: |
| Melhor índice `Median_IPCA` (escolhido olhando o resultado) | 87 % | 0,83 | 97 % |
| Rede neural definitiva | 84 % | 0,79 | 97 % |
| MLP pequena, configuração escolhida sem a data testada | 77 % | 0,71 | 94 % |
| Índice escolhido sem a data testada | 77 % | 0,70 | 90 % |

Nenhum método troca baixa por alta. A busca da MLP está em `pipeline/exploratorio/11_mlp_class_search.py`
(`python run.py mlp_class_search`, ~3 min; rode `chl_classes` de novo depois para incluí-la). Tabelas `chl_class_*`
e `mlp_class_*`; página **Classes** do site.

## Análises complementares (`python run.py complementares`)

Scripts em `pipeline/07_analises_complementares/`, tabelas `sci_*`, página **Aprofundamento** do site.

| # | Pergunta | Resultado |
| :--- | :--- | :--- |
| 1 | A foto mede a parcela com menos ruído? (ICC, Shrout e Fleiss, 1979) | Confiabilidade da média: foto 0,97 × Falker 0,94; F da dose ~16 × ~6 (21/05 e 26/05). Recortes da mesma foto compartilham a luz: favorece um pouco a foto. |
| 2 | Foto e Falker concordam? (Bland e Altman, 1986) | Sem viés geral (≈ 0), mas limites de ±5 SPAD (Falker × Falker: ±2,7). Viés por data de −1,6 (21/05) a +1,2 (26/05); sem ele, ±4,6. |
| 3 | A foto aponta a mesma dose ótima? | Nenhuma medida atinge máximo até 100 kg N: resposta linear (AICc) no Falker e na foto, nas três datas. Em 26/05 a foto explica melhor a dose (R² 0,82 × 0,66). |
| 4 | Qual clorofila limita a produção? (Cate e Nelson, 1971; 26/05) | Nível crítico 34,7 SPAD (Falker) e 34,1–34,4 pela foto (em SPAD equivalente); coincide com o limite baixa/média das classes (35,6). Com 12 parcelas, indicativo. |
| 5 | Clorofila a × b | A razão a/b cai com o N (4,1 → 3,4); F = 12 em 26/05. A foto não separa a de b (r = 0,95 entre elas). |
| 6 | Heterogeneidade dentro da parcela | Pela foto, parcelas sem N são mais desuniformes (CV de b/r 28,5 % × 18,4 %; ρ = −0,68). O Falker (7 folhas) não detecta; parte do efeito vem da cobertura menor. |
| 7 | Dinâmica entre coletas (modelo misto) | O SPAD caiu 4–5,6 em 8 dias em todas as doses; sem interação dose × data (p = 0,14). Parcelas mais altas perderam menos: não é diluição. |
| 8 | Planejamento | 29 leituras/parcela para ±1 SPAD; 28 leituras levam o teto de classe de 80 % a 89 %. Para detectar 25 kg N: 10 blocos com a foto × 25 com o Falker. |

### Explorações (scripts 10–16, tabelas `ext_*`, página **Explorações**)

| Pergunta | Resultado |
| :--- | :--- |
| Partição folha/colmo e matéria seca mudam com o N? | Não (p ≥ 0,29): o N aumentou a produção sem mudar a proporção de folha nem o teor de matéria seca. |
| A clorofila satura antes da produção? | Ao contrário: com 50 kg a massa seca já tinha 97 % do ganho de 0 a 100 kg; o SPAD, 44 % (massa satura antes em 99 % das reamostragens). Padrão de consumo de luxo. |
| Eficiência de uso do N | 27 → 19 → 14 kg de massa seca por kg de N (50, 75, 100 kg); de 75 para 100 kg, retorno ≈ 0. |
| Uniformidade da altura | O CV das 5 alturas não muda com a dose nem acompanha as manchas de cor. |
| Cor da folha × cor da luz | A data explica 34 % da variação de cor, o SPAD 56 %; direções a ~40°. Remover a direção da luz (estimada com 2 datas) piorou o R² (0,60 → 0,43): corrigir exige mais datas ou cartão de cor. |
| O que a foto vê além da folha? | Para a massa, a foto supera o Falker (R² 0,56 × 0,47); a cobertura soma ao SPAD (R² 0,68). Para a altura, o SPAD domina (0,90). |
| Escala das manchas | Variação de cor na escala da folha (~1 bloco de 10 px). Sem N o amarelado é salpicado; com N, estruturas maiores (ρ = 0,55 com a dose). |
| Gradiente no terreno (GPS) | Nenhum: inclinações não significativas (p ≥ 0,13) e I de Moran ≈ 0. |
| Distribuição das leituras | Simétrica (assimetria 0,03); 4 % discrepantes; a média varia menos que a mediana (±1,9 × ±2,2 com 7 leituras). |
| Ordem e horário | O Falker não deriva; nas fotos o brilho muda em 2–3 min (ρ = −0,57), mas os índices cromáticos não. |

## Site estático (GitHub Pages)

`python run.py build` gera em `dist/` uma versão do site que não precisa de servidor nem API: o próprio servidor é
iniciado internamente e cada rota é gravada como JSON; tabelas (paginação, busca, ordenação), CSV e ZIP funcionam no
navegador. Sem as fotos de campo o site tem ~14 MB; `--with-images` inclui fotos e etapas do processamento (~170 MB).

Para testar localmente: `python -m http.server -d dist`. Para publicar:

1. Uma vez só: no GitHub, *Settings → Pages → Build and deployment → Source: GitHub Actions*.
2. `python run.py build`, depois `git add dist && git commit && git push`.

O workflow `.github/workflows/deploy-pages.yml` publica `dist/` a cada push que altera essa pasta (ou manualmente,
em *Actions → Deploy do site → Run workflow*). O site fica em `https://antfconeto.github.io/tcc_n_predict_analisys/`.

## Achado: o R² 0,753 depende de um resíduo numérico

O canal `rgb` = (r+g+b)/3 vale sempre 1/3. Mediana, P75 e P90 dessa coluna saem exatamente constantes, mas
`Mean_rgb` carrega um resíduo de arredondamento (desvio de 2×10⁻⁸). O `StandardScaler` divide por esse desvio
e transforma o ruído em uma entrada grande. Essa é a entrada de maior peso da rede (−0,76 no primeiro neurônio).

| Entradas da MLP campeã (mesma rede, mesma validação por data) | R² |
| :--- | :---: |
| 28 entradas do banco (modelo definitivo) | 0,753 |
| `*_rgb` fixados em exatamente 1/3 | 0,594 |
| sem as 4 colunas `*_rgb` (24 entradas) | 0,589 |
| `Mean_rgb` = 1/3 + ruído aleatório do mesmo tamanho | 0,52–0,58 |

O desempenho reproduzível da MLP v1 é de cerca de **0,59**. O modelo foi retreinado sem os canais `*_rgb` e com a máscara HSV (v2, 0,616). O resíduo
muda com a biblioteca, a plataforma e a ordem das contas, então o 0,753 não se reproduz no app (que usa esses
mesmos pesos em Dart). Detalhes: `pipeline/exploratorio/10_illumination_study.py` e tabela
`illumination_study_summary`.

## Observações sobre a migração

- **Índices da análise anterior.** Os CSVs antigos `extracted_indices_{10x10,5x5,2x2}.csv` foram gerados com uma configuração de pré-processamento anterior à do código atual. Por isso ANOVA, fórmulas por índice, LOBO/LODO e modelo misto dão valores diferentes das tabelas `legacy__*`. O código original, rodado hoje, reproduz exatamente os índices novos.
- **Resultados que batem com os antigos.** A variante bilateral, a análise vegetativa (11), a altura × biomassa (13) e a validação do pipeline alinhado batem com os resultados antigos.
- **Relatórios com números fixos no texto.** Os relatórios da análise vegetativa têm parágrafos de discussão com números escritos à mão no código, que nem sempre batem com as tabelas geradas. Revise antes de usar no texto do TCC.
