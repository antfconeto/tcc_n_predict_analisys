#!/usr/bin/env python3
"""Executor do pipeline e utilitários do banco.

Exemplos:
  python run.py list                      # etapas disponíveis
  python run.py setup                     # cria o banco + importa a análise anterior
  python run.py extract_indices anova     # roda etapas específicas (em ordem)
  python run.py all                       # pipeline principal
  python run.py exploratorio              # buscas de modelos (demorado, opcional)
  python run.py preprocess_search -- bilateral   # repassa argumentos ao script
  python run.py runs                      # histórico de execuções
  python run.py tables                    # tabelas e última execução
  python run.py show definitive_model_summary
  python run.py export definitive_model_predictions --format xlsx
  python run.py report vegetative_analysis_report
  python run.py site [--port 8000]        # site da análise (consulta o banco ao vivo)
  python run.py build [--out dist]        # site estático (sem API) para o GitHub Pages
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _ensure_dependencies() -> None:
    """Se o Python atual não tem as dependências, reexecuta com o .venv do projeto."""
    try:
        import numpy, pandas, sklearn  # noqa: F401
        return
    except ImportError as exc:
        missing = exc.name
    import os
    for venv in (ROOT / ".venv", ROOT.parent / ".venv"):
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if python.exists() and Path(sys.executable).resolve() != python.resolve():
            print(f"[run.py] '{missing}' não está instalado em {sys.executable}; usando {python}", file=sys.stderr)
            os.execv(str(python), [str(python), *sys.argv])
    raise SystemExit(
        f"Dependência ausente: {missing}. Instale com `pip install -r requirements.txt` "
        "ou crie um .venv no repositório."
    )


_ensure_dependencies()

# (alias, script, descrição) — na ordem de dependência
STEPS = [
    ("init_db", "00_configuracao/01_init_db.py", "Cria o banco e carrega dados brutos"),
    ("import_legacy", "00_configuracao/02_import_legacy.py", "Importa resultados da análise anterior (legacy__*)"),
    ("extract_indices", "01_extracao/01_extract_indices.py", "Índices por imagem (10x10, 5x5, 2x2 + variante bilateral)"),
    ("anova", "02_clorofila_medida/01_anova.py", "ANOVA dose de N + regressão índice × clorofila"),
    ("plot_chlorophyll", "02_clorofila_medida/02_plot_chlorophyll.py", "Gráficos da clorofila medida"),
    ("measured_by_image", "02_clorofila_medida/03_measured_by_image.py", "Clorofila Falker associada a cada imagem"),
    ("index_regression", "03_modelos_clorofila/01_index_regression.py", "Fórmulas lineares por índice + LOBO/LODO/LMM"),
    ("definitive_model", "03_modelos_clorofila/02_definitive_model.py", "MLP v2 (24 entradas, máscara HSV) × melhor índice"),
    ("per_date", "03_modelos_clorofila/03_per_date_analysis.py", "Análise dentro de cada coleta (sem comparar datas)"),
    ("chl_classes", "03_modelos_clorofila/04_chlorophyll_classes.py", "Classes de clorofila pelas faixas de Beaufils sobre o SPAD medido"),
    ("prediction_maps", "04_mapas/01_prediction_maps.py", "Mapas de clorofila bloco a bloco"),
    ("vegetative", "05_analise_vegetativa/01_vegetative_analysis.py", "Altura/biomassa × dose e clorofila medida"),
    ("predicted_chlorophyll", "05_analise_vegetativa/02_predicted_chlorophyll_analysis.py", "Altura/biomassa × clorofila predita"),
    ("height_biomass", "05_analise_vegetativa/03_height_biomass.py", "Altura × biomassa"),
    ("vegetative_indices", "05_analise_vegetativa/04_indices_vs_vegetative.py", "Índices × altura/biomassa e busca de MLP (26/05)"),
    ("prepare_app_dataset", "06_validacao_app/01_prepare_cropped_dataset.py", "Imagens recortadas + manifest para o app"),
    ("validate_mlp_dart", "06_validacao_app/02_validate_mlp_dart.py", "Pesos Dart × sklearn × OOF"),
    ("validate_aligned", "06_validacao_app/03_validate_aligned_pipeline.py", "Pipeline alinhado (com e sem crop)"),
    ("validate_stages", "06_validacao_app/04_validate_pipeline_stages.py", "Validação etapa a etapa"),
    ("app_vs_falker", "06_validacao_app/05_app_vs_falker.py", "Predição do app × Falker (JSON de debug)"),
    # Exploratórios: buscas de hiperparâmetros/alternativas que não entraram no modelo final.
    # Não fazem parte de `all`; os resultados da execução original estão em legacy__*.
    ("preprocess_search", "exploratorio/01_preprocess_search.py", "[exp] Busca de pré-processamento (36 variantes)"),
    ("ensemble", "exploratorio/02_ensemble_ridge_pca_mlp.py", "[exp] Ensemble Ridge + PCA + MLP (R² negativo)"),
    ("mlp_rgb", "exploratorio/03_mlp_rgb.py", "[exp] Grid de MLP só RGB"),
    ("mlp_rgb_advanced", "exploratorio/04_mlp_rgb_advanced.py", "[exp] MLP RGB com PCA/bagging/ativações"),
    ("symbolic_regression", "exploratorio/05_symbolic_regression.py", "[exp] Regressão simbólica (gplearn)"),
    ("physical_search", "exploratorio/06_physical_index_search.py", "[exp] Busca de fórmulas físicas"),
    ("mlp_custom_indices", "exploratorio/07_mlp_custom_indices.py", "[exp] MLP com índices customizados"),
    ("mlp_height", "exploratorio/08_mlp_height.py", "[exp] Grid de MLP prevendo altura (R² negativo)"),
    ("match_debug_export", "exploratorio/09_match_debug_export.py", "[exp] Reidentifica parcelas no JSON do app (v1)"),
    ("illumination_study", "exploratorio/10_illumination_study.py", "[exp] Correções de luz/cor e o artefato de Mean_rgb"),
    ("mlp_class_search", "exploratorio/11_mlp_class_search.py", "[exp] Busca de MLP para as classes de clorofila (aninhada)"),
]
EXPLORATORY = [alias for alias, script, _ in STEPS if script.startswith("exploratorio/")]
GROUPS = {
    "setup": ["init_db", "import_legacy"],
    "all": [alias for alias, _, _ in STEPS if alias != "import_legacy" and alias not in EXPLORATORY],
    "exploratorio": EXPLORATORY,
}


def cmd_list() -> None:
    print(f"{'etapa':<24} script")
    for alias, script, desc in STEPS:
        print(f"  {alias:<22} pipeline/{script}\n  {'':<22} {desc}")
    print("\nGrupos:", ", ".join(f"{k} ({len(v)} etapas)" for k, v in GROUPS.items()))


def resolve(names: list[str]) -> list[tuple[str, str]]:
    by_alias = {alias: script for alias, script, _ in STEPS}
    selected: list[str] = []
    for name in names:
        if name in GROUPS:
            selected.extend(GROUPS[name])
        elif name in by_alias:
            selected.append(name)
        else:
            raise SystemExit(f"Etapa desconhecida: {name}. Use `python run.py list`.")
    order = [alias for alias, _, _ in STEPS]
    unique = sorted(set(selected), key=order.index)
    return [(alias, by_alias[alias]) for alias in unique]


def cmd_run(names: list[str], passthrough: list[str]) -> None:
    steps = resolve(names)
    for alias, script in steps:
        print("\n" + "#" * 80 + f"\n# {alias}  (pipeline/{script})\n" + "#" * 80, flush=True)
        t0 = time.time()
        cmd = [sys.executable, str(ROOT / "pipeline" / script)]
        if len(steps) == 1:
            cmd += passthrough
        result = subprocess.run(cmd, cwd=ROOT)
        print(f"\n[{alias}] terminou em {time.time() - t0:.1f}s (código {result.returncode})")
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def cmd_runs() -> None:
    from tcc_analysis import db
    import pandas as pd
    pd.set_option("display.width", 200)
    print(db.query(
        "SELECT run_id, script, status, started_at, finished_at FROM runs ORDER BY run_id DESC LIMIT 50"
    ).to_string(index=False))


def cmd_tables() -> None:
    from tcc_analysis import db
    import pandas as pd
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    print(db.query(
        "SELECT table_name, last_run_id, updated_at, description FROM catalog "
        "ORDER BY table_name LIKE 'legacy%', table_name"
    ).to_string(index=False))


def cmd_show(table: str, extra: list[str]) -> None:
    from tcc_analysis import db
    import pandas as pd
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    df = db.load_df(table)
    n = int(extra[0]) if extra else 20
    print(f"{table}: {len(df)} linhas × {df.shape[1]} colunas (última execução)\n")
    print(df.head(n).to_string(index=False))


def cmd_export(table: str, extra: list[str]) -> None:
    from tcc_analysis import config, db
    fmt = extra[extra.index("--format") + 1] if "--format" in extra else "csv"
    df = db.load_df(table)
    config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.EXPORTS_DIR / f"{table}.{fmt}"
    if fmt == "csv":
        df.to_csv(out, index=False)
    elif fmt == "xlsx":
        df.to_excel(out, index=False)
    else:
        raise SystemExit("Formatos: csv, xlsx")
    print(f"{len(df)} linhas exportadas para {out}")


def cmd_report(name: str) -> None:
    from tcc_analysis import db
    print(db.load_report(name))


def main() -> None:
    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "list":
        cmd_list()
    elif cmd == "runs":
        cmd_runs()
    elif cmd == "tables":
        cmd_tables()
    elif cmd == "show" and rest:
        cmd_show(rest[0], rest[1:])
    elif cmd == "export" and rest:
        cmd_export(rest[0], rest[1:])
    elif cmd == "report" and rest:
        cmd_report(rest[0])
    elif cmd == "site":
        subprocess.run([sys.executable, str(ROOT / "site" / "server.py"), *rest, *passthrough], cwd=ROOT)
    elif cmd == "build":
        subprocess.run([sys.executable, str(ROOT / "site" / "build_static.py"), *rest, *passthrough], cwd=ROOT)
    else:
        cmd_run(argv, passthrough)


if __name__ == "__main__":
    main()
