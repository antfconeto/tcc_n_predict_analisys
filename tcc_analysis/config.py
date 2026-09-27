"""Configuração central: caminhos, delineamento experimental e constantes compartilhadas."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Dados brutos tabulares (versionados no repositório)
RAW_DIR = REPO_ROOT / "data" / "raw"
FALKER_FILES = {
    "cloro": RAW_DIR / "clorofila" / "cloro.csv",
    "test": RAW_DIR / "clorofila" / "test.csv",
}
HEIGHT_WEIGHT_CSV = RAW_DIR / "campo" / "dados-altura-peso.csv"
BIOMASS_CSV = RAW_DIR / "campo" / "massa_verde_seca_compilada.csv"
APP_MANIFEST_JSON = RAW_DIR / "validacao_app" / "manifest.json"
APP_DEBUG_EXPORT_JSON = RAW_DIR / "validacao_app" / "app_debug_export.json"


def _first_existing(*candidates):
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return Path(candidates[-1])


# Imagens (~200 MB) ficam fora do git. Por padrão usa ../data/image (projeto original).
IMAGE_ROOT = _first_existing(
    os.environ.get("TCC_IMAGE_ROOT"),
    REPO_ROOT / "data" / "image",
    REPO_ROOT.parent / "data" / "image",
)

# Projeto original (análise anterior), usado apenas para importação legada.
LEGACY_ROOT = Path(os.environ.get("TCC_LEGACY_ROOT", REPO_ROOT.parent))

# Banco de dados local e saídas geradas
DB_PATH = Path(os.environ.get("TCC_DB_PATH", REPO_ROOT / "database" / "tcc.sqlite"))
OUTPUT_DIR = REPO_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
EXPORTS_DIR = OUTPUT_DIR / "exports"

# Datas de avaliação (pastas de imagens) e ano do experimento
YEAR = 2026
DATE_FOLDERS = ["18-05", "21-05", "26-05", "01-06"]
EXPERIMENTAL_DATES = ["18-05-2026", "21-05-2026", "26-05-2026"]
TEST_DATE = "01-06-2026"

# Delineamento em blocos casualizados (DBC): 3 blocos × 4 doses de N
PONTO_TO_DBC = {
    1:  {"Bloco": "Bloco 1", "Tratamento": "T3", "Dose": 75},
    2:  {"Bloco": "Bloco 1", "Tratamento": "T2", "Dose": 50},
    3:  {"Bloco": "Bloco 1", "Tratamento": "T4", "Dose": 100},
    4:  {"Bloco": "Bloco 1", "Tratamento": "T1", "Dose": 0},
    5:  {"Bloco": "Bloco 2", "Tratamento": "T4", "Dose": 100},
    6:  {"Bloco": "Bloco 2", "Tratamento": "T3", "Dose": 75},
    7:  {"Bloco": "Bloco 2", "Tratamento": "T1", "Dose": 0},
    8:  {"Bloco": "Bloco 2", "Tratamento": "T2", "Dose": 50},
    9:  {"Bloco": "Bloco 3", "Tratamento": "T2", "Dose": 50},
    10: {"Bloco": "Bloco 3", "Tratamento": "T4", "Dose": 100},
    11: {"Bloco": "Bloco 3", "Tratamento": "T3", "Dose": 75},
    12: {"Bloco": "Bloco 3", "Tratamento": "T1", "Dose": 0},
}

# Modelo definitivo (v2): filtro bilateral + máscara HSV (sombra, reflexo) e 24 entradas.
# As colunas *_rgb foram removidas: (r+g+b)/3 vale sempre 1/3, e o resíduo de arredondamento de
# Mean_rgb inflava o R² da versão anterior (0,753 → ~0,59 sem o artefato). Ver README.
BEST_PREPROCESS_VARIANT = "bilateral_hsv"
BEST_PREPROCESS_CONFIG = {"filter_type": "bilateral", "hsv_enabled": True, "remove_specular": True}
INDEX_STATS = ["Median", "Mean", "P75", "P90"]
MODEL_CHANNELS = ["r", "g", "b", "rg", "rb", "gb"]
MODEL_FEATURES = [f"{stat}_{ch}" for stat in INDEX_STATS for ch in MODEL_CHANNELS]
CHAMP_MLP_PARAMS = {"hidden_layer_sizes": (3,), "alpha": 8.0, "activation": "relu"}

# Versão 1 do app (pesos embarcados hoje): bilateral sem máscara HSV, 28 entradas com *_rgb.
# Mantida só para as validações do app em pipeline/06_validacao_app.
APP_V1_VARIANT = "bilateral"
APP_V1_PREPROCESS_CONFIG = {"filter_type": "bilateral"}
RGB_CHANNELS = ["r", "g", "b", "rg", "rb", "gb", "rgb"]
RGB28_FEATURES = [f"{stat}_{ch}" for stat in INDEX_STATS for ch in RGB_CHANNELS]
