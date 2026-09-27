"""
Cria o banco SQLite e carrega os dados brutos.

- dbc_layout        ← delineamento (config.PONTO_TO_DBC)
- falker_readings   ← data/raw/clorofila/{cloro,test}.csv
- campo_altura_peso ← data/raw/campo/dados-altura-peso.csv
- campo_massa_verde_seca ← data/raw/campo/massa_verde_seca_compilada.csv
- images            ← catálogo das fotos em config.IMAGE_ROOT (arquivos ficam em disco)

Pode ser executado novamente: as tabelas de dados brutos são recarregadas, os
resultados de análises anteriores são mantidos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.ingest import ingest_raw


def main():
    print(f"Banco: {config.DB_PATH}")
    print(f"Imagens: {config.IMAGE_ROOT}")
    ingest_raw()
    print("\nTabelas:", ", ".join(db.list_tables()))


if __name__ == "__main__":
    db.run_main(main)
