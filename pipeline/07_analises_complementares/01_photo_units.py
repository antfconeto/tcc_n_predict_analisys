"""
Índices por foto inteira e por recortes 3×3, e a variação entre blocos de cada foto.

Base das análises complementares: os recortes funcionam como repetições dentro da parcela (como as folhas do
Falker) e a dispersão entre blocos (DP, IQR e CV de cada índice) mede a heterogeneidade do dossel.
A linha Recorte = −1 (foto inteira) é idêntica à tabela image_indices da variante definitiva.

Tabela: sci_photo_units.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import db
from tcc_analysis.tiles import extract_photo_units


def main():
    df = extract_photo_units(tiles=3)
    db.save_df(df, "sci_photo_units", description="Índices por foto inteira (Recorte −1) e por recortes 3×3, com a heterogeneidade entre blocos")
    print(f"{(df.Recorte == -1).sum()} fotos, {(df.Recorte >= 0).sum()} recortes, {df.shape[1]} colunas")


if __name__ == "__main__":
    db.run_main(main)
