"""
Extrai os índices espectrais de todas as imagens (grades 10×10, 5×5 e 2×2).

Pré-processamento padrão (tcc_analysis/preprocess.py): filtro gaussiano, gamma 0.8,
crop de 20% e máscara de vegetação ExG > 0.15. Para cada índice são gravadas a mediana,
a média, o P75 e o P90 dos blocos vegetados.

Também extrai, na grade 10×10:
  - a variante do modelo definitivo (filtro bilateral + máscara HSV de sombra/reflexo);
  - a variante da versão 1 do app (só filtro bilateral), usada nas validações do app.

Saída: tabela `image_indices`
  Variant='default'        Block_Size = 10 | 5 | 2
  Variant='bilateral_hsv'  Block_Size = 10   (modelo definitivo)
  Variant='bilateral'      Block_Size = 10   (app v1)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config, db
from tcc_analysis.data import save_indices
from tcc_analysis.indices import extract_features_from_images

BLOCK_SIZES = [10, 5, 2]


def main():
    for bs in BLOCK_SIZES:
        print("\n==================================================")
        print(f"RUNNING ANALYSIS FOR BLOCK SIZE {bs}x{bs}")
        print("==================================================")
        df_indices = extract_features_from_images(block_size=bs)

        if df_indices.empty:
            print(f"Warning: No data was extracted for block size {bs}x{bs}.")
            continue

        save_indices(df_indices, block_size=bs, variant="default")
        print(f"\nSuccessfully extracted indices for {len(df_indices)} samples using block size {bs}x{bs}.")
        print("Saved to table image_indices (Variant='default')")
        print("\nFirst 3 rows of extracted data:")
        print(df_indices.head(3).to_string())

    for variant, cfg in ((config.BEST_PREPROCESS_VARIANT, config.BEST_PREPROCESS_CONFIG),
                         (config.APP_V1_VARIANT, config.APP_V1_PREPROCESS_CONFIG)):
        print("\n==================================================")
        print(f"VARIANTE {variant} (10x10)")
        print("==================================================")
        df_var = extract_features_from_images(block_size=10, preprocess_config=cfg)
        save_indices(df_var, block_size=10, variant=variant)
        print(f"Saved {len(df_var)} samples to table image_indices (Variant='{variant}')")


if __name__ == "__main__":
    db.run_main(main)
