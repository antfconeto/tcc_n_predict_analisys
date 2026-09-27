"""
Importa a análise anterior (projeto original) para o banco, sem alterar os arquivos de origem.

- cada CSV de <legado>/data/output/**   → tabela legacy__<nome>
- relatórios .md/.txt                  → tabela reports (name = legacy/...)
- .xlsx e gráficos .png                → tabela artifacts (apenas o caminho)
- manifest.json do dataset do app      → legacy__app_validation_manifest
- docs/*.md                            → tabela reports

Todas as linhas ficam associadas a um run com script = 'legacy_import'.
Uso: python pipeline/00_configuracao/02_import_legacy.py [--force] [caminho_do_projeto_antigo]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcc_analysis import config
from tcc_analysis.ingest import drop_legacy, import_legacy, legacy_imported


def main():
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv[1:]
    legacy_root = Path(args[0]) if args else config.LEGACY_ROOT

    if legacy_imported():
        if not force:
            print("A análise anterior já foi importada. Use --force para reimportar.")
            return
        print("Removendo importação legada anterior...")
        drop_legacy()

    print(f"Importando análise anterior de: {legacy_root}")
    counts = import_legacy(legacy_root)
    print(f"\nTabelas: {counts['tables']}  |  Relatórios: {counts['reports']}  |  Arquivos referenciados: {counts['artifacts']}")


if __name__ == "__main__":
    main()
