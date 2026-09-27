#!/usr/bin/env python3
"""
Prepara imagens já recortadas (crop 20% central) para validação no app Nutrinitro.

O app não aplica crop no pipeline; ao importar estas imagens no celular, a análise
equivale ao pipeline TCC com crop 20%.

Saídas:
  data/export_app_validation/cropped/{data}/P{n}.jpg
  data/export_app_validation/manifest.json
  data/export_app_validation/README.txt
  data/export_app_validation/nutrinitro_validation.zip  (opcional)
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s09", ROOT / "09-definitive-nn-model.py")
s09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s09)

DATES = ["18-05", "21-05", "26-05", "01-06"]
IMAGE_ROOT = ROOT / "data/image"
OUT_ROOT = ROOT / "data/export_app_validation"
CROPPED_DIR = OUT_ROOT / "cropped"
CROP_FRACTION = 0.20
JPEG_QUALITY = 92


def crop_center(img: np.ndarray, fraction: float = CROP_FRACTION) -> np.ndarray:
    h, w = img.shape[:2]
    y0, y1 = int(fraction * h), int((1.0 - fraction) * h)
    x0, x1 = int(fraction * w), int((1.0 - fraction) * w)
    return img[y0:y1, x0:x1].copy()


def load_measured_lookup() -> dict[tuple[str, int], dict]:
    df_cloro = s09.load_falker_csv(str(ROOT / "data/clorophyll/cloro.csv"))
    df_test = s09.load_falker_csv(str(ROOT / "data/clorophyll/test.csv"))
    df = __import__("pandas").concat([df_cloro, df_test], ignore_index=True)
    df = df.dropna(subset=["Clorofila Total", "Ponto"]).copy()

    lookup: dict[tuple[str, int], dict] = {}
    for _, row in df.iterrows():
        date_key = str(row["Data"])
        ponto = int(row["Ponto"])
        key = (date_key, ponto)
        if key not in lookup:
            lookup[key] = {
                "falker_spad_mean": float(row["Clorofila Total"]),
                "n_measurements": 1,
            }
        else:
            lookup[key]["n_measurements"] += 1
            lookup[key]["falker_spad_mean"] = float(
                df[(df["Data"] == date_key) & (df["Ponto"] == ponto)]["Clorofila Total"].mean()
            )
    return lookup


def main() -> None:
    print("=" * 72)
    print("PREPARANDO DATASET CROPPADO PARA VALIDAÇÃO NO APP")
    print("=" * 72)

    if CROPPED_DIR.exists():
        shutil.rmtree(CROPPED_DIR)
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)

    falker = load_measured_lookup()
    entries: list[dict] = []
    skipped = 0

    for folder in DATES:
        src_dir = IMAGE_ROOT / folder
        if not src_dir.is_dir():
            print(f"  [pulando] pasta ausente: {src_dir}")
            continue

        date_key = f"{folder}-2026"
        out_date_dir = CROPPED_DIR / folder
        out_date_dir.mkdir(parents=True, exist_ok=True)

        for filename in sorted(os.listdir(src_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            m = re.match(r"^[Pp](\d+)", filename)
            if not m:
                skipped += 1
                continue

            ponto = int(m.group(1))
            src_path = src_dir / filename
            out_name = f"{folder}_P{ponto:02d}.jpg"
            out_path = out_date_dir / out_name

            img = cv2.imread(str(src_path))
            if img is None:
                print(f"  [erro] não leu {src_path}")
                continue

            h, w = img.shape[:2]
            cropped = crop_center(img)
            ch, cw = cropped.shape[:2]

            cv2.imwrite(
                str(out_path),
                cropped,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            ref = falker.get((date_key, ponto), {})
            dbc = s09.ponto_to_dbc.get(ponto, {})
            entries.append({
                "id": f"{folder}_P{ponto:02d}",
                "source_file": str(src_path.relative_to(ROOT)),
                "cropped_file": str(out_path.relative_to(ROOT)),
                "date_folder": folder,
                "date_key": date_key,
                "ponto": ponto,
                "bloco": dbc.get("Bloco"),
                "tratamento": dbc.get("Tratamento"),
                "dose_kg_ha": dbc.get("Dose"),
                "falker_spad_reference": ref.get("falker_spad_mean"),
                "crop_fraction": CROP_FRACTION,
                "original_size": {"width": w, "height": h},
                "cropped_size": {"width": cw, "height": ch},
                "suggested_app_title": f"Validação {folder} P{ponto}",
            })

    entries.sort(key=lambda e: (e["date_folder"], e["ponto"]))

    manifest = {
        "dataset": "nutrinitro_validation_cropped",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "crop_fraction": CROP_FRACTION,
        "pipeline_note": (
            "Imagens pré-recortadas (20% bordas). O app Nutrinitro NÃO aplica crop; "
            "use estas fotos como entrada para equivaler ao treino TCC."
        ),
        "analysis_recommendation": {
            "crop": "Capim Marandu",
            "analysis_type": "nitrogen_mlp",
            "block_size": 10,
        },
        "image_count": len(entries),
        "images": entries,
    }

    manifest_path = OUT_ROOT / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    readme = OUT_ROOT / "README.txt"
    readme.write_text(
        """Dataset de validação Nutrinitro (imagens croppadas 20%)

COMO TRANSFERIR PARA O CELULAR
------------------------------
1. Copie a pasta 'cropped/' para o celular (USB, Drive, adb push, etc.)
2. No app: crie análise de Capim Marandu
3. Importe as imagens da galeria (pasta 18-05, 21-05, 26-05, 01-06)
4. Rode análise: Rede Neural MLP Campeã (nitrogen_mlp)
5. Após concluir: botão DEBUG (ícone de bug) → Copiar JSON
6. Envie o JSON para avaliação de acurácia

ADB (exemplo):
  adb push data/export_app_validation/cropped /sdcard/Download/nutrinitro_validation/

ARQUIVOS
--------
  cropped/{data}/P01.jpg ...  — imagens prontas
  manifest.json               — referência Falker SPAD por parcela
  nutrinitro_validation.zip   — pacote completo para download

REFERÊNCIA
----------
  falker_spad_reference em manifest.json = média Falker (SPAD) por Data+Ponto
""",
        encoding="utf-8",
    )

    zip_path = OUT_ROOT / "nutrinitro_validation.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, "manifest.json")
        zf.write(readme, "README.txt")
        for entry in entries:
            cropped = ROOT / entry["cropped_file"]
            arcname = f"cropped/{entry['date_folder']}/{cropped.name}"
            zf.write(cropped, arcname)

    print(f"\nImagens exportadas: {len(entries)}")
    print(f"Puladas (sem padrão Pn): {skipped}")
    print(f"Manifest: {manifest_path}")
    print(f"ZIP: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("\nPronto para transferir ao celular.")


if __name__ == "__main__":
    main()
