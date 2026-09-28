"""
Gera uma versão estática do site (sem servidor nem API), pronta para o GitHub Pages.

    python run.py build [--out dist] [--with-images]

O próprio servidor do site é iniciado numa porta livre e cada rota da API é gravada como JSON, então a versão
estática mostra exatamente o que o site ao vivo mostra no momento do build. Conteúdo gerado em <out>/:

    index.html, pages/, static/      a interface (com window.TCC_STATIC = true)
    data/api/*.json                  respostas da API (overview, field, plot/<id>, classes, …)
    data/tables/<tabela>.json        cada tabela do banco com todas as execuções; paginação, busca, ordenação e
                                     CSV/ZIP são feitos no navegador
    data/photos/{sm,lg}/…            só com --with-images: fotos em 640 e 1280 px (JPEG qualidade 80, ~40 MB)
    data/steps/<etapa>/{sm,lg}/…     só com --with-images: etapas do pré-processamento (~115 MB)
    data/sem-foto.svg                aviso mostrado no lugar das fotos quando elas ficam de fora
    data/plots/…                     os gráficos de outputs/plots citados pela API, convertidos para JPEG
    .nojekyll                        o GitHub Pages serve as pastas como estão

Para publicar: copie o conteúdo de <out>/ para a branch/pasta do GitHub Pages (por exemplo, docs/ ou gh-pages).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SITE_DIR.parent))
sys.path.insert(0, str(SITE_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import imaging  # noqa: E402
import queries  # noqa: E402
import server  # noqa: E402
from tcc_analysis import config  # noqa: E402

SIZES = {"sm": 640, "lg": 1280}
JPEG_QUALITY = 80          # só no site estático, para caber bem no GitHub Pages
SIMPLE_ROUTES = ["overview", "tables", "field", "outliers", "figures", "photos", "per_date", "vegetativo", "classes", "ciencia", "exploracoes",
                 "maps", "reports", "runs", "downloads"]


def safe_name(name: str) -> str:
    """Mesmo critério do app.js (safeName): só letras, dígitos, ponto, hífen e sublinhado."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def rewrite_paths(obj):
    """Caminhos absolutos do servidor (/plots/…) viram relativos à raiz do site estático."""
    if isinstance(obj, dict):
        return {k: rewrite_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_paths(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("/plots/"):
        return "data/plots/" + re.sub(r"\.png$", ".jpg", obj[len("/plots/"):], flags=re.I)
    return obj


def recompress(data: bytes) -> bytes:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    return buf.tobytes() if ok and len(buf) < len(data) else data


def collect_plot_files(obj, out: set):
    if isinstance(obj, dict):
        for v in obj.values():
            collect_plot_files(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_plot_files(v, out)
    elif isinstance(obj, str) and obj.startswith("/plots/"):
        out.add(obj[len("/plots/"):])


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def render_image(job):
    kind, rel, step, size = job
    width = SIZES[size]
    data = imaging.thumbnail(rel, width) if kind == "photo" else imaging.step_image(rel, step, width)
    return job, recompress(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(SITE_DIR.parent / "dist"), help="pasta de saída (padrão: dist/)")
    parser.add_argument("--with-images", action="store_true",
                        help="inclui as fotos de campo e as etapas do pré-processamento (~155 MB a mais)")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    t0 = time.time()

    if out.exists():
        shutil.rmtree(out)
    data = out / "data"

    # 1. interface
    shutil.copytree(SITE_DIR / "static", out / "static")
    shutil.copytree(SITE_DIR / "pages", out / "pages")
    index = (SITE_DIR / "index.html").read_text(encoding="utf-8")
    index = index.replace('<script src="static/app.js"></script>',
                          f'<script>window.TCC_STATIC = true; window.TCC_STATIC_PHOTOS = {str(args.with_images).lower()};</script>\n'
                          '<script src="static/app.js"></script>')
    (out / "index.html").write_text(index, encoding="utf-8")
    (out / ".nojekyll").write_text("")

    # 2. API: o servidor real, numa porta livre
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/api/"

    def fetch(route):
        with urllib.request.urlopen(base + route, timeout=600) as r:
            return json.load(r)

    plot_files: set[str] = set()

    def save(route, payload, file_rel):
        collect_plot_files(payload, plot_files)
        write_json(data / "api" / file_rel, rewrite_paths(payload))

    for route in SIMPLE_ROUTES:
        save(route, fetch(route), f"{route}.json")
    field = fetch("field")
    plot_ids = [p["id"] for p in field["plots"]] + [t["id"] for t in field.get("tests", [])]
    for pid in plot_ids:
        save(f"plot/{pid}", fetch(f"plot/{urllib.parse.quote(str(pid))}"), f"plot/{pid}.json")
    for rep in fetch("reports"):
        save("report", fetch("report?" + urllib.parse.urlencode({"name": rep["name"]})), f"report/{safe_name(rep['name'])}.json")
    httpd.shutdown()
    print(f"API: {len(SIMPLE_ROUTES)} rotas, {len(plot_ids)} parcelas  [{time.time() - t0:.0f}s]")

    # 3. tabelas completas (todas as execuções), para o motor de tabelas do navegador
    n_tab = 0
    with sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True) as conn:
        for name in sorted(server.valid_tables(conn)):
            cols = server.table_columns(conn, name)
            rows = [[server._json_value(v) for v in r] for r in conn.execute(f'SELECT * FROM "{name}"')]
            has_run = "run_id" in cols and name not in queries.RAW_TABLES
            write_json(data / "tables" / f"{safe_name(name)}.json",
                       server.sanitize({"table": name, "columns": cols, "rows": rows, "has_run": has_run}))
            n_tab += 1
    for name in queries.VIRTUAL_TABLES:
        df = queries.virtual_table(name).astype(object)
        df = df.where(df.notna(), None)
        rows = [[server._json_value(v.item() if hasattr(v, "item") else v) for v in r] for r in df.itertuples(index=False)]
        write_json(data / "tables" / f"{safe_name(name)}.json",
                   server.sanitize({"table": name, "columns": list(df.columns), "rows": rows, "has_run": False}))
        n_tab += 1
    print(f"Tabelas: {n_tab}  [{time.time() - t0:.0f}s]")

    # 4. gráficos citados pela API
    for rel in sorted(plot_files):
        src = config.PLOTS_DIR / rel
        if src.is_file():
            dst = data / "plots" / re.sub(r"\.png$", ".jpg", rel, flags=re.I)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() == ".png":
                img = cv2.imread(str(src), cv2.IMREAD_COLOR)
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            else:
                shutil.copy2(src, dst)
    print(f"Gráficos: {len(plot_files)}  [{time.time() - t0:.0f}s]")

    # 5. fotos e etapas do pré-processamento
    (data / "sem-foto.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420">'
        '<rect width="640" height="420" fill="#8a8a85" fill-opacity="0.12"/>'
        '<text x="320" y="200" text-anchor="middle" font-family="sans-serif" font-size="22" fill="#8a8a85">Foto não incluída</text>'
        '<text x="320" y="232" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#8a8a85">na versão estática do site</text></svg>',
        encoding="utf-8")
    if args.with_images:
        root = Path(config.IMAGE_ROOT)
        rels = sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*")
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        jobs = [("photo", rel, None, size) for rel in rels for size in SIZES]
        jobs += [("step", rel, step, size) for rel in rels for step in imaging.STEPS for size in SIZES]
        done = 0
        with ProcessPoolExecutor() as pool:
            for (kind, rel, step, size), img in pool.map(render_image, jobs, chunksize=2):
                dst = data / ("photos" if kind == "photo" else f"steps/{step}") / size / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(img)
                done += 1
                if done % 50 == 0:
                    print(f"  imagens {done}/{len(jobs)}  [{time.time() - t0:.0f}s]")
        print(f"Imagens: {len(jobs)} ({len(rels)} fotos)  [{time.time() - t0:.0f}s]")

    write_json(data / "meta.json", {"gerado_em": datetime.now().strftime("%d/%m/%Y %H:%M"), "banco": config.DB_PATH.name,
                                    "fotos": bool(args.with_images)})
    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"\nSite estático em {out}  ({size_mb:.0f} MB)  [{time.time() - t0:.0f}s]")
    print("Para testar: python -m http.server -d", out)


if __name__ == "__main__":
    main()
