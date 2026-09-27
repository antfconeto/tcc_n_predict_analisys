"""
Servidor local do site da análise: páginas estáticas + API JSON que consulta o banco SQLite.

Uso:  python run.py site            (ou python site/server.py [--port 8000] [--no-browser])
      abre http://localhost:8000

Rotas da API (todas leem o banco no momento da requisição, somente leitura):
  GET /api/overview                     números e blocos calculados para as seções
  GET /api/tables                       catálogo de tabelas (grupo, descrição, linhas, execuções)
  GET /api/table/<nome>                 linhas paginadas
        ?run=latest|all|<run_id>  &limit=50 &offset=0  &sort=<col> &dir=asc|desc
        &q=<texto>  &f.<coluna>=<valor>  (filtros por igualdade, repetíveis)
  GET /api/table/<nome>/distinct?col=<coluna>[&f.<col>=<v>]   valores distintos (todas as execuções)
  GET /api/table/<nome>.csv             mesma consulta, em CSV (sem limite); fmt=br → ";" e vírgula decimal
  GET /api/downloads                    tabelas agrupadas por tema para a aba "Baixar dados"
  GET /api/export.zip?t=a&t=b           várias tabelas em CSV num ZIP (run=latest|all, fmt=std|br)
  GET /api/maps                         imagens de mapa + resumo por imagem
  GET /api/reports | /api/report?name=  relatórios markdown guardados no banco
  GET /api/runs                         histórico de execuções
  GET /api/field                        mapa do campo (parcelas, médias por data, fotos, alertas)
  GET /api/plot/<id>                    tudo sobre uma parcela (id = 1..12 ou teste-1..3)
  GET /api/outliers                     casos fora do padrão e justificativas calculadas
  GET /api/figures                      dados dos gráficos da página de análise
  GET /api/per_date                     análise dentro de cada coleta (resumo, ANOVA, pontos, consistência)
  GET /api/classes                      classes de clorofila (faixas de Beaufils): esquema, parcelas, concordância
  GET /api/ciencia                      análises complementares (repetibilidade, Bland-Altman, dose ótima, …)
  GET /api/vegetativo                   índices × altura e biomassa e busca de MLP
  GET /api/photos                       catálogo das fotos com diagnósticos
  GET /photo/<caminho>?w=640            miniatura da foto original (cache em outputs/cache)
  GET /photo-step/<etapa>/<caminho>     etapa do processamento: crop | pre | mask | blocks
  GET /plots/<caminho>                  gráficos PNG de outputs/plots
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import sqlite3
import sys
import threading
import traceback
import zipfile
from datetime import datetime
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

SITE_DIR = Path(__file__).resolve().parent
STATIC_DIR = SITE_DIR / "static"
sys.path.insert(0, str(SITE_DIR))
sys.path.insert(0, str(SITE_DIR.parent))

import imaging  # noqa: E402
import queries  # noqa: E402
from tcc_analysis import config  # noqa: E402

MAX_LIMIT = 5000


def ro_connect() -> sqlite3.Connection:
    if not config.DB_PATH.exists():
        raise FileNotFoundError(f"Banco não encontrado: {config.DB_PATH}. Rode `python run.py setup` e `python run.py all`.")
    conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    return conn


def table_columns(conn: sqlite3.Connection, name: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')]


def valid_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows} - queries.INTERNAL_TABLES


def _json_value(v):
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    return v


def sanitize(obj):
    """Converte NaN/inf (inválidos em JSON) em null, recursivamente."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return _json_value(queries._clean(obj))


def build_table_query(conn, name: str, params: dict[str, list[str]], paginate: bool = True):
    if name not in valid_tables(conn):
        raise LookupError(f"Tabela desconhecida: {name}")
    cols = table_columns(conn, name)
    has_run = "run_id" in cols and name not in queries.RAW_TABLES
    where, args = [], []

    for key, values in params.items():
        if key.startswith("f."):
            col = key[2:]
            if col not in cols:
                raise ValueError(f"Coluna desconhecida: {col}")
            where.append(f'CAST("{col}" AS TEXT) = ?')
            args.append(values[0])

    # "latest" = execução mais recente *entre as linhas que passam nos filtros*
    run = params.get("run", ["latest"])[0]
    run_ids = []
    run_id = None
    if has_run:
        filt_sql = (" WHERE " + " AND ".join(where)) if where else ""
        run_ids = [r[0] for r in conn.execute(
            f'SELECT DISTINCT run_id FROM "{name}"{filt_sql} ORDER BY run_id DESC', args)]
        if run == "latest":
            run_id = run_ids[0] if run_ids else None
        elif run != "all":
            run_id = int(run)
        if run_id is not None:
            where.append("run_id = ?")
            args.append(run_id)

    q = params.get("q", [""])[0].strip()
    if q:
        where.append("(" + " OR ".join(f'CAST("{c}" AS TEXT) LIKE ?' for c in cols) + ")")
        args.extend([f"%{q}%"] * len(cols))

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    order_sql = ""
    sort = params.get("sort", [""])[0]
    if sort:
        if sort not in cols:
            raise ValueError(f"Coluna desconhecida: {sort}")
        direction = "DESC" if params.get("dir", ["asc"])[0].lower() == "desc" else "ASC"
        order_sql = f' ORDER BY "{sort}" IS NULL, "{sort}" {direction}'

    select_cols = [c for c in cols if not (has_run and c == "run_id" and run != "all")]
    select_sql = ", ".join(f'"{c}"' for c in select_cols)
    total = conn.execute(f'SELECT COUNT(*) FROM "{name}"{where_sql}', args).fetchone()[0]
    sql = f'SELECT {select_sql} FROM "{name}"{where_sql}{order_sql}'
    if paginate:
        limit = max(1, min(int(params.get("limit", ["50"])[0]), MAX_LIMIT))
        offset = max(0, int(params.get("offset", ["0"])[0]))
        sql += f" LIMIT {limit} OFFSET {offset}"
    return sql, args, select_cols, total, run_id, run_ids


class Handler(BaseHTTPRequestHandler):
    server_version = "TCCSite/1.0"

    def log_message(self, fmt, *args):  # silencioso, exceto erros
        pass

    # ── respostas ───────────────────────────────────────────────────────────
    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(sanitize(payload), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, root: Path):
        try:
            path = path.resolve()
            path.relative_to(root.resolve())
        except ValueError:
            return self.send_error(HTTPStatus.FORBIDDEN)
        if not path.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data: bytes, ctype: str):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    # ── roteamento ──────────────────────────────────────────────────────────
    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        params = parse_qs(url.query)
        try:
            if path in ("/", "/index.html"):
                return self.send_file(SITE_DIR / "index.html", SITE_DIR)
            if path.startswith("/pages/"):
                return self.send_file(SITE_DIR / "pages" / path[len("/pages/"):], SITE_DIR / "pages")
            if path.startswith("/static/"):
                return self.send_file(STATIC_DIR / path[len("/static/"):], STATIC_DIR)
            if path.startswith("/photo/"):
                width = int(params.get("w", ["640"])[0])
                return self.send_bytes(imaging.thumbnail(path[len("/photo/"):], width), "image/jpeg")
            if path.startswith("/photo-step/"):
                step, _, rel = path[len("/photo-step/"):].partition("/")
                width = int(params.get("w", ["640"])[0])
                return self.send_bytes(imaging.step_image(rel, step, width), "image/jpeg")
            if path.startswith("/plots/"):
                return self.send_file(config.PLOTS_DIR / path[len("/plots/"):], config.PLOTS_DIR)
            if path.startswith("/api/"):
                return self.route_api(path[len("/api/"):], params)
            return self.send_json({"error": f"Rota não encontrada: {path}"}, HTTPStatus.NOT_FOUND)
        except (LookupError, ValueError) as exc:
            return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as exc:
            return self.send_json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self.send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def route_api(self, route: str, params):
        if route == "overview":
            return self.send_json(queries.overview())
        if route == "tables":
            return self.send_json(queries.table_catalog())
        if route == "downloads":
            return self.send_json(queries.downloads())
        if route == "export.zip":
            return self.export_zip(params)
        if route == "field":
            return self.send_json(queries.field_map())
        if route.startswith("plot/"):
            return self.send_json(queries.plot_detail(route[len("plot/"):]))
        if route == "outliers":
            return self.send_json(queries.outliers())
        if route == "per_date":
            return self.send_json(queries.per_date())
        if route == "classes":
            return self.send_json(queries.chl_classes())
        if route == "ciencia":
            return self.send_json(queries.science())
        if route == "vegetativo":
            return self.send_json(queries.vegetative())
        if route == "figures":
            return self.send_json(queries.figures())
        if route == "photos":
            return self.send_json(queries.photos())
        if route == "maps":
            return self.send_json(queries.map_gallery())
        if route == "runs":
            return self.send_json(queries.runs())
        if route == "reports":
            return self.send_json(queries.report_list())
        if route == "report":
            name = params.get("name", [""])[0]
            return self.send_json({"name": name, "content": queries.db.load_report(name)})
        if route.startswith("table/"):
            rest = route[len("table/"):]
            if rest.endswith(".csv"):
                return self.table_csv(rest[:-4], params)
            if rest.endswith("/distinct"):
                return self.table_distinct(rest[: -len("/distinct")], params)
            return self.table_rows(rest, params)
        return self.send_error(HTTPStatus.NOT_FOUND)

    def table_rows(self, name, params):
        with ro_connect() as conn:
            sql, args, cols, total, run_id, run_ids = build_table_query(conn, name, params)
            rows = [[_json_value(v) for v in r] for r in conn.execute(sql, args)]
        self.send_json({"table": name, "columns": cols, "rows": rows, "total": total,
                        "run_id": run_id, "runs": run_ids})

    def table_distinct(self, name, params):
        col = params.get("col", [""])[0]
        with ro_connect() as conn:
            if name not in valid_tables(conn) or col not in table_columns(conn, name):
                raise LookupError("Tabela ou coluna desconhecida")
            where, args = [], []
            for key, values in params.items():
                if key.startswith("f."):
                    if key[2:] not in table_columns(conn, name):
                        raise ValueError(f"Coluna desconhecida: {key[2:]}")
                    where.append(f'CAST("{key[2:]}" AS TEXT) = ?')
                    args.append(values[0])
            where_sql = (" WHERE " + " AND ".join(where)) if where else ""
            vals = [r[0] for r in conn.execute(
                f'SELECT DISTINCT "{col}" FROM "{name}"{where_sql} ORDER BY "{col}" LIMIT 500', args)]
        self.send_json(vals)

    def table_csv(self, name, params):
        fmt = params.get("fmt", ["std"])[0]
        data = csv_bytes(name, params, fmt)
        self.send_download(data, "text/csv; charset=utf-8", f"{name}.csv")

    def export_zip(self, params):
        names = params.get("t", [])
        if not names:
            raise ValueError("Nenhuma tabela selecionada")
        fmt = params.get("fmt", ["std"])[0]
        run = params.get("run", ["latest"])[0]
        catalog = {m["name"]: m for c in queries.downloads() for m in c["tables"]}
        buf = io.BytesIO()
        readme = [f"Exportado de database/tcc.sqlite em {datetime.now():%d/%m/%Y %H:%M}.",
                  "Execução: " + ("todas (coluna run_id identifica cada uma)" if run == "all" else "a mais recente de cada tabela"),
                  "Formato: " + ("ponto e vírgula, vírgula decimal, UTF-8 com BOM (Excel em português)" if fmt == "br"
                                 else "vírgula, ponto decimal, UTF-8"), ""]
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in names:
                if name not in catalog:
                    raise LookupError(f"Tabela desconhecida: {name}")
                folder = queries.download_category(name)
                zf.writestr(f"{folder}/{name}.csv", csv_bytes(name, {"run": [run]}, fmt))
                readme.append(f"{folder}/{name}.csv: {catalog[name].get('description') or ''}")
            zf.writestr("LEIAME.txt", "\n".join(readme) + "\n")
        self.send_download(buf.getvalue(), "application/zip", f"dados_tcc_{datetime.now():%Y%m%d}.zip")

    def send_download(self, data, ctype, filename):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _num(v, br):
    """Floats com 10 algarismos significativos (sem ruído de arredondamento); br → vírgula decimal."""
    if isinstance(v, float):
        if v != v:
            return ""
        out = f"{v:.10g}"
        return out.replace(".", ",") if br else out
    return v


def csv_bytes(name: str, params: dict, fmt: str = "std") -> bytes:
    """CSV de uma tabela do banco ou de uma planilha consolidada. fmt=br: ';', vírgula decimal e BOM (Excel pt-BR)."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";" if fmt == "br" else ",")
    if name in queries.VIRTUAL_TABLES:
        df = queries.virtual_table(name)
        cols, rows = list(df.columns), [[None if pd_isna(v) else (v.item() if hasattr(v, "item") else v) for v in r]
                                         for r in df.itertuples(index=False)]
    else:
        with ro_connect() as conn:
            sql, args, cols, *_ = build_table_query(conn, name, params, paginate=False)
            rows = conn.execute(sql, args).fetchall()
    writer.writerow(cols)
    br = fmt == "br"
    for r in rows:
        writer.writerow([_num(v, br) for v in r])
    text = buf.getvalue()
    return ("\ufeff" + text).encode("utf-8") if fmt == "br" else text.encode("utf-8")


def pd_isna(v):
    try:
        return v is None or (isinstance(v, float) and v != v) or str(v) == "<NA>"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    ro_connect().close()  # falha cedo se o banco não existir
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Site da análise em {url}  (banco: {config.DB_PATH})  — Ctrl+C para parar")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")


if __name__ == "__main__":
    main()
