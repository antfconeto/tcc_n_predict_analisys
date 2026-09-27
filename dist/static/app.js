/* Núcleo do site: API, formatação, tabelas, gráficos SVG, foto ampliada e roteamento. */
"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// utilitários
// ─────────────────────────────────────────────────────────────────────────────
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}
const SVGNS = "http://www.w3.org/2000/svg";
function s(tag, attrs = {}, text) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}

class ServerMissing extends Error {}
// Versão estática (python run.py build): window.TCC_STATIC = true e os dados vêm de data/*.json
const STATIC = !!window.TCC_STATIC;
async function api(path) {
  if (STATIC) return staticApi(path);
  let res;
  try { res = await fetch(path, { cache: "no-store" }); } catch { throw new ServerMissing("O servidor do site não respondeu."); }
  if (!(res.headers.get("content-type") || "").includes("application/json")) {
    throw new ServerMissing("A página foi aberta sem o servidor do site.");
  }
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}
const cache = new Map();
function cached(path) {
  if (!cache.has(path)) cache.set(path, api(path).catch((e) => { cache.delete(path); throw e; }));
  return cache.get(path);
}
function qs(params) {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== "" && v != null) u.append(k, v);
  return u.toString();
}


// ─────────────────────────────────────────────────────────────────────────────
// modo estático: a "API" lê arquivos gerados pelo build; tabelas, CSV e ZIP são feitos aqui
// ─────────────────────────────────────────────────────────────────────────────
const safeName = (n) => String(n).replace(/[^A-Za-z0-9._-]/g, "_");   // igual a build_static.safe_name
async function fetchJson(rel) {
  const res = await fetch(rel);
  if (!res.ok) throw new Error(`Arquivo não encontrado: ${rel}`);
  return res.json();
}
const staticTables = new Map();
function staticTable(name) {
  if (!staticTables.has(name)) staticTables.set(name, fetchJson(`data/tables/${safeName(name)}.json`));
  return staticTables.get(name);
}
async function staticApi(path) {
  const u = new URL(path, "http://x");
  const route = decodeURIComponent(u.pathname.replace(/^\/api\//, ""));
  const params = Object.fromEntries(u.searchParams);
  if (route.startsWith("table/")) {
    const name = route.slice("table/".length);
    return staticQuery(name, await staticTable(name), params);
  }
  if (route === "report") return fetchJson(`data/api/report/${safeName(params.name)}.json`);
  return fetchJson(`data/api/${route.split("/").map(encodeURIComponent).join("/")}.json`);
}
/** Mesmas regras de build_table_query (server.py): filtros f.*, execução, busca, ordenação e paginação. */
function staticQuery(name, t, params, paginate = true) {
  const idx = Object.fromEntries(t.columns.map((c, i) => [c, i]));
  const txt = (v) => (v == null ? "" : String(v));
  let rows = t.rows;
  for (const [k, v] of Object.entries(params)) {
    if (!k.startsWith("f.")) continue;
    const i = idx[k.slice(2)];
    if (i == null) throw new Error(`Coluna desconhecida: ${k.slice(2)}`);
    rows = rows.filter((r) => txt(r[i]) === String(v) || (typeof r[i] === "number" && r[i] === Number(v)));
  }
  const run = params.run ?? "latest", ri = idx.run_id;
  let runs = [], runId = null;
  if (t.has_run && ri != null) {
    runs = [...new Set(rows.map((r) => r[ri]))].sort((a, b) => b - a);
    runId = run === "latest" ? (runs[0] ?? null) : run === "all" ? null : Number(run);
    if (runId != null) rows = rows.filter((r) => r[ri] === runId);
  }
  const q = (params.q || "").trim().toLowerCase();
  if (q) rows = rows.filter((r) => r.some((v) => txt(v).toLowerCase().includes(q)));
  if (params.sort) {
    const i = idx[params.sort], dir = (params.dir || "asc").toLowerCase() === "desc" ? -1 : 1;
    if (i == null) throw new Error(`Coluna desconhecida: ${params.sort}`);
    rows = [...rows].sort((a, b) => {
      const x = a[i], y = b[i];
      if (x == null || y == null) return (x == null) - (y == null);
      return (typeof x === "number" && typeof y === "number" ? x - y : txt(x) < txt(y) ? -1 : txt(x) > txt(y) ? 1 : 0) * dir;
    });
  }
  const keep = t.columns.map((c, i) => [c, i]).filter(([c]) => !(t.has_run && c === "run_id" && run !== "all"));
  const total = rows.length;
  if (paginate) {
    const off = Number(params.offset || 0), lim = Number(params.limit || 50);
    rows = rows.slice(off, off + lim);
  }
  return { table: name, columns: keep.map(([c]) => c), rows: rows.map((r) => keep.map(([, i]) => r[i])), total, run_id: runId, runs };
}
/** CSV igual ao do servidor (csv_bytes): 10 algarismos significativos; br → ";" + vírgula decimal + BOM. */
function toCsv(columns, rows, fmtv = "std") {
  const br = fmtv === "br", sep = br ? ";" : ",";
  const cell = (v) => {
    if (v == null) return "";
    let s = typeof v === "number" && !Number.isInteger(v) ? String(Number(v.toPrecision(10))) : String(v);
    if (br && typeof v === "number") s = s.replace(".", ",");
    return s.includes(sep) || /["\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;   // como o csv do Python
  };
  const text = [columns, ...rows].map((r) => r.map(cell).join(sep)).join("\r\n") + "\r\n";
  return (br ? "\ufeff" : "") + text;
}
async function staticCsv(name, params, fmtv) {
  const t = await staticTable(name);
  const d = staticQuery(name, t, params, false);
  return toCsv(d.columns, d.rows, fmtv);
}
function saveBlob(blob, filename) {
  const a = h("a", { href: URL.createObjectURL(blob), download: filename });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}
/** ZIP sem compressão (método "store"): suficiente para CSV e sem biblioteca externa. */
const CRC_TABLE = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(bytes) { let c = 0xffffffff; for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
function makeZip(files) {
  const enc = new TextEncoder(), parts = [], central = [];
  let offset = 0;
  const now = new Date(), dosTime = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
  const dosDate = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  for (const f of files) {
    const name = enc.encode(f.name), data = typeof f.data === "string" ? enc.encode(f.data) : f.data, crc = crc32(data);
    const head = new DataView(new ArrayBuffer(30));
    [[0, 0x04034b50, 4], [4, 20, 2], [6, 0x0800, 2], [8, 0, 2], [10, dosTime, 2], [12, dosDate, 2], [14, crc, 4],
      [18, data.length, 4], [22, data.length, 4], [26, name.length, 2], [28, 0, 2]].forEach(([o, v, n]) => (n === 4 ? head.setUint32(o, v, true) : head.setUint16(o, v, true)));
    const cen = new DataView(new ArrayBuffer(46));
    [[0, 0x02014b50, 4], [4, 20, 2], [6, 20, 2], [8, 0x0800, 2], [10, 0, 2], [12, dosTime, 2], [14, dosDate, 2], [16, crc, 4],
      [20, data.length, 4], [24, data.length, 4], [28, name.length, 2], [30, 0, 2], [32, 0, 2], [34, 0, 2], [36, 0, 2], [38, 0, 4], [42, offset, 4]]
      .forEach(([o, v, n]) => (n === 4 ? cen.setUint32(o, v, true) : cen.setUint16(o, v, true)));
    parts.push(head, name, data);
    central.push(cen, name);
    offset += 30 + name.length + data.length;
  }
  const size = central.reduce((a, p) => a + p.byteLength, 0);
  const end = new DataView(new ArrayBuffer(22));
  [[0, 0x06054b50, 4], [4, 0, 2], [6, 0, 2], [8, files.length, 2], [10, files.length, 2], [12, size, 4], [16, offset, 4], [20, 0, 2]]
    .forEach(([o, v, n]) => (n === 4 ? end.setUint32(o, v, true) : end.setUint16(o, v, true)));
  return new Blob([...parts, ...central, end], { type: "application/zip" });
}

const nf = (d) => new Intl.NumberFormat("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });
function fmt(v, f) {
  if (v == null || v === "") return "–";
  if (typeof v !== "number") return String(v);
  if (f === "p") {
    if (v === 0) return "0";
    if (Math.abs(v) < 0.001) { const [m, e] = v.toExponential(1).split("e"); return `${m.replace(".", ",")}×10${sup(e)}`; }
    return nf(3).format(v);
  }
  if (f === "+") return (v > 0 ? "+" : "") + nf(1).format(v);
  if (f != null && f !== "") return nf(Number(f)).format(v);
  if (Number.isInteger(v)) return new Intl.NumberFormat("pt-BR").format(v);
  const a = Math.abs(v);
  const upTo = (d) => new Intl.NumberFormat("pt-BR", { maximumFractionDigits: d }).format(v);
  if (a >= 100) return upTo(1);
  if (a >= 10) return upTo(2);
  if (a >= 1) return upTo(4);
  if (a === 0) return "0";
  if (a < 0.001) return v.toExponential(2).replace(".", ",");
  return nf(4).format(v);
}
function sup(e) {
  const m = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  return String(Number(e)).split("").map((c) => m[c] ?? c).join("");
}
const shortDate = (d) => (d ? d.slice(0, 5).replace("-", "/") : "");
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// cor fixa por data (a mesma em todo o site); 01/06 é neutra (fora do experimento)
const DATES = ["18-05-2026", "21-05-2026", "26-05-2026"];
const TEST = "01-06-2026";
function dateColor(d) {
  const i = DATES.indexOf(d);
  return i >= 0 ? css(`--series-${i + 1}`) : css("--neutral");
}
const SEQ = () => ["--seq-100", "--seq-200", "--seq-300", "--seq-400", "--seq-500", "--seq-600", "--seq-700"].map(css);
const DOSE_STEP = { 0: 1, 50: 3, 75: 4, 100: 6 };   // passos do ramp sequencial

// ─────────────────────────────────────────────────────────────────────────────
// tooltip e foto ampliada
// ─────────────────────────────────────────────────────────────────────────────
const tip = $("#tooltip");
function showTip(evt, title, rows) {
  tip.replaceChildren();
  if (title) tip.append(h("div", { class: "tt-title" }, title));
  for (const r of rows) {
    const key = r.color ? h("span", { class: r.kind === "line" ? "key-line" : "key-dot", style: `background:${r.color}` }) : null;
    tip.append(h("div", { class: "tt-row" }, key, h("strong", {}, r.value), r.label ? h("span", { class: "muted" }, r.label) : null));
  }
  tip.style.opacity = "1";
  const w = tip.offsetWidth, hh = tip.offsetHeight;
  tip.style.left = `${Math.min(evt.clientX + 14, innerWidth - w - 8)}px`;
  tip.style.top = `${Math.max(8, evt.clientY - hh - 12)}px`;
}
function hideTip() { tip.style.opacity = "0"; }

const STEP_INFO = {
  photo: ["Foto original", "Como saiu da câmera."],
  crop: ["Recorte de 20%", "Só o retângulo central entra na análise; as bordas (sombras, estacas, parcela vizinha) são descartadas."],
  pre: ["Filtro, gamma e máscara HSV", "Filtro bilateral suaviza o ruído sem borrar as folhas; gamma 0,8 ajusta o contraste; pixels de sombra (V < 45), estourados (V > 240) ou sem cor (S < 25) são descartados (em preto)."],
  mask: ["Máscara de vegetação", "Blocos de 10×10 px com ExG ≤ 0,15 (solo, palha, sombra) aparecem em cinza e são ignorados."],
  blocks: ["O que o modelo vê", "Cada bloco vira sua cor média. As 24 entradas da rede são estatísticas dessas cores."],
};
const STATIC_PHOTOS = STATIC && !!window.TCC_STATIC_PHOTOS;   // build com --with-images
function photoUrl(rel, w = 640) {
  if (STATIC && !STATIC_PHOTOS) return "data/sem-foto.svg";
  return STATIC ? `data/photos/${w > 700 ? "lg" : "sm"}/${encodeURI(rel)}` : `/photo/${encodeURI(rel)}?w=${w}`;
}
function stepUrl(rel, step, w = 640) {
  if (step === "photo") return photoUrl(rel, w);
  if (STATIC && !STATIC_PHOTOS) return "data/sem-foto.svg";
  return STATIC ? `data/steps/${step}/${w > 700 ? "lg" : "sm"}/${encodeURI(rel)}` : `/photo-step/${step}/${encodeURI(rel)}?w=${w}`;
}

const lb = { el: $("#lightbox"), img: $("#lb-img"), title: $("#lb-title"), caption: $("#lb-caption"), opts: $("#lb-opts") };
function openPhoto(rel, title, caption) {
  const show = (step) => {
    lb.img.src = stepUrl(rel, step, 1400);
    lb.img.alt = `${title} — ${STEP_INFO[step][0]}`;
    $$("button", lb.opts).forEach((b) => b.classList.toggle("on", b.dataset.step === step));
    lb.caption.textContent = `${STEP_INFO[step][1]}${caption ? `  ${caption}` : ""}`;
  };
  lb.title.textContent = title;
  lb.opts.replaceChildren(...Object.entries(STEP_INFO).map(([k, [label]]) => h("button", { type: "button", "data-step": k, onclick: () => show(k) }, label)));
  show("photo");
  lb.el.classList.add("open");
  $("#lb-close").focus();
}
function closePhoto() { lb.el.classList.remove("open"); lb.img.removeAttribute("src"); }
$("#lb-close").addEventListener("click", closePhoto);
lb.el.addEventListener("click", (e) => { if (e.target === lb.el || e.target.classList.contains("lb-body")) closePhoto(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePhoto(); });

// ─────────────────────────────────────────────────────────────────────────────
// tabelas
// ─────────────────────────────────────────────────────────────────────────────
function cell(v, f) {
  return h("td", { class: typeof v === "number" ? "num" : v == null ? "null" : "" }, fmt(v, f));
}
function localTable(columns, rows, { formats = {}, labels = {} } = {}) {
  const first = rows[0];
  const isNum = (c, i) => first && typeof (Array.isArray(first) ? first[i] : first[c]) === "number";
  return h("div", { class: "table-wrap" }, h("table", { class: "data" },
    h("thead", {}, h("tr", {}, columns.map((c, i) => h("th", { class: isNum(c, i) ? "num" : "" }, labels[c] ?? c)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, columns.map((c, i) => cell(Array.isArray(r) ? r[i] : r[c], formats[c])))))));
}

/** Tabela paginada: cada página, busca ou ordenação é uma consulta a /api/table/<nome>. */
function serverTable(container, table, opts = {}) {
  const st = { limit: opts.limit ?? 15, offset: 0, sort: opts.sort ?? "", dir: opts.dir ?? "asc", q: "", filters: { ...(opts.filters || {}) }, run: "latest" };
  const search = h("input", { type: "search", placeholder: "Buscar…", "aria-label": "Buscar na tabela" });
  const status = h("span"), prev = h("button", { type: "button" }, "← anterior"), next = h("button", { type: "button" }, "próxima →");
  const csv = h("a", { href: "#" }, "baixar CSV");
  const wrap = h("div");
  container.replaceChildren(opts.noSearch ? "" : h("div", { class: "search-row" }, search), wrap,
    h("div", { class: "table-footer" }, status, h("span", { class: "pager" }, prev, next, csv)));
  let timer;
  search.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => { st.q = search.value; st.offset = 0; load(); }, 250); });
  prev.addEventListener("click", () => { st.offset = Math.max(0, st.offset - st.limit); load(); });
  next.addEventListener("click", () => { st.offset += st.limit; load(); });
  const params = (page = true) => {
    const p = { run: st.run, q: st.q, sort: st.sort, dir: st.dir, ...(page ? { limit: st.limit, offset: st.offset } : {}) };
    for (const [k, v] of Object.entries(st.filters)) p[`f.${k}`] = v;
    return p;
  };
  async function load() {
    wrap.style.opacity = ".5";
    try {
      const d = await api(`/api/table/${encodeURIComponent(table)}?${qs(params())}`);
      const hide = new Set(opts.hide || []);
      const cols = d.columns.map((c, i) => [c, i]).filter(([c]) => !hide.has(c));
      const numeric = (i) => d.rows.some((r) => typeof r[i] === "number");
      wrap.replaceChildren(h("div", { class: "table-wrap" }, h("table", { class: "data" },
        h("thead", {}, h("tr", {}, cols.map(([c, i]) => h("th", {
          class: `sortable ${numeric(i) ? "num" : ""}`, title: "Ordenar",
          onclick: () => { st.dir = st.sort === c && st.dir === "asc" ? "desc" : "asc"; st.sort = c; st.offset = 0; load(); },
        }, (opts.labels?.[c] ?? c) + (st.sort === c ? (st.dir === "asc" ? " ↑" : " ↓") : ""))))),
        h("tbody", {}, d.rows.map((r) => h("tr", {}, cols.map(([c, i]) => cell(r[i], opts.formats?.[c]))))))));
      const end = Math.min(st.offset + st.limit, d.total);
      status.textContent = d.total ? `${fmt(st.offset + 1)}–${fmt(end)} de ${fmt(d.total)}${d.run_id != null ? ` · execução ${d.run_id}` : ""}` : "nenhuma linha";
      prev.disabled = st.offset === 0; next.disabled = end >= d.total;
      if (STATIC) {
        csv.href = "#";
        csv.onclick = async (e) => { e.preventDefault(); saveBlob(new Blob([await staticCsv(table, params(false), "std")], { type: "text/csv" }), `${table}.csv`); };
      } else csv.href = `/api/table/${encodeURIComponent(table)}.csv?${qs(params(false))}`;
      opts.onLoad?.(d);
    } catch (err) { wrap.replaceChildren(h("p", { class: "muted" }, `Erro ao consultar ${table}: ${err.message}`)); }
    wrap.style.opacity = "1";
  }
  load();
  return { setRun(r) { st.run = r; st.offset = 0; load(); } };
}
async function fetchAll(table, filters = {}, extra = {}) {
  const p = { limit: 5000, ...extra };
  for (const [k, v] of Object.entries(filters)) p[`f.${k}`] = v;
  const d = await api(`/api/table/${encodeURIComponent(table)}?${qs(p)}`);
  return d.rows.map((r) => Object.fromEntries(d.columns.map((c, i) => [c, r[i]])));
}

// ─────────────────────────────────────────────────────────────────────────────
// gráficos SVG
// ─────────────────────────────────────────────────────────────────────────────
const charts = new Map();
const chartTables = new Map();
function registerChart(id, render, table) { charts.set(id, render); if (table) chartTables.set(id, table); render(); }
let rt;
addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => charts.forEach((r) => r()), 150); });

function niceTicks(min, max, count = 5) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min, raw = span / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((x) => span / x <= count) ?? 10 * mag;
  const out = [];
  for (let v = Math.floor(min / step) * step; v <= Math.ceil(max / step) * step + step / 2; v += step) out.push(Number(v.toFixed(10)));
  return out;
}
const decimals = (t) => { const st = Math.abs((t[1] ?? t[0] + 1) - t[0]); return Number.isInteger(Number(st.toFixed(6))) ? 0 : st >= 0.1 ? 1 : 2; };
const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
function frame(el, height) {
  const width = Math.max(280, el.clientWidth);
  const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img" });
  el.replaceChildren(svg);
  return { svg, width };
}
function legend(el, items) {
  el.append(h("div", { class: "legend" }, items.map((it) => h("span", {},
    h("span", { class: it.kind === "ring" ? "key-ring" : it.kind === "line" ? "key-line" : it.kind === "rect" ? "key-rect" : "key-dot", style: it.kind === "ring" ? `border-color:${it.color}` : `background:${it.color}` }), it.name))));
}
function axes(svg, { x, y, xt, yt, x0, x1, y0, y1, xLabel, yLabel, xFmt, yFmt, height }) {
  const yd = decimals(yt), xd = decimals(xt || [0, 1]);
  for (const t of yt) {
    svg.append(s("line", { x1: x0, x2: x1, y1: y(t), y2: y(t), class: "gridline" }));
    svg.append(s("text", { x: x0 - 8, y: y(t) + 4, "text-anchor": "end", class: "tick-label" }, (yFmt || ((v) => fmt(v, yd)))(t)));
  }
  svg.append(s("line", { x1: x0, x2: x1, y1: y0, y2: y0, class: "baseline" }));
  if (xt) for (const t of xt) svg.append(s("text", { x: x(t), y: y0 + 18, "text-anchor": "middle", class: "tick-label" }, (xFmt || ((v) => fmt(v, xd)))(t)));
  if (xLabel) svg.append(s("text", { x: (x0 + x1) / 2, y: height - 6, "text-anchor": "middle", class: "axis-label" }, xLabel));
  if (yLabel) svg.append(s("text", { x: x0, y: y1 - 12, class: "axis-label" }, yLabel));
}

/** Linhas com barras de erro. */
function lineChart(el, { series, xTicks, xLabel, yLabel, height = 320, xFmt = (v) => fmt(v) }) {
  const { svg, width } = frame(el, height);
  const m = { l: 44, r: 70, t: 30, b: 44 };
  const ys = series.flatMap((se) => se.points.flatMap((p) => [p.y - (p.e || 0), p.y + (p.e || 0)]));
  const yt = niceTicks(Math.min(...ys), Math.max(...ys), 5);
  const x = scale(Math.min(...xTicks), Math.max(...xTicks), m.l + 12, width - m.r);
  const y = scale(yt[0], yt.at(-1), height - m.b, m.t);
  axes(svg, { x, y, xt: xTicks, yt, x0: m.l, x1: width - m.r + 12, y0: height - m.b, y1: m.t, xLabel, yLabel, xFmt, height });
  const surface = css("--paper"), labels = [];
  for (const se of series) {
    const pts = [...se.points].sort((a, b) => a.x - b.x);
    for (const p of pts) if (p.e) svg.append(s("line", { x1: x(p.x), x2: x(p.x), y1: y(p.y - p.e), y2: y(p.y + p.e), stroke: se.color, "stroke-width": 1.5, opacity: .6 }));
    svg.append(s("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join(""), fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round" }));
    for (const p of pts) svg.append(s("circle", { cx: x(p.x), cy: y(p.y), r: 4, fill: se.color, stroke: surface, "stroke-width": 2 }));
    const last = pts.at(-1);
    labels.push({ x: x(last.x) + 10, y: y(last.y) + 4, t: se.name });
  }
  labels.sort((a, b) => a.y - b.y);
  if (!labels.some((l, i) => i && l.y - labels[i - 1].y < 14)) for (const l of labels) svg.append(s("text", { x: l.x, y: l.y, class: "data-label" }, l.t));
  const cross = s("line", { y1: m.t, y2: height - m.b, class: "crosshair", opacity: 0 });
  const hit = s("rect", { x: m.l, y: m.t, width: width - m.l - m.r + 12, height: height - m.t - m.b, class: "hit" });
  svg.append(cross, hit);
  hit.addEventListener("pointermove", (evt) => {
    const nearest = xTicks.reduce((a, b) => (Math.abs(x(b) - evt.offsetX) < Math.abs(x(a) - evt.offsetX) ? b : a));
    cross.setAttribute("x1", x(nearest)); cross.setAttribute("x2", x(nearest)); cross.setAttribute("opacity", 1);
    showTip(evt, `${xLabel}: ${xFmt(nearest)}`, series.map((se) => {
      const p = se.points.find((q) => q.x === nearest);
      return p && { color: se.color, kind: "line", value: fmt(p.y, 1), label: `${se.name}${p.e ? ` (± ${fmt(p.e, 1)})` : ""}` };
    }).filter(Boolean));
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
  legend(el, series.map((se) => ({ name: se.name, color: se.color, kind: "line" })));
}

/**
 * Dispersão. points: [{x, y, color, hollow, label, tip:[linhas extras], mark:"rótulo direto"}]
 * opts.identity desenha a linha 1:1; opts.fit desenha a reta de mínimos quadrados.
 */
function scatter(el, { points, xLabel, yLabel, height = 340, identity = false, fit = false, square = false, legendItems, onClick, xFmt, yFmt, refLines = [] }) {
  const { svg, width } = frame(el, height);
  const m = { l: 48, r: refLines.some((r) => r.axis !== "x") ? 90 : 20, t: 30, b: 44 };
  let xs = [...points.map((p) => p.x), ...refLines.filter((r) => r.axis === "x").map((r) => r.value)];
  let ys = [...points.map((p) => p.y), ...refLines.filter((r) => r.axis !== "x").map((r) => r.value)];
  if (square) { xs = ys = [...xs, ...ys]; }
  const xt = niceTicks(Math.min(...xs), Math.max(...xs), 5), yt = square ? xt : niceTicks(Math.min(...ys), Math.max(...ys), 5);
  const plotW = square ? Math.min(width - m.l - m.r, height - m.t - m.b + 140) : width - m.l - m.r;
  const x = scale(xt[0], xt.at(-1), m.l, m.l + plotW), y = scale(yt[0], yt.at(-1), height - m.b, m.t);
  axes(svg, { x, y, xt, yt, x0: m.l, x1: m.l + plotW, y0: height - m.b, y1: m.t, xLabel, yLabel, xFmt, yFmt, height });
  if (identity) svg.append(s("line", { x1: x(xt[0]), y1: y(xt[0]), x2: x(xt.at(-1)), y2: y(xt.at(-1)), class: "ref-line" }),
    s("text", { x: x(xt.at(-1)) - 4, y: y(xt.at(-1)) + 14, "text-anchor": "end", class: "tick-label" }, "1:1"));
  let nx = 0;
  for (const r of refLines) {   // linhas de referência: {axis: "x"|"y", value, label, dash}
    const attrs = { class: "ref-line", "stroke-dasharray": r.dash === false ? null : "4 4" };
    if (r.axis === "x") {
      svg.append(s("line", { x1: x(r.value), x2: x(r.value), y1: y(yt[0]), y2: y(yt.at(-1)), ...attrs }));
      if (r.label) svg.append(s("text", { x: x(r.value) + 4, y: y(yt.at(-1)) + 12 + 14 * nx++, class: "tick-label" }, r.label));
    } else {
      svg.append(s("line", { x1: m.l, x2: m.l + plotW, y1: y(r.value), y2: y(r.value), ...attrs }));
      if (r.label) svg.append(s("text", { x: m.l + plotW + 6, y: y(r.value) + 4, class: "tick-label" }, r.label));
    }
  }
  if (fit && points.length > 2) {
    const n = points.length, mx = points.reduce((a, p) => a + p.x, 0) / n, my = points.reduce((a, p) => a + p.y, 0) / n;
    const b = points.reduce((a, p) => a + (p.x - mx) * (p.y - my), 0) / points.reduce((a, p) => a + (p.x - mx) ** 2, 0);
    const lo = Math.min(...points.map((p) => p.x)), hi = Math.max(...points.map((p) => p.x));
    svg.append(s("line", { x1: x(lo), y1: y(my + b * (lo - mx)), x2: x(hi), y2: y(my + b * (hi - mx)), class: "fit-line" }));
  }
  const surface = css("--paper");
  const dots = points.map((p) => {
    const c = p.hollow
      ? s("circle", { cx: x(p.x), cy: y(p.y), r: 4.5, fill: surface, stroke: p.color, "stroke-width": 1.8 })
      : s("circle", { cx: x(p.x), cy: y(p.y), r: 5, fill: p.color, stroke: surface, "stroke-width": 2 });
    svg.append(c);
    if (p.mark) svg.append(s("text", { x: x(p.x) + 8, y: y(p.y) - 7, class: "data-label strong" }, p.mark));
    return c;
  });
  const hit = s("rect", { x: m.l, y: m.t, width: plotW, height: height - m.t - m.b, class: "hit", style: onClick ? "cursor:pointer" : null });
  svg.append(hit);
  let act = -1;
  const nearest = (evt) => {
    let best = -1, bd = 26 * 26;
    points.forEach((p, i) => { const d = (x(p.x) - evt.offsetX) ** 2 + (y(p.y) - evt.offsetY) ** 2; if (d < bd) { bd = d; best = i; } });
    return best;
  };
  hit.addEventListener("pointermove", (evt) => {
    const i = nearest(evt);
    if (act >= 0) dots[act].setAttribute("r", points[act].hollow ? 4.5 : 5);
    act = i;
    if (i < 0) return hideTip();
    dots[i].setAttribute("r", 7);
    const p = points[i];
    showTip(evt, p.label, [{ color: p.color, value: fmt(p.y, 2), label: yLabel }, { value: fmt(p.x, 2), label: xLabel }, ...(p.tip || [])]);
  });
  hit.addEventListener("pointerleave", () => { if (act >= 0) dots[act].setAttribute("r", points[act].hollow ? 4.5 : 5); act = -1; hideTip(); });
  if (onClick) hit.addEventListener("click", (evt) => { const i = nearest(evt); if (i >= 0) onClick(points[i]); });
  if (legendItems) legend(el, legendItems);
}

/** Pontos por categoria (strip plot): groups = [{name, points:[{y, color, label, tip, mark, hollow}]}] */
function strip(el, { groups, yLabel, height = 320, zeroLine = false, means = true, onClick, legendItems }) {
  const { svg, width } = frame(el, height);
  const m = { l: 48, r: 16, t: 30, b: 40 };
  const ys = groups.flatMap((g) => g.points.map((p) => p.y)).concat(zeroLine ? [0] : []);
  const yt = niceTicks(Math.min(...ys), Math.max(...ys), 5);
  const y = scale(yt[0], yt.at(-1), height - m.b, m.t);
  const band = (width - m.l - m.r) / groups.length;
  axes(svg, { x: null, y, xt: null, yt, x0: m.l, x1: width - m.r, y0: height - m.b, y1: m.t, yLabel, height });
  if (zeroLine) svg.append(s("line", { x1: m.l, x2: width - m.r, y1: y(0), y2: y(0), class: "ref-line" }));
  const surface = css("--paper"), all = [];
  groups.forEach((g, gi) => {
    const cx = m.l + band * gi + band / 2;
    svg.append(s("text", { x: cx, y: height - m.b + 18, "text-anchor": "middle", class: "tick-label" }, g.name));
    const w = Math.min(band * 0.55, 90);
    g.points.forEach((p, i) => {
      const jitter = p.noMean ? 0 : ((i * 0.618) % 1 - 0.5) * w;
      const px = cx + jitter, py = y(p.y);
      const c = p.hollow ? s("circle", { cx: px, cy: py, r: 4, fill: surface, stroke: p.color, "stroke-width": 1.6 })
        : s("circle", { cx: px, cy: py, r: 4.5, fill: p.color, stroke: surface, "stroke-width": 1.5, opacity: .9 });
      svg.append(c);
      if (p.mark) svg.append(s("text", { x: px + 8, y: py + 4, class: "data-label strong" }, p.mark));
      all.push({ px, py, p, c });
    });
    if (means && g.points.some((p) => !p.noMean)) {
      const pts = g.points.filter((p) => !p.noMean);
      const mu = pts.reduce((a, p) => a + p.y, 0) / pts.length;
      svg.append(s("line", { x1: cx - w / 2 - 6, x2: cx + w / 2 + 6, y1: y(mu), y2: y(mu), stroke: css("--ink"), "stroke-width": 2 }));
    }
  });
  const hit = s("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, class: "hit", style: onClick ? "cursor:pointer" : null });
  svg.append(hit);
  const nearest = (evt) => { let best = null, bd = 22 * 22; for (const a of all) { const d = (a.px - evt.offsetX) ** 2 + (a.py - evt.offsetY) ** 2; if (d < bd) { bd = d; best = a; } } return best; };
  hit.addEventListener("pointermove", (evt) => { const a = nearest(evt); if (!a) return hideTip(); showTip(evt, a.p.label, [{ color: a.p.color, value: fmt(a.p.y, 2), label: yLabel }, ...(a.p.tip || [])]); });
  hit.addEventListener("pointerleave", hideTip);
  if (onClick) hit.addEventListener("click", (evt) => { const a = nearest(evt); if (a) onClick(a.p); });
  legend(el, [...(legendItems || []), ...(means ? [{ name: "média", color: css("--ink"), kind: "line" }] : [])]);
}

/** Barras horizontais (aceita negativos). */
function hbar(el, { items, xLabel, valueFmt = (v) => fmt(v, 3) }) {
  const rowH = 28, m = { t: 8, b: 38, r: 56 };
  const height = m.t + m.b + items.length * rowH;
  const { svg, width } = frame(el, height);
  const ml = Math.min(Math.max(...items.map((i) => i.label.length)) * 6.8 + 14, width * 0.48);
  const t = niceTicks(Math.min(0, ...items.map((i) => i.value)), Math.max(0, ...items.map((i) => i.value)), 5);
  const x = scale(t[0], t.at(-1), ml, width - m.r);
  for (const v of t) {
    svg.append(s("line", { x1: x(v), x2: x(v), y1: m.t, y2: height - m.b, class: "gridline" }));
    svg.append(s("text", { x: x(v), y: height - m.b + 16, "text-anchor": "middle", class: "tick-label" }, fmt(v, decimals(t))));
  }
  svg.append(s("text", { x: (ml + width - m.r) / 2, y: height - 4, "text-anchor": "middle", class: "axis-label" }, xLabel));
  svg.append(s("line", { x1: x(0), x2: x(0), y1: m.t, y2: height - m.b, class: "baseline" }));
  items.forEach((it, i) => {
    const cy = m.t + i * rowH + rowH / 2, bh = 14;
    const x0 = Math.min(x(0), x(it.value)), w = Math.max(1, Math.abs(x(it.value) - x(0))), r = Math.min(3, w / 2);
    const d = it.value >= 0
      ? `M${x0},${cy - bh / 2}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${bh - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`
      : `M${x0 + w},${cy - bh / 2}h${-(w - r)}a${r},${r} 0 0 0 ${-r},${r}v${bh - 2 * r}a${r},${r} 0 0 0 ${r},${r}h${w - r}z`;
    const color = it.color || (it.highlight ? css("--series-1") : css("--neutral"));
    const bar = s("path", { d, fill: color });
    svg.append(bar,
      s("text", { x: ml - 10, y: cy + 4, "text-anchor": "end", class: it.highlight ? "data-label strong" : "data-label" }, it.label),
      s("text", { x: it.value >= 0 ? x(it.value) + 6 : x(it.value) - 6, y: cy + 4, "text-anchor": it.value >= 0 ? "start" : "end", class: "data-label" }, valueFmt(it.value)));
    const hit = s("rect", { x: 0, y: cy - rowH / 2, width, height: rowH, class: "hit" });
    svg.append(hit);
    hit.addEventListener("pointermove", (evt) => { bar.classList.add("mark-hover"); showTip(evt, it.label, [{ color, value: valueFmt(it.value), label: xLabel }, ...(it.note ? [{ value: it.note }] : [])]); });
    hit.addEventListener("pointerleave", () => { bar.classList.remove("mark-hover"); hideTip(); });
  });
}

/** Mapa de calor sequencial em [0, 1]. */
function heatmap(el, { rows, cols, value, colLabel = (c) => c, label = (v) => fmt(v, 2), title = "R²", legendLabel }) {
  const steps = SEQ();
  const w0 = Math.max(280, el.clientWidth);
  const lw = Math.min(Math.max(...rows.map((r) => r.length)) * 6.8 + 12, w0 * 0.34);
  const cw = Math.max(48, (w0 - lw - 4) / cols.length), chh = 28, head = 44;
  const height = head + rows.length * chh + 6;
  const { svg } = frame(el, height);
  cols.forEach((c, j) => svg.append(s("text", { x: lw + j * cw + cw / 2, y: head - 10, "text-anchor": "middle", class: "tick-label" }, colLabel(c))));
  rows.forEach((r, i) => {
    const cy = head + i * chh;
    svg.append(s("text", { x: lw - 10, y: cy + chh / 2 + 4, "text-anchor": "end", class: "data-label" }, r));
    cols.forEach((c, j) => {
      const v = value(r, c), cx = lw + j * cw;
      if (v == null) return;
      const k = Math.max(0, Math.min(6, Math.floor(Math.max(0, v) * 7)));
      const rect = s("rect", { x: cx + 1, y: cy + 1, width: cw - 2, height: chh - 2, fill: steps[k] });
      svg.append(rect, s("text", { x: cx + cw / 2, y: cy + chh / 2 + 4, "text-anchor": "middle", "font-size": 11.5, fill: k >= 3 ? "#fff" : "#1b1b1a", "pointer-events": "none" }, label(v, r, c)));
      rect.addEventListener("pointermove", (evt) => showTip(evt, `${r} × ${colLabel(c)}`, [{ value: label(v, r, c), label: title }]));
      rect.addEventListener("pointerleave", hideTip);
    });
  });
  legend(el, steps.map((c, i) => ({ name: `${fmt(i / 7, 2)}–${fmt((i + 1) / 7, 2)}`, color: c, kind: "rect" })));
  if (legendLabel) el.querySelector(".legend").prepend(h("span", {}, legendLabel));
}

// botão "ver dados" de cada figura → tabela logo abaixo do gráfico
document.addEventListener("click", (evt) => {
  const btn = evt.target.closest("[data-table-for]");
  if (!btn) return;
  const id = btn.dataset.tableFor, fig = btn.closest("figure") || btn.closest("section");
  const open = fig.querySelector(`.chart-table[data-for="${id}"]`);
  if (open) { open.remove(); btn.textContent = "ver dados"; return; }
  const t = chartTables.get(id);
  if (!t) return;
  $(`#${id}`).after(h("div", { class: "chart-table", "data-for": id }, localTable(t.columns, t.rows, t)));
  btn.textContent = "ocultar dados";
});

// ─────────────────────────────────────────────────────────────────────────────
// markdown (relatórios)
// ─────────────────────────────────────────────────────────────────────────────
function mdInline(text, parent) {
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\$[^$]+\$)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) parent.append(text.slice(last, m.index));
    const t = m[0];
    parent.append(t.startsWith("**") ? h("strong", {}, t.slice(2, -2)) : t.startsWith("`") ? h("code", {}, t.slice(1, -1))
      : t.startsWith("$") ? h("em", {}, t.slice(1, -1).replace(/\^2/g, "²").replace(/\\approx/g, "≈").replace(/\\/g, "")) : h("em", {}, t.slice(1, -1)));
    last = m.index + t.length;
  }
  if (last < text.length) parent.append(text.slice(last));
  return parent;
}
function renderMarkdown(md) {
  const root = h("div", { class: "md" }), lines = md.split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^#{1,4} /.test(line)) { const n = line.match(/^#+/)[0].length; root.append(mdInline(line.slice(n + 1), h(`h${Math.min(n + 1, 5)}`))); i++; continue; }
    if (/^\s*\|/.test(line)) {
      const rows = []; while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const [head, ...rest] = rows.filter((r) => !/^\s*\|[\s:|-]+\|\s*$/.test(r));
      root.append(h("table", {}, h("thead", {}, h("tr", {}, cells(head).map((c) => mdInline(c, h("th"))))), h("tbody", {}, rest.map((r) => h("tr", {}, cells(r).map((c) => mdInline(c, h("td"))))))));
      continue;
    }
    if (/^\s*([-*]|\d+\.) /.test(line)) {
      const list = h(/^\s*\d+\./.test(line) ? "ol" : "ul");
      while (i < lines.length && /^\s*([-*]|\d+\.) /.test(lines[i])) list.append(mdInline(lines[i++].replace(/^\s*([-*]|\d+\.) /, ""), h("li")));
      root.append(list); continue;
    }
    if (/^!\[/.test(line)) { root.append(h("p", { class: "muted small" }, `[gráfico: ${line.match(/^!\[([^\]]*)\]/)[1]}]`)); i++; continue; }
    if (/^---+\s*$/.test(line)) { root.append(h("hr")); i++; continue; }
    if (line.startsWith("```")) { const code = []; i++; while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]); i++; root.append(h("pre", {}, h("code", {}, code.join("\n")))); continue; }
    if (!line.trim()) { i++; continue; }
    const para = []; while (i < lines.length && lines[i].trim() && !/^(#|\s*\||\s*[-*] |\s*\d+\. |!\[|---|```)/.test(lines[i])) para.push(lines[i++]);
    root.append(mdInline(para.join(" "), h("p")));
  }
  return root;
}

// ─────────────────────────────────────────────────────────────────────────────
// roteamento: #/pagina[/parametro]
// ─────────────────────────────────────────────────────────────────────────────
const PAGES = [
  { id: "", title: "Resumo" },
  { id: "campo", title: "Campo" },
  { id: "coletas", title: "Coletas" },
  { id: "classes", title: "Classes" },
  { id: "analise", title: "Análise" },
  { id: "vegetativo", title: "Altura e massa" },
  { id: "ciencia", title: "Aprofundamento" },
  { id: "outliers", title: "Outliers" },
  { id: "imagens", title: "Imagens" },
  { id: "dados", title: "Dados" },
];
const pageRender = {};   // preenchido por pages.js: (view, param, overview) => Promise

function fillKeys(root, key) {
  for (const el of $$("[data-k]", root)) { const v = key[el.dataset.k]; el.textContent = v == null ? "n/d" : fmt(v, el.dataset.fmt); }
}
function setupTabs(root, initial) {
  for (const tabs of $$(".tabs", root)) {
    const buttons = $$("button[data-tab]", tabs), scope = tabs.parentElement;
    const show = (name) => {
      buttons.forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
      $$(".tab-panel", scope).forEach((p) => { p.hidden = p.dataset.panel !== name; });
      charts.forEach((r, id) => { if (document.getElementById(id)?.offsetParent) r(); });
    };
    buttons.forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
    show(buttons.some((b) => b.dataset.tab === initial) ? initial : buttons[0].dataset.tab);
  }
}
function serverMissing(err) {
  return h("div", { class: "error-box" }, h("strong", {}, "Não foi possível consultar o banco"),
    `${err.message} No terminal, dentro do repositório, rode `, h("code", {}, "python run.py site"), " e abra ", h("code", {}, "http://localhost:8000"), ".");
}

let navToken = 0, currentPage = null;
async function route() {
  const [, page = "", param = null] = location.hash.match(/^#\/?([\w-]*)(?:\/([\w-]+))?/) || [];
  const known = PAGES.some((p) => p.id === page) ? page : "";
  $("#nav").replaceChildren(...PAGES.filter((p) => p.id).map((p) => h("a", { href: `#/${p.id}`, class: p.id === known ? "on" : "" }, p.title)));
  const view = $("#view"), token = ++navToken;
  if (location.protocol === "file:") { view.replaceChildren(serverMissing(new Error("O arquivo foi aberto direto do disco."))); return; }
  // mesma página, outro parâmetro (ex.: outra parcela no Campo): só atualiza
  if (currentPage === known && pageRender[known]?.update) { await pageRender[known].update(view, param); return; }
  charts.clear(); chartTables.clear(); hideTip();
  currentPage = known;
  try {
    const res = await fetch(`pages/${known || "resumo"}.html`, { cache: "no-store" });
    if (!res.ok) throw new ServerMissing("A página foi aberta sem o servidor do site.");
    const html = await res.text();
    if (token !== navToken) return;
    view.innerHTML = html;
    document.title = `${PAGES.find((p) => p.id === known).title} · Clorofila por Imagem`;
    scrollTo(0, 0);
    const ov = await cached("/api/overview");
    fillKeys(view, ov.key);
    setupTabs(view, param);
    await pageRender[known]?.(view, param, ov);
    $("#stamp").textContent = `${fmt(ov.key.n_falker)} leituras Falker · ${ov.key.n_samples} fotos com referência`;
    if (STATIC && !$("#static-note")) {
      const meta = await fetchJson("data/meta.json").catch(() => null);
      $(".colophon span").replaceWith(h("span", { id: "static-note" }, `Versão estática gerada em ${meta?.gerado_em ?? "–"} a partir de `, h("code", {}, "database/tcc.sqlite"), "."));
    }
  } catch (err) {
    console.error(err);
    currentPage = null;
    view.prepend(err instanceof ServerMissing ? serverMissing(err) : h("div", { class: "error-box" }, h("strong", {}, "Erro ao montar a página"), err.message));
  }
}

function setupTheme() {
  let saved = null;
  try { saved = localStorage.getItem("tcc-theme"); } catch { /* sem armazenamento */ }
  if (saved) document.documentElement.dataset.theme = saved;
  const current = () => document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const label = () => { $("#theme").textContent = current() === "dark" ? "Tema claro" : "Tema escuro"; };
  $("#theme").addEventListener("click", () => {
    document.documentElement.dataset.theme = current() === "dark" ? "light" : "dark";
    try { localStorage.setItem("tcc-theme", document.documentElement.dataset.theme); } catch { /* ok */ }
    label(); charts.forEach((r) => r()); pageRender[currentPage]?.retheme?.();
  });
  label();
}
addEventListener("hashchange", route);
document.addEventListener("DOMContentLoaded", () => { setupTheme(); route(); });
