/* Montagem de cada página (site/pages/<id>.html). Depende de app.js. */
"use strict";

const TRAT = ["T1", "T2", "T3", "T4"];
const DOSE_OF = { T1: 0, T2: 50, T3: 75, T4: 100 };
const doseColor = (dose) => SEQ()[DOSE_STEP[dose] ?? 3];
const plotLabel = (id) => (String(id).startsWith("teste-") ? `ponto de teste ${String(id).split("-")[1]}` : `parcela ${id}`);
const goPlot = (id) => { if (id) location.hash = `#/campo/${id}`; };

function options(el, items, onChange, initial) {
  let current = initial ?? items[0][0];
  const draw = () => el.replaceChildren(...items.map(([v, label]) => h("button", {
    type: "button", class: v === current ? "on" : "", onclick: () => { current = v; draw(); onChange(v); },
  }, label)));
  draw();
  onChange(current);
}
function dateLegend() {
  return [...DATES.map((d) => ({ name: shortDate(d), color: dateColor(d) })), { name: "01/06 (teste)", color: css("--neutral"), kind: "ring" }];
}
function shot(rel, title, rows, { width = 500, caption } = {}) {
  const frame = rel
    ? h("button", { type: "button", class: "frame", "aria-label": `Ampliar ${title}`, onclick: () => openPhoto(rel, title, caption) },
      h("img", { src: photoUrl(rel, width), alt: title, loading: "lazy" }))
    : h("div", { class: "frame empty" }, "sem foto nesta data");
  return h("figure", { class: "shot" }, frame, h("figcaption", {}, h("div", { class: "d" }, title),
    rows?.length ? h("dl", {}, rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])) : null));
}

// ─────────────────────────────────────────────────────────────────────────────
// Resumo
// ─────────────────────────────────────────────────────────────────────────────
pageRender[""] = async (view) => {
  const field = await cached("/api/field");
  const fig = $("#overview-photo", view);
  if (field.overview_photo) {
    fig.replaceChildren(
      h("img", { src: photoUrl(field.overview_photo, 1800), alt: "Vista de cima da área experimental", style: "cursor:zoom-in", onclick: () => openPhoto(field.overview_photo, "Vista geral da área experimental") }),
      h("figcaption", {}, "Vista de cima da área experimental, registrada em 01/06 (clique para ampliar)."));
  } else fig.remove();
};

// ─────────────────────────────────────────────────────────────────────────────
// Campo
// ─────────────────────────────────────────────────────────────────────────────
const campo = { field: null, metric: "dose", selected: null };

function metricDefs(field) {
  const all = field.plots.concat(field.tests);
  const obsVals = all.flatMap((p) => Object.values(p.obs).filter((v) => v != null));
  const resVals = all.flatMap((p) => Object.values(p.res).filter((v) => v != null));
  const heights = field.plots.map((p) => p.altura).filter((v) => v != null);
  const seq = (lo, hi) => (v) => SEQ()[Math.max(0, Math.min(6, Math.floor(((v - lo) / (hi - lo || 1)) * 6.999)))];
  const spad = seq(Math.min(...obsVals), Math.max(...obsVals));
  const rmax = Math.max(...resVals.map(Math.abs));
  const div = (v) => {   // divergente: azul (subestima) · cinza · vermelho (superestima)
    const t = Math.max(-1, Math.min(1, v / rmax));
    const mix = (a, b, k) => { const pa = a.match(/\w\w/g).map((x) => parseInt(x, 16)), pb = b.match(/\w\w/g).map((x) => parseInt(x, 16)); return `rgb(${pa.map((c, i) => Math.round(c + (pb[i] - c) * k)).join(",")})`; };
    return t < 0 ? mix(css("--div-mid"), css("--div-neg"), -t) : mix(css("--div-mid"), css("--div-pos"), t);
  };
  const meanRes = (p) => { const v = Object.values(p.res).filter((x) => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  const defs = {
    dose: { label: "Dose de N", value: (p) => (p.Dose ?? null), color: (v) => doseColor(v), show: (v) => `${v} kg`, scale: [0, 50, 75, 100].map((d) => [doseColor(d), `${d}`]), unit: "kg N/ha" },
    res: { label: "Erro da rede (média)", value: meanRes, color: div, show: (v) => fmt(v, "+"), scale: "div", lo: -rmax, hi: rmax, unit: "SPAD (previsto − medido)" },
    altura: { label: "Altura 26/05", value: (p) => p.altura ?? null, color: seq(Math.min(...heights), Math.max(...heights)), show: (v) => fmt(v, 0), scale: "seq", lo: Math.min(...heights), hi: Math.max(...heights), unit: "cm" },
  };
  for (const d of DATES) defs[d] = { label: `SPAD ${shortDate(d)}`, value: (p) => p.obs[d] ?? null, color: spad, show: (v) => fmt(v, 1), scale: "seq", lo: Math.min(...obsVals), hi: Math.max(...obsVals), unit: "SPAD" };
  defs.test = { label: "", value: (p) => p.obs[TEST] ?? null, color: spad, show: (v) => fmt(v, 1) };
  return defs;
}

function drawMap(view) {
  const f = campo.field, defs = metricDefs(f), def = defs[campo.metric];
  const grid = $("#plot-grid", view);
  const tile = (p, valueFn = def.value, colorFn = def.color, showFn = def.show) => {
    const v = valueFn(p);
    const bg = v == null ? css("--paper-2") : colorFn(v);
    const dark = v != null && isDark(bg);
    const test = String(p.id).startsWith("teste-");
    return h("button", {
      type: "button", class: `plot ${campo.selected === p.id ? "sel" : ""}`, style: `background:${bg};color:${v == null ? css("--ink-3") : dark ? "#fff" : "#1b1b1a"}`,
      "aria-label": `${test ? "Ponto de teste" : "Parcela"} ${p.Ponto}${p.Tratamento ? `, ${p.Tratamento}` : ""}`, "aria-pressed": campo.selected === p.id,
      onclick: () => goPlot(p.id),
    }, h("span", { class: "pid" }, test ? `t${p.Ponto}` : `P${p.Ponto}`),
    p.flags ? h("span", { class: "flag", title: `${p.flags} alerta(s)` }, "!") : null,
    h("span", {}, h("div", { class: "pval" }, v == null ? "–" : showFn(v)),
      h("div", { class: "ptr" }, test ? "01/06" : campo.metric === "dose" ? p.Tratamento : `${p.Tratamento} · ${p.Dose} kg`)));
  };
  const byBlock = {};
  for (const p of f.plots) (byBlock[p.Bloco] ??= []).push(p);
  grid.replaceChildren(h("span"), ...[1, 2, 3, 4].map((i) => h("span", { class: "collab" }, `posição ${i}`)));
  for (const [bloco, plots] of Object.entries(byBlock)) {
    grid.append(h("span", { class: "rowlab" }, bloco));
    for (const p of plots.sort((a, b) => a.Ponto - b.Ponto)) grid.append(tile(p));
  }
  // pontos de teste: SPAD de 01/06 nas métricas de SPAD, erro na métrica de erro, vazio nas demais
  const testTile = (p) => DATES.includes(campo.metric) ? tile(p, defs.test.value, defs.test.color, defs.test.show)
    : campo.metric === "res" ? tile(p) : tile(p, () => null);
  $("#test-row", view).replaceChildren(h("span", { class: "rowlab" }, "Teste"), ...f.tests.map(testTile));
  // escala
  const sc = $("#map-scale", view);
  if (Array.isArray(def.scale)) sc.replaceChildren(...def.scale.map(([c, l]) => h("span", { style: "display:inline-flex;gap:6px;align-items:center" }, h("span", { class: "key-rect", style: `background:${c}` }), l)), h("span", {}, def.unit));
  else {
    const stops = def.scale === "div" ? [def.color(def.lo), def.color(0), def.color(def.hi)] : SEQ();
    sc.replaceChildren(h("span", {}, def.scale === "div" ? fmt(def.lo, "+") : fmt(def.lo, 0)),
      h("span", { class: "bar", style: `background:linear-gradient(90deg,${stops.join(",")})` }),
      h("span", {}, def.scale === "div" ? fmt(def.hi, "+") : fmt(def.hi, 0)), h("span", {}, def.unit));
  }
}
function isDark(color) {
  const m = color.match(/\d+/g) || [];
  let r, g, b;
  if (color.startsWith("#")) [r, g, b] = color.slice(1).match(/\w\w/g).map((x) => parseInt(x, 16));
  else [r, g, b] = m.map(Number);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b < 140;
}

async function drawDetail(view) {
  const box = $("#detail", view);
  const id = campo.selected;
  box.style.opacity = ".5";
  let d;
  try { d = await cached(`/api/plot/${id}`); } catch (err) { box.replaceChildren(h("p", { class: "error-box" }, err.message)); box.style.opacity = "1"; return; }
  if (campo.selected !== id) return;
  const withPhoto = d.dates.filter((x) => x.rel_path);
  const obs = d.dates.filter((x) => x.obs != null);
  const errs = d.dates.filter((x) => x.res != null);
  const mae = errs.length ? errs.reduce((a, x) => a + Math.abs(x.res), 0) / errs.length : null;

  const title = d.teste ? `Ponto de teste ${d.Ponto}` : `Parcela ${d.Ponto}`;
  const meta = d.teste ? "Área fora do delineamento, medida só em 01/06 (sem dose controlada)"
    : `${d.layout.Bloco} · tratamento ${d.layout.Tratamento} · ${d.layout.Dose} kg N/ha`;
  let summary = "";
  if (obs.length > 1) {
    const a = obs[0], b = obs.at(-1);
    summary = `O SPAD medido foi de ${fmt(a.obs, 1)} em ${shortDate(a.Data)} para ${fmt(b.obs, 1)} em ${shortDate(b.Data)}. `;
  } else if (obs.length) summary = `SPAD medido: ${fmt(obs[0].obs, 1)} (média de ${obs[0].n_leituras} leituras). `;
  if (mae != null) summary += errs.length > 1
    ? `A rede neural errou em média ${fmt(mae, 1)} SPAD nas ${errs.length} fotos desta ${d.teste ? "área" : "parcela"}`
    : `Na única foto desta ${d.teste ? "área" : "parcela"}, a rede neural errou ${fmt(errs[0].res, "+")} SPAD`;
  if (withPhoto.length < d.dates.length) summary += `; ${d.dates.length - withPhoto.length === 1 ? "uma data ficou" : "algumas datas ficaram"} sem foto`;
  summary += ".";

  const shots = h("div", { class: `strip ${d.dates.length === 1 ? "one" : ""}` }, d.dates.map((x) => shot(x.rel_path, shortDate(x.Data), [
    ["Falker", x.obs != null ? `${fmt(x.obs, 1)} (${x.n_leituras} leit.)` : "–"],
    ["Rede neural", fmt(x.Predicao_MLP, 1)],
    ["Erro", x.res != null ? fmt(x.res, "+") : "–"],
    ["App", fmt(x.app_pred, 1)],
    ["Cobertura", x.Canopy_Cover != null ? `${fmt(x.Canopy_Cover, 0)}%` : "–"],
  ], { caption: `${title}, ${shortDate(x.Data)}: Falker ${fmt(x.obs, 1)} SPAD, rede neural ${fmt(x.Predicao_MLP, 1)}.` })));

  const readChart = h("div", { class: "chart", id: "plot-readings" });
  const tableRows = d.dates.map((x) => ({
    Data: shortDate(x.Data), Leituras: x.n_leituras, Media: x.obs, DP: x.sd, CV: x.cv != null ? x.cv * 100 : null,
    Cobertura: x.Canopy_Cover, Blocos: x.Vegetation_Blocks, Rede: x.Predicao_MLP, Indice: x.Predicao_Melhor_Indice, App: x.app_pred,
  }));

  const parts = [
    h("h2", {}, title), h("p", { class: "meta" }, meta), h("p", { class: "prose", style: "margin-top:14px" }, summary),
    h("h3", {}, "Fotos"), h("p", { class: "small muted" }, "Clique para ampliar e ver as etapas do processamento."), shots,
    h("h3", {}, "Leituras do clorofilômetro"),
    h("p", { class: "small muted" }, "Cada ponto é uma folha medida; o traço é a média; o círculo vazado é a predição da rede para a foto daquela data."),
    readChart,
    h("div", {}, localTable(Object.keys(tableRows[0]), tableRows, {
      formats: { Media: 2, DP: 2, CV: 1, Cobertura: 1, Rede: 2, Indice: 2, App: 1 },
      labels: { Media: "Média", CV: "CV (%)", Cobertura: "Cobertura (%)", Rede: "Rede neural", Indice: "Índice r/b" },
    })),
  ];

  if (d.vegetacao) {
    const v = d.vegetacao, mt = v.media_tratamento || {};
    parts.push(h("h3", {}, "Crescimento em 26/05"),
      h("div", { class: "kv" },
        h("div", {}, h("div", { class: "k" }, "Altura média"), h("div", { class: "v" }, fmt(v.Altura_Media, 1), h("small", {}, "cm")), h("div", { class: "k" }, `média do ${d.layout.Tratamento}: ${fmt(mt.Altura_Media, 1)} cm`)),
        h("div", {}, h("div", { class: "k" }, "Massa seca total"), h("div", { class: "v" }, fmt(v.Massa_Seca_Total_ha, 2), h("small", {}, "t/ha")), h("div", { class: "k" }, `média do ${d.layout.Tratamento}: ${fmt(mt.Massa_Seca_Total_ha, 2)} t/ha`)),
        h("div", {}, h("div", { class: "k" }, "Massa seca de folha"), h("div", { class: "v" }, fmt(v.Massa_Seca_Folha_ha, 2), h("small", {}, "t/ha"))),
        h("div", {}, h("div", { class: "k" }, "Relação folha/colmo"), h("div", { class: "v" }, fmt(v.Rel_Folha_Colmo_Seco, 2), h("small", {}, "seca")))),
      h("p", { class: "small muted", style: "margin-top:8px" }, `Alturas anotadas: ${v["Alturas Anotadas (cm)"]} cm.`));
  }

  if (withPhoto.length) {
    const stepsBox = h("div", { class: "steps-row" });
    const opts = h("div", { class: "options" });
    parts.push(h("h3", {}, "O que o algoritmo vê"),
      h("p", { class: "small muted" }, "As etapas aplicadas à foto antes de chegar à rede neural."), opts, stepsBox);
    options(opts, withPhoto.map((x) => [x.rel_path, shortDate(x.Data)]), (rel) => {
      stepsBox.replaceChildren(...["crop", "pre", "mask", "blocks"].map((st) => h("figure", {},
        h("img", { src: stepUrl(rel, st, 420), alt: STEP_INFO[st][0], loading: "lazy", style: "cursor:zoom-in", onclick: () => openPhoto(rel, title) }),
        h("figcaption", {}, h("b", {}, STEP_INFO[st][0]), STEP_INFO[st][1]))));
    }, withPhoto.at(-1).rel_path);
  }

  if (d.mapa?.image) {
    parts.push(h("h3", {}, "Mapa de clorofila (26/05)"),
      h("figure", { class: "wide-photo", style: "margin-top:12px" }, h("img", { src: d.mapa.image, alt: "Mapa de clorofila", loading: "lazy" }),
        h("figcaption", {}, `Média do mapa ${fmt(d.mapa.SPAD_Medio_Mapa, 1)} SPAD (desvio ${fmt(d.mapa.SPAD_Desvio, 1)}) contra ${fmt(d.mapa.Clorofila_Predita, 1)} previstos para a parcela inteira. Áreas vermelhas indicam blocos com menos clorofila.`)));
  }

  parts.push(h("h3", {}, "Alertas"));
  if (d.outliers.length) parts.push(...d.outliers.map((o) => caseItem(o, false)));
  else parts.push(h("p", { class: "muted" }, "Nenhum critério de outlier foi disparado para este ponto."));

  parts.push(h("details", { class: "report", style: "margin-top:28px" }, h("summary", {}, `Todas as ${d.readings.length} leituras (dados brutos)`),
    localTable(["data", "hora", "medicao", "clorofila_a", "clorofila_b", "clorofila_total"], d.readings,
      { labels: { data: "Data", hora: "Hora", medicao: "Medição", clorofila_a: "Clorofila A", clorofila_b: "Clorofila B", clorofila_total: "Total" }, formats: { medicao: 0, clorofila_a: 1, clorofila_b: 1, clorofila_total: 1 } })));

  box.replaceChildren(h("div", {}, ...parts));
  box.style.opacity = "1";

  registerChart("plot-readings", () => strip(readChart, {
    yLabel: "SPAD",
    height: 280,
    groups: d.dates.map((x) => ({
      name: shortDate(x.Data),
      points: [
        ...d.readings.filter((r) => r.data === x.Data).map((r) => ({ y: r.clorofila_total, color: dateColor(x.Data), label: `${shortDate(x.Data)} · ${r.hora}`, tip: [{ value: `A ${fmt(r.clorofila_a, 1)} · B ${fmt(r.clorofila_b, 1)}` }] })),
        ...(x.Predicao_MLP != null ? [{ y: x.Predicao_MLP, hollow: true, noMean: true, color: css("--ink"), label: `${shortDate(x.Data)} · predição da rede` }] : []),
      ],
    })),
    legendItems: [{ name: "predição da rede", color: css("--ink"), kind: "ring" }],
  }));
}

pageRender.campo = async (view, param) => {
  campo.field = await cached("/api/field");
  campo.selected = param || campo.selected || "5";
  options($("#color-by", view), [["dose", "Dose"], ...DATES.map((d) => [d, `SPAD ${shortDate(d)}`]), ["res", "Erro da rede"], ["altura", "Altura"]],
    (m) => { campo.metric = m; drawMap(view); }, campo.metric);
  await drawDetail(view);
};
pageRender.campo.update = async (view, param) => {
  campo.selected = param || campo.selected;
  drawMap(view);
  charts.clear(); chartTables.clear();
  await drawDetail(view);
  if (innerWidth < 980) $("#detail", view).scrollIntoView({ behavior: "smooth" });
};
pageRender.campo.retheme = () => { const v = $("#view"); if ($("#plot-grid", v)) drawMap(v); };

// ─────────────────────────────────────────────────────────────────────────────
// Análise
// ─────────────────────────────────────────────────────────────────────────────
pageRender.analise = async (view, _p, ov) => {
  const fig = await cached("/api/figures");
  const imgs = fig.images;
  $("#date-legend", view).replaceChildren(...dateLegend().map((it, i) => h("span", { style: "white-space:nowrap" }, i ? " · " : "",
    h("span", { class: it.kind === "ring" ? "key-ring" : "key-dot", style: `${it.kind === "ring" ? "border-color" : "background"}:${it.color};margin-right:6px;vertical-align:1px` }), it.name)));
  const imgLabel = (r) => `${shortDate(r.Data)} · ${plotLabel(r.id)}${r.Tratamento ? ` (${r.Tratamento})` : ""}`;
  const imgPoint = (r, x, y) => ({ x, y, color: dateColor(r.Data), hollow: r.Data === TEST, label: imgLabel(r), id: r.id });

  // 1 dose
  registerChart("f-dose", () => lineChart($("#f-dose", view), {
    series: DATES.map((d) => ({ name: shortDate(d), color: dateColor(d), points: ov.dose_response.filter((r) => r.Data === d).map((r) => ({ x: r.Dose, y: r.Media, e: r.EP })) })),
    xTicks: [0, 50, 75, 100], xLabel: "Dose de N (kg/ha)", yLabel: "SPAD (média ± erro padrão)", xFmt: (v) => `${v}`,
  }), { columns: ["Data", "Dose", "n", "Media", "EP"], rows: ov.dose_response, formats: { Media: 2, EP: 2 }, labels: { Media: "SPAD médio", EP: "erro padrão", n: "leituras" } });

  // 2 leituras por tratamento
  const readings = fig.readings.filter((r) => r.Tratamento);
  options($("#f-readings-date", view), [["all", "todas as datas"], ...DATES.map((d) => [d, shortDate(d)])], (sel) => {
    const rows = readings.filter((r) => sel === "all" || r.Data === sel);
    registerChart("f-readings", () => strip($("#f-readings", view), {
      yLabel: "SPAD (leitura individual)", height: 320,
      groups: TRAT.map((t) => ({ name: `${t} · ${DOSE_OF[t]} kg`, points: rows.filter((r) => r.Tratamento === t).map((r) => ({ y: r.v, color: dateColor(r.Data), label: `${shortDate(r.Data)} · parcela ${r.Ponto}`, id: String(r.Ponto) })) })),
      legendItems: DATES.map((d) => ({ name: shortDate(d), color: dateColor(d) })), onClick: (p) => goPlot(p.id),
    }), { columns: ["Data", "Ponto", "Tratamento", "Dose", "v"], rows, formats: { v: 1 }, labels: { v: "SPAD" } });
  });

  // 3 índice r/b
  const withIdx = imgs.filter((r) => r.Median_r_over_b != null);
  registerChart("f-index", () => scatter($("#f-index", view), {
    points: withIdx.map((r) => ({ ...imgPoint(r, r.Median_r_over_b, r.obs), tip: [{ value: fmt(r.Median_r_over_b, 3), label: "r/b" }] })),
    xLabel: "Razão r/b (mediana dos blocos de vegetação)", yLabel: "SPAD medido", legendItems: dateLegend(), onClick: (p) => goPlot(p.id), xFmt: (v) => fmt(v, 1),
  }), { columns: ["Data", "Ponto", "Median_r_over_b", "obs"], rows: withIdx, formats: { Median_r_over_b: 3, obs: 2 }, labels: { obs: "SPAD" } });

  // 4 observado × previsto
  const pred = imgs.filter((r) => r.Predicao_MLP != null);
  const predTable = { columns: ["Data", "Ponto", "Tratamento", "obs", "Predicao_MLP", "Predicao_Melhor_Indice", "res"], rows: pred, formats: { obs: 2, Predicao_MLP: 2, Predicao_Melhor_Indice: 2, res: 2 }, labels: { obs: "Falker", Predicao_MLP: "Rede", Predicao_Melhor_Indice: "Índice", res: "Erro da rede" } };
  registerChart("f-mlp", () => scatter($("#f-mlp", view), { points: pred.map((r) => imgPoint(r, r.obs, r.Predicao_MLP)), xLabel: "SPAD medido", yLabel: "SPAD previsto", identity: true, square: true, height: 330, onClick: (p) => goPlot(p.id), legendItems: dateLegend() }), predTable);
  registerChart("f-idx", () => scatter($("#f-idx", view), { points: pred.map((r) => imgPoint(r, r.obs, r.Predicao_Melhor_Indice)), xLabel: "SPAD medido", yLabel: "SPAD previsto", identity: true, square: true, height: 330, onClick: (p) => goPlot(p.id) }));

  // 5 erro por data
  const rmse = ov.key.mlp_rmse;
  const allDates = [...DATES, TEST];
  registerChart("f-res", () => strip($("#f-res", view), {
    yLabel: "Erro da rede (SPAD)", zeroLine: true, height: 320,
    groups: allDates.map((d) => ({ name: d === TEST ? "01/06 (teste)" : shortDate(d), points: pred.filter((r) => r.Data === d).map((r) => ({
      y: r.res, color: dateColor(d), hollow: d === TEST, label: imgLabel(r), id: r.id, mark: Math.abs(r.res) > 1.5 * rmse ? (d === TEST ? `t${r.Ponto}` : `P${r.Ponto}`) : null,
      tip: [{ value: `${fmt(r.obs, 1)} → ${fmt(r.Predicao_MLP, 1)}`, label: "medido → previsto" }] })) })),
    onClick: (p) => goPlot(p.id),
  }), { columns: ["Data", "n", "vies_modelo", "mae_modelo", "obs_media", "hora"], rows: fig.date_stats, formats: { vies_modelo: 2, mae_modelo: 2, obs_media: 1 }, labels: { vies_modelo: "viés", mae_modelo: "erro médio abs.", obs_media: "SPAD médio", hora: "hora da 1ª foto" } });
  const ds = Object.fromEntries(fig.date_stats.map((r) => [r.Data, r]));
  $("#f-res-text", view).replaceChildren(
    "Erro = previsto − medido; o traço é o viés (erro médio) de cada data. ",
    h("strong", {}, `18/05 fica centrada em zero (viés ${fmt(ds["18-05-2026"]?.vies_modelo, "+")})`),
    `, 26/05 levemente acima (${fmt(ds["26-05-2026"]?.vies_modelo, "+")}), `,
    h("strong", {}, `21/05 abaixo (${fmt(ds["21-05-2026"]?.vies_modelo, "+")})`), " e ",
    h("strong", {}, `a área de teste bem acima (${fmt(ds[TEST]?.vies_modelo, "+")})`),
    ". Quando uma data inteira se desloca, a causa está na condição da coleta (luz, umidade, estágio da planta), e não em uma parcela específica. Os rótulos marcam os erros acima de 1,5 × RMSE.");

  // 6 abordagens
  registerChart("f-appr", () => hbar($("#f-appr", view), {
    xLabel: "R² fora da amostra", items: ov.approaches.map((a) => ({ label: a.Abordagem, value: a.R2, highlight: !!a.Principal, note: `${a.Validacao} · ${a.Fonte}` })),
  }), { columns: ["Abordagem", "R2", "Validacao", "Fonte"], rows: ov.approaches, formats: { R2: 3 }, labels: { Validacao: "Validação", Fonte: "Tabela" } });

  // 7 cobertura × erro
  const cov = pred.filter((r) => r.Canopy_Cover != null);
  registerChart("f-cover", () => scatter($("#f-cover", view), {
    points: cov.map((r) => ({ ...imgPoint(r, r.Canopy_Cover, Math.abs(r.res)), tip: [{ value: fmt(r.res, "+"), label: "erro com sinal" }] })),
    xLabel: "Cobertura vegetal da foto (%)", yLabel: "|erro| da rede (SPAD)", legendItems: dateLegend(), onClick: (p) => goPlot(p.id),
  }), { columns: ["Data", "Ponto", "Canopy_Cover", "Vegetation_Blocks", "res"], rows: cov, formats: { Canopy_Cover: 1, res: 2 }, labels: { Canopy_Cover: "Cobertura (%)", res: "Erro" } });
  const exp = cov.filter((r) => r.Data !== TEST);
  const corr = (a, b) => { const n = a.length, ma = a.reduce((x, y) => x + y, 0) / n, mb = b.reduce((x, y) => x + y, 0) / n; let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } return sab / Math.sqrt(saa * sbb); };
  $("#cover-corr", view).textContent = fmt(corr(exp.map((r) => r.Canopy_Cover), exp.map((r) => Math.abs(r.res))), 2);
  $("#cover-min", view).textContent = fmt(Math.min(...exp.map((r) => r.Canopy_Cover)), 0);
  $("#cover-max", view).textContent = fmt(Math.max(...exp.map((r) => r.Canopy_Cover)), 0);

  // 8 altura × clorofila
  const outl = await cached("/api/outliers");
  const hFlag = new Set(outl.filter((o) => o.categoria === "Altura × clorofila").map((o) => o.Ponto));
  registerChart("f-height", () => scatter($("#f-height", view), {
    points: fig.vegetation.map((r) => ({ x: r["Clorofila Total"], y: r.Altura_Media, color: doseColor(r.Dose), label: `Parcela ${r.Ponto} (${r.Tratamento})`, id: String(r.Ponto), mark: hFlag.has(r.Ponto) ? `P${r.Ponto}` : null,
      tip: [{ value: `${fmt(r.Massa_Seca_Total_ha, 2)} t/ha`, label: "massa seca" }] })),
    xLabel: "SPAD medido em 26/05", yLabel: "Altura média (cm)", fit: true, onClick: (p) => goPlot(p.id),
    legendItems: [0, 50, 75, 100].map((d) => ({ name: `${d} kg N/ha`, color: doseColor(d) })).concat([{ name: "reta ajustada", color: css("--ink-2"), kind: "line" }]),
  }), { columns: ["Ponto", "Tratamento", "Clorofila Total", "Altura_Media", "Massa_Seca_Total_ha", "Massa_Seca_Folha_ha"], rows: fig.vegetation, formats: { "Clorofila Total": 1, Altura_Media: 1, Massa_Seca_Total_ha: 2, Massa_Seca_Folha_ha: 2 } });

  // 9 heatmap
  const reg = await fetchAll("regression_summary");
  const models = ["Linear vs Dose", "Quadratic vs Dose", "Linear vs Clorofila Total", "Quadratic vs Clorofila Total", "Dose + Clorofila Total"];
  const ml = { "Linear vs Dose": "dose", "Quadratic vs Dose": "dose²", "Linear vs Clorofila Total": "clorofila", "Quadratic vs Clorofila Total": "clorofila²", "Dose + Clorofila Total": "dose + clor." };
  options($("#f-veg-level", view), [["Parcela (N=12)", "12 parcelas"], ["Média (N=4)", "4 médias de tratamento"]], (level) => {
    const sub = reg.filter((r) => r.Analysis_Level === level);
    const metrics = [...new Set(sub.map((r) => r.Vegetative_Metric))];
    registerChart("f-veg", () => heatmap($("#f-veg", view), { rows: metrics, cols: models, colLabel: (c) => ml[c], value: (m, c) => sub.find((r) => r.Vegetative_Metric === m && r.Model_Type === c)?.R2 }),
      { columns: ["Vegetative_Metric", "Model_Type", "R2", "Adj_R2"], rows: sub, formats: { R2: 3, Adj_R2: 3 } });
  });

  // 10 app
  const app = imgs.filter((r) => r.app_pred != null);
  registerChart("f-app", () => scatter($("#f-app", view), {
    points: app.map((r) => ({ ...imgPoint(r, r.obs, r.app_pred), mark: Math.abs(r.app_res) > 2.5 ? (r.Data === TEST ? `t${r.Ponto}` : `P${r.Ponto}`) : null })),
    xLabel: "SPAD medido (Falker)", yLabel: "SPAD previsto pelo app", identity: true, square: true, legendItems: dateLegend(), onClick: (p) => goPlot(p.id),
  }), { columns: ["Data", "Ponto", "obs", "app_pred", "app_res"], rows: app, formats: { obs: 2, app_pred: 1, app_res: 2 }, labels: { obs: "Falker", app_pred: "App", app_res: "Erro" } });
  $("#app-out-n", view).textContent = app.filter((r) => Math.abs(r.app_res) > 2.5).length;
};

// ─────────────────────────────────────────────────────────────────────────────
// Outliers
// ─────────────────────────────────────────────────────────────────────────────
function caseItem(o, withLink = true) {
  const where = `${shortDate(o.Data)} · ${plotLabel(o.id)}`;
  return h("article", { class: "case" },
    o.foto ? h("img", { src: photoUrl(o.foto, 260), alt: `Foto de ${where}`, loading: "lazy", onclick: () => openPhoto(o.foto, where, o.titulo) }) : h("div", { class: "noimg" }),
    h("div", {},
      h("div", { class: "where" }, withLink ? h("a", { href: `#/campo/${o.id}` }, where) : where, ` · ${o.categoria}`),
      h("h4", {}, o.titulo),
      h("div", { class: "crit" }, `Critério: ${o.criterio}`),
      h("ul", {}, o.motivos.map((m) => h("li", {}, m)))));
}

pageRender.outliers = async (view, _p, ov) => {
  const [list, fig] = await Promise.all([cached("/api/outliers"), cached("/api/figures")]);
  const rmse = ov.key.mlp_rmse;
  $("#rule-res", view).textContent = fmt(1.5 * rmse, 1);
  const flagged = new Set(list.filter((o) => o.categoria === "Erro do modelo").map((o) => `${o.Data}|${o.Ponto}`));
  const pred = fig.images.filter((r) => r.res != null);
  registerChart("o-res", () => strip($("#o-res", view), {
    yLabel: "Erro da rede (SPAD)", zeroLine: true, height: 300,
    groups: [...DATES, TEST].map((d) => ({ name: d === TEST ? "01/06 (teste)" : shortDate(d), points: pred.filter((r) => r.Data === d).map((r) => ({
      y: r.res, color: flagged.has(`${r.Data}|${r.Ponto}`) ? css("--ink") : css("--neutral"), hollow: d === TEST && !flagged.has(`${r.Data}|${r.Ponto}`),
      label: `${shortDate(r.Data)} · ${plotLabel(r.id)}`, id: r.id, mark: flagged.has(`${r.Data}|${r.Ponto}`) ? (d === TEST ? `t${r.Ponto}` : `P${r.Ponto}`) : null })) })),
    legendItems: [{ name: "outlier do modelo", color: css("--ink") }, { name: "demais fotos", color: css("--neutral") }, { name: "01/06 (teste)", color: css("--neutral"), kind: "ring" }],
    onClick: (p) => goPlot(p.id),
  }));
  const cats = [...new Set(list.map((o) => o.categoria))];
  const box = $("#o-list", view);
  options($("#o-filter", view), [["all", `todos (${list.length})`], ...cats.map((c) => [c, `${c.toLowerCase()} (${list.filter((o) => o.categoria === c).length})`])], (sel) => {
    box.replaceChildren(...cats.filter((c) => sel === "all" || c === sel).flatMap((c) => {
      const items = list.filter((o) => o.categoria === c);
      return [h("div", { class: "cat-head" }, h("h3", {}, c), h("span", {}, `${items.length} caso${items.length > 1 ? "s" : ""}`)), ...items.map((o) => caseItem(o))];
    }));
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// Imagens
// ─────────────────────────────────────────────────────────────────────────────
pageRender.imagens = async (view) => {
  const photos = await cached("/api/photos");
  const plotPhotos = photos.filter((p) => !p.extra);
  const label = (p) => p.extra ? `${shortDate(p.Data)} · ${p.rel_path.split("/").pop()}` : `${shortDate(p.Data)} · ${plotLabel(p.id)}${p.Tratamento ? ` (${p.Tratamento})` : ""}`;
  const optionList = (sel, initial) => {
    sel.replaceChildren(...[...DATES, TEST].map((d) => h("optgroup", { label: shortDate(d) },
      plotPhotos.filter((p) => p.Data === d).map((p) => h("option", { value: p.rel_path }, label(p))))));
    if (initial && plotPhotos.some((p) => p.rel_path === initial)) sel.value = initial;
  };
  const byRel = Object.fromEntries(photos.map((p) => [p.rel_path, p]));

  // etapas
  const stepsSel = $("#steps-photo", view);
  optionList(stepsSel, "21-05/P5.jpg");
  const drawSteps = () => {
    const rel = stepsSel.value, p = byRel[rel];
    $("#steps-row", view).replaceChildren(...["crop", "pre", "mask", "blocks"].map((st) => h("figure", {},
      h("img", { src: stepUrl(rel, st, 480), alt: STEP_INFO[st][0], loading: "lazy", style: "cursor:zoom-in", onclick: () => openPhoto(rel, label(p)) }),
      h("figcaption", {}, h("b", {}, STEP_INFO[st][0]), STEP_INFO[st][1]))));
  };
  stepsSel.addEventListener("change", drawSteps);
  drawSteps();

  // comparação
  const a = $("#cmp-a", view), b = $("#cmp-b", view), st = $("#cmp-step", view);
  optionList(a, "26-05/P4.JPG");
  optionList(b, "26-05/P3.JPG");
  const drawCmp = () => {
    const pa = byRel[a.value], pb = byRel[b.value];
    const side = (p) => h("figure", { class: "shot" }, h("button", { type: "button", class: "frame", style: "aspect-ratio:auto", onclick: () => openPhoto(p.rel_path, label(p)) },
      h("img", { src: stepUrl(p.rel_path, st.value, 900), alt: label(p), style: "height:auto" })),
      h("figcaption", {}, h("div", { class: "d" }, label(p)), h("dl", {},
        h("dt", {}, "Falker"), h("dd", {}, fmt(p.obs, 1)), h("dt", {}, "Rede neural"), h("dd", {}, fmt(p.Predicao_MLP, 1)),
        h("dt", {}, "Cobertura"), h("dd", {}, p.Canopy_Cover != null ? `${fmt(p.Canopy_Cover, 0)}%` : "–"))));
    $("#compare", view).replaceChildren(side(pa), side(pb));
    const diff = pa.obs != null && pb.obs != null ? pb.obs - pa.obs : null;
    $("#compare-text", view).textContent = diff == null ? "" :
      `A foto da direita tem ${fmt(Math.abs(diff), 1)} SPAD ${diff > 0 ? "a mais" : "a menos"} que a da esquerda pelo Falker; a rede neural prevê uma diferença de ${fmt(Math.abs((pb.Predicao_MLP ?? 0) - (pa.Predicao_MLP ?? 0)), 1)}. ` +
      "Compare a quantidade de solo exposto e o tom de verde, que é o que a rede usa.";
  };
  [a, b, st].forEach((el) => el.addEventListener("change", drawCmp));
  drawCmp();

  // galeria
  const g = $("#gallery", view);
  let date = "all", trat = "all";
  const drawGallery = () => {
    const list = photos.filter((p) => (date === "all" || p.Data === date) && (trat === "all" || p.Tratamento === trat || (trat === "outras" && !p.Tratamento)));
    g.replaceChildren(...list.map((p) => shot(p.rel_path, label(p), p.extra ? [["Hora", p.photo_time || "–"]] : [
      ["Falker", fmt(p.obs, 1)], ["Rede neural", fmt(p.Predicao_MLP, 1)], ["Cobertura", p.Canopy_Cover != null ? `${fmt(p.Canopy_Cover, 0)}%` : "–"],
    ], { width: 420 })));
    if (!list.length) g.replaceChildren(h("p", { class: "muted" }, "Nenhuma foto com esse filtro."));
  };
  options($("#g-date", view), [["all", "todas as datas"], ...DATES.map((d) => [d, shortDate(d)]), [TEST, "01/06"]], (v) => { date = v; drawGallery(); });
  options($("#g-trat", view), [["all", "todos os tratamentos"], ...TRAT.map((t) => [t, `${t} · ${DOSE_OF[t]} kg`]), ["outras", "sem tratamento"]], (v) => { trat = v; drawGallery(); });

  // mapas
  const maps = await cached("/api/maps");
  $("#maps", view).replaceChildren(...maps.filter((m) => m.image).map((m) => h("figure", {},
    h("img", { src: m.image, alt: `Mapa de clorofila ${m.Data} ponto ${m.Ponto}`, loading: "lazy" }),
    h("figcaption", {}, `${m.Data.replace("-", "/")} · ${m.Data === "01-06" ? "ponto de teste" : "parcela"} ${m.Ponto} — Falker ${fmt(m.Clorofila_Observada, 1)}, rede ${fmt(m.Clorofila_Predita, 1)}, média do mapa ${fmt(m.SPAD_Medio_Mapa, 1)} SPAD.`))));
};

// ─────────────────────────────────────────────────────────────────────────────
// Dados
// ─────────────────────────────────────────────────────────────────────────────
pageRender.dados = async (view) => {
  const meta = await api("/api/tables");
  const sel = $("#db-table", view), runSel = $("#db-run", view);
  const groups = {};
  for (const m of meta) (groups[m.group] ??= []).push(m);
  sel.replaceChildren(...Object.entries(groups).map(([g, items]) => h("optgroup", { label: g }, items.map((m) => h("option", { value: m.name }, `${m.name} (${fmt(m.rows)})`)))));
  if (meta.some((m) => m.name === "falker_readings")) sel.value = "falker_readings";
  let current;
  const open = () => {
    const m = meta.find((x) => x.name === sel.value);
    $("#db-desc", view).textContent = [m.group, m.description, m.script ? `gerada por ${m.script}` : null, m.n_runs > 1 ? `${m.n_runs} execuções guardadas` : null].filter(Boolean).join(" · ");
    runSel.dataset.table = "";
    current = serverTable($("#db-view", view), m.name, {
      limit: 20,
      onLoad: (d) => {
        if (runSel.dataset.table === m.name) return;
        runSel.dataset.table = m.name;
        runSel.replaceChildren(...[["latest", "mais recente"], ...d.runs.map((r) => [String(r), `execução ${r}`]), ["all", "todas"]].map(([v, l]) => h("option", { value: v }, l)));
        runSel.disabled = d.runs.length === 0;
      },
    });
  };
  sel.addEventListener("change", open);
  runSel.addEventListener("change", () => current?.setRun(runSel.value));
  open();

  renderDownloads(view);

  const reports = await api("/api/reports");
  $("#reports", view).replaceChildren(...reports.map((r) => {
    const det = h("details", { class: "report" }, h("summary", {}, r.name, h("span", { class: "muted small" }, `  · execução ${r.run_id}`)));
    det.addEventListener("toggle", async () => {
      if (!det.open || det.dataset.loaded) return;
      det.dataset.loaded = "1";
      const rep = await api(`/api/report?${qs({ name: r.name })}`);
      det.append(r.format === "markdown" ? renderMarkdown(rep.content) : h("pre", { class: "md" }, rep.content));
    });
    return det;
  }));
  const runs = await api("/api/runs");
  $("#runs", view).replaceChildren(localTable(["run_id", "script", "status", "started_at", "finished_at"], runs,
    { labels: { run_id: "execução", script: "script", status: "status", started_at: "início", finished_at: "fim" } }));
};

// ─────────────────────────────────────────────────────────────────────────────
// Coletas (análise dentro de cada data)
// ─────────────────────────────────────────────────────────────────────────────
pageRender.coletas = async (view) => {
  const pd = await cached("/api/per_date");
  const byDate = Object.fromEntries(pd.summary.map((r) => [r.Data, r]));
  const dates = pd.summary.map((r) => r.Data);
  const loo = (r) => (r.Confiavel ? r.Melhor_R2_LOO : null);

  $("#pd-summary", view).replaceChildren(localTable(
    ["Data", "n_fotos", "n_leituras", "SPAD_media", "SPAD_faixa", "Melhor_variavel", "Melhor_r", "Melhor_R2_LOO", "Significativos", "Modelo_r_na_data", "Modelo_vies"],
    pd.summary.map((r) => ({ ...r, Data: shortDate(r.Data), SPAD_faixa: `${fmt(r.SPAD_min, 1)}–${fmt(r.SPAD_max, 1)}`,
      Melhor_R2_LOO: loo(r), Significativos: `${r.Variaveis_q05} de ${r.Variaveis_testadas}` })),
    { formats: { SPAD_media: 1, Melhor_r: 2, Melhor_R2_LOO: 2, Modelo_r_na_data: 2, Modelo_vies: "+" },
      labels: { n_fotos: "fotos", n_leituras: "leituras", SPAD_media: "SPAD médio", SPAD_faixa: "faixa", Melhor_variavel: "melhor índice",
        Melhor_r: "r", Melhor_R2_LOO: "R² validação", Significativos: "índices com q < 0,05", Modelo_r_na_data: "r do modelo", Modelo_vies: "viés do modelo" } }));

  const drawDate = (d) => {
    const s = byDate[d], reliable = !!s.Confiavel;
    const top = pd.top.filter((r) => r.Data === d);
    const points = pd.points.filter((r) => r.Data === d);
    const anova = pd.anova.filter((r) => r.Data === d);

    // texto
    const txt = $("#pd-text", view);
    if (!reliable) {
      txt.replaceChildren(h("strong", {}, `${shortDate(d)}: só ${s.n_fotos} pontos. `),
        "Qualquer reta passa perto de 3 pontos, então as correlações abaixo (algumas perto de 1) não significam nada. Os pontos de teste ficam registrados, mas não entram na conclusão.");
    } else {
      txt.replaceChildren(
        `Em ${shortDate(d)}, ${s.n_fotos} parcelas tinham foto e leitura Falker (SPAD de ${fmt(s.SPAD_min, 1)} a ${fmt(s.SPAD_max, 1)}). `,
        h("strong", {}, `${s.Variaveis_q05} dos ${s.Variaveis_testadas} índices`), " têm correlação significativa com o SPAD mesmo depois da correção para múltiplos testes. ",
        "O melhor, ", h("code", {}, s.Melhor_variavel), `, tem r = ${fmt(s.Melhor_r, 2)} e `,
        h("strong", {}, `R² de validação ${fmt(s.Melhor_R2_LOO, 2)}`), ` (erro médio de ${fmt(s.Melhor_MAE_LOO, 1)} SPAD ao prever uma parcela pelas outras). `,
        s.rb_r != null ? `A razão r/b (mediana), usada em todo o trabalho, tem r = ${fmt(s.rb_r, 2)} nesta data. ` : "",
        s.Modelo_r_na_data != null ? `O modelo definitivo, que não viu esta data no treino, ordena as parcelas com r = ${fmt(s.Modelo_r_na_data, 2)}, mas com viés de ${fmt(s.Modelo_vies, "+")} SPAD: dentro da data a relação é boa; o que muda de uma data para outra é o nível.` : "");
    }

    // ranking
    const metric = reliable ? "R2_LOO" : "R2";
    $("#pd-rank-title", view).textContent = reliable ? `Os 15 índices que melhor acompanham o SPAD em ${shortDate(d)}` : `${shortDate(d)}: correlações com 3 pontos (sem valor inferencial)`;
    registerChart("pd-rank", () => hbar($("#pd-rank", view), {
      xLabel: reliable ? "R² de validação dentro da data" : "R² (3 pontos)", valueFmt: (v) => fmt(v, 2),
      items: top.map((r, i) => ({ label: r.Variavel, value: r[metric], highlight: i === 0, color: i === 0 ? css("--series-1") : css("--neutral"),
        note: `r = ${fmt(r.r, 2)} · q = ${fmt(r.q, "p")}` })),
    }), { columns: ["Rank", "Variavel", "r", "rho", "R2", "R2_LOO", "MAE_LOO", "p", "q"], rows: top,
      formats: { r: 3, rho: 3, R2: 3, R2_LOO: 3, MAE_LOO: 2, p: "p", q: "p" }, labels: { rho: "Spearman", R2_LOO: "R² validação", MAE_LOO: "erro validação" } });

    // dispersão
    const vars = [...new Set(points.map((p) => p.Variavel))];
    const sel = $("#pd-var", view);
    sel.replaceChildren(...vars.map((v) => h("option", { value: v }, v)));
    const drawScatter = () => {
      const v = sel.value, pts = points.filter((p) => p.Variavel === v), info = top.find((r) => r.Variavel === v);
      registerChart("pd-scatter", () => scatter($("#pd-scatter", view), {
        points: pts.map((p) => ({ x: p.Valor, y: p.SPAD, color: p.Dose != null ? doseColor(p.Dose) : css("--neutral"), hollow: p.Dose == null,
          label: `${d === TEST ? "ponto de teste" : "parcela"} ${p.Ponto}${p.Tratamento && d !== TEST ? ` (${p.Tratamento})` : ""}`,
          mark: d === TEST ? `t${p.Ponto}` : `P${p.Ponto}`, id: d === TEST ? `teste-${p.Ponto}` : String(p.Ponto),
          tip: p.Previsto_LOO != null ? [{ value: fmt(p.Previsto_LOO, 1), label: "previsto pelas outras parcelas" }] : [] })),
        xLabel: v, yLabel: "SPAD medido", fit: true, height: 330, onClick: (p) => goPlot(p.id),
        legendItems: d === TEST ? [{ name: "ponto de teste", color: css("--neutral"), kind: "ring" }] :
          [0, 50, 75, 100].map((x) => ({ name: `${x} kg N/ha`, color: doseColor(x) })).concat([{ name: "reta ajustada", color: css("--ink-2"), kind: "line" }]),
      }));
      $("#pd-scatter-text", view).textContent = info
        ? `r = ${fmt(info.r, 2)} · R² = ${fmt(info.R2, 2)}${reliable ? ` · R² de validação = ${fmt(info.R2_LOO, 2)} · erro médio ao prever cada parcela pelas outras = ${fmt(info.MAE_LOO, 1)} SPAD` : ""}. Cor mais escura = dose maior. Clique num ponto para abrir a parcela.`
        : "";
    };
    sel.onchange = drawScatter;
    drawScatter();

    // ANOVA
    const at = $("#pd-anova", view), atext = $("#pd-anova-text", view);
    if (!anova.length) { at.replaceChildren(h("p", { class: "muted" }, "Sem ANOVA nesta data (pontos fora do delineamento).")); atext.textContent = ""; }
    else {
      at.replaceChildren(localTable(["Variavel", "F_tratamento", "p_tratamento", "F_bloco", "p_bloco", "Parcelas", "Tratamentos_com_foto"], anova,
        { formats: { F_tratamento: 1, p_tratamento: "p", F_bloco: 1, p_bloco: "p" },
          labels: { Variavel: "variável", F_tratamento: "F (dose)", p_tratamento: "p (dose)", F_bloco: "F (bloco)", p_bloco: "p (bloco)", Parcelas: "parcelas", Tratamentos_com_foto: "doses" } }));
      const spad = anova.find((r) => r.Variavel.startsWith("SPAD")), idx = anova.filter((r) => !r.Variavel.startsWith("SPAD"));
      const best = idx.reduce((a, b) => (b.F_tratamento > a.F_tratamento ? b : a), idx[0]);
      atext.replaceChildren("Mesmo modelo em blocos usado para o SPAD, aplicado aos índices da foto (média por parcela). ",
        spad && best ? (best.F_tratamento > spad.F_tratamento
          ? h("strong", {}, `Nesta data, o índice ${best.Variavel} separa as doses com F = ${fmt(best.F_tratamento, 1)}, mais que o próprio SPAD (F = ${fmt(spad.F_tratamento, 1)}). `)
          : `O SPAD separa as doses com F = ${fmt(spad.F_tratamento, 1)}; o melhor índice, com F = ${fmt(best.F_tratamento, 1)}. `) : "",
        spad && best && best.F_tratamento > spad.F_tratamento ? "Uma explicação plausível: a foto resume o dossel inteiro, enquanto o Falker amostra cerca de 7 folhas por parcela. " : "",
        anova[0].Parcelas < 12 ? `Com ${anova[0].Parcelas} parcelas, sobram poucos graus de liberdade, e os p-valores ficam altos mesmo com F grande.` : "");
    }

    // tabela completa
    serverTable($("#pd-table", view), "per_date_correlations", {
      limit: 12, filters: { Data: d }, sort: "Rank", dir: "asc", hide: ["Data", "Confiavel", "n", "Inclinacao", "Intercepto"],
      formats: { r: 3, rho: 3, R2: 3, R2_LOO: 3, MAE_LOO: 2, p: "p", q: "p" },
      labels: { Variavel: "variável", Indice: "índice", Estatistica: "estatística", rho: "Spearman", R2_LOO: "R² validação", MAE_LOO: "erro validação" },
    });
  };
  options($("#pd-date", view), dates.map((d) => [d, d === TEST ? "01/06 (teste)" : shortDate(d)]), drawDate, dates[0]);

  // consistência
  const cons = pd.consistency;
  const cols = ["r_18-05", "r_21-05", "r_26-05"];
  registerChart("pd-cons", () => heatmap($("#pd-cons", view), {
    rows: cons.map((r) => r.Variavel), cols, colLabel: (c) => c.slice(2).replace("-", "/"),
    value: (row, c) => { const r = cons.find((x) => x.Variavel === row)?.[c]; return r == null ? null : Math.abs(r); },
    label: (v, row, c) => fmt(cons.find((x) => x.Variavel === row)[c], 2), title: "r", legendLabel: "|r|: ",
  }), { columns: ["Variavel", ...cols, "R2_LOO_min", "R2_LOO_medio", "Mesmo_sinal"], rows: cons,
    formats: Object.fromEntries([...cols, "R2_LOO_min", "R2_LOO_medio"].map((c) => [c, 3])), labels: { R2_LOO_min: "R² validação (pior data)", R2_LOO_medio: "R² validação (média)" } });
  if (cons.length) {
    const c0 = cons[0], allSame = cons.every((r) => r.Mesmo_sinal);
    $("#pd-cons-text", view).replaceChildren(
      "O mais consistente é ", h("code", {}, c0.Variavel), `, com R² de validação de pelo menos ${fmt(c0.R2_LOO_min, 2)} em qualquer das três datas (média ${fmt(c0.R2_LOO_medio, 2)}). `,
      allSame ? "Todos os 20 mantêm o mesmo sinal nas três datas: a direção da relação é estável; o que varia entre datas é a força e o nível. " : "",
      (() => {
        const weakest = cons.map((r) => cols.reduce((a, c) => (Math.abs(r[c]) < Math.abs(r[a]) ? c : a), cols[0]));
        const n21 = weakest.filter((c) => c === "r_21-05").length;
        return n21 ? `21/05 é a data mais fraca para ${n21} dos ${cons.length}; é também a data em que o modelo, treinado nas outras, subestimou as parcelas.` : "";
      })());
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Altura e massa (índices × variáveis de campo, busca de MLP)
// ─────────────────────────────────────────────────────────────────────────────
pageRender.vegetativo = async (view) => {
  const vg = await cached("/api/vegetativo");
  if (!vg.summary.length) { view.replaceChildren(h("p", { class: "muted" }, "Rode python run.py vegetative_indices para gerar esta análise.")); return; }
  const S = Object.fromEntries(vg.summary.map((r) => [r.Variavel_campo, r]));
  const unit = (v) => (v === "Altura_Media" ? "cm" : v.startsWith("Rel_") ? "" : "t/ha");
  const short = (v) => S[v].Rotulo_campo.replace(/ \(.*\)$/, "");
  const REF_LABEL = { "Clorofila Total": "SPAD medido (Falker)", SPAD_previsto: "SPAD previsto pela foto", Dose: "Dose de N", Canopy_Cover: "Cobertura vegetal" };
  const model = (v, name) => vg.models.find((m) => m.Variavel_campo === v && m.Abordagem.startsWith(name));

  // números de destaque
  const fact = (n, l) => h("div", {}, h("div", { class: "n" }, n), h("div", { class: "l" }, l));
  // só números validados sem olhar a parcela testada (nada escolhido com as 12 parcelas)
  const honestPhoto = (v) => vg.models.filter((m) => m.Variavel_campo === v && m.Honesto && !m.Abordagem.includes("SPAD medido"))
    .reduce((a, b) => (b.R2 > a.R2 ? b : a));
  const hp = Object.fromEntries([...new Set(vg.models.map((m) => m.Variavel_campo))].map((v) => [v, honestPhoto(v)]));
  const lower = (s) => (/^(Ridge|MLP)/.test(s) ? s : s.charAt(0).toLowerCase() + s.slice(1)).replace(/ \(.*\)$/, "");
  $("#vg-facts", view).replaceChildren(
    fact(fmt(hp.Altura_Media.R2, 2), `R² da altura só com a foto (${lower(hp.Altura_Media.Abordagem)})`),
    fact(fmt(hp.Massa_Verde_Folha_ha.R2, 2), `R² da massa verde de folha (${lower(hp.Massa_Verde_Folha_ha.Abordagem)})`),
    fact(fmt(S.Massa_Verde_Folha_ha.SPAD_medido_R2_LOO, 2), "R² da mesma massa pelo SPAD do Falker"),
    fact(fmt(hp.Massa_Verde_Total_ha.R2, 2), `R² da massa verde total (${lower(hp.Massa_Verde_Total_ha.Abordagem)})`));

  $("#vg-summary", view).replaceChildren(localTable(
    ["Rotulo_campo", "Melhor_indice", "Melhor_r", "Melhor_R2_LOO", "Significativos", "Cobertura_R2_LOO", "SPAD_medido_R2_LOO", "SPAD_previsto_R2_LOO", "Dose_R2_LOO"],
    vg.summary.map((r) => ({ ...r, Significativos: `${r.Indices_q05} de ${r.Indices_testados}` })),
    { formats: { Melhor_r: 2, Melhor_R2_LOO: 2, Cobertura_R2_LOO: 2, SPAD_medido_R2_LOO: 2, SPAD_previsto_R2_LOO: 2, Dose_R2_LOO: 2 },
      labels: { Rotulo_campo: "variável", Melhor_indice: "melhor índice", Melhor_r: "r", Melhor_R2_LOO: "R² validação",
        Significativos: "índices com q < 0,05", Cobertura_R2_LOO: "R² cobertura", SPAD_medido_R2_LOO: "R² SPAD medido",
        SPAD_previsto_R2_LOO: "R² SPAD previsto", Dose_R2_LOO: "R² dose" } }));

  const targets = vg.summary.map((r) => r.Variavel_campo);
  const drawTarget = (v) => {
    const s = S[v], u = unit(v);
    const top = vg.top.filter((r) => r.Variavel_campo === v);
    const points = vg.points.filter((r) => r.Variavel_campo === v);
    const beats = s.Melhor_R2_LOO > s.SPAD_medido_R2_LOO;
    $("#vg-text", view).replaceChildren(
      `${s.Rotulo_campo}: de ${fmt(s.Min, 1)} a ${fmt(s.Max, 1)} ${u} entre as parcelas. `,
      s.Indices_q05 ? h("strong", {}, `${s.Indices_q05} dos ${s.Indices_testados} índices`) : "Nenhum índice",
      s.Indices_q05 ? " têm correlação significativa depois da correção para múltiplos testes. " : " passa na correção para múltiplos testes. ",
      "O melhor, ", h("code", {}, s.Melhor_indice), `, tem r = ${fmt(s.Melhor_r, 2)} e R² de validação ${fmt(s.Melhor_R2_LOO, 2)} `,
      `(erro médio de ${fmt(s.Melhor_MAE_LOO, 2)} ${u}). `,
      s.Melhor_R2_LOO > 0.3 ? (beats
        ? h("strong", {}, `A foto explica esta variável melhor do que o próprio Falker (R² ${fmt(s.SPAD_medido_R2_LOO, 2)} com o SPAD medido).`)
        : `O SPAD medido pelo Falker chega a R² ${fmt(s.SPAD_medido_R2_LOO, 2)}.`) : "Com esse R², a foto não serve para estimar esta variável.",
      (() => {
        const lin = model(v, "Reta no melhor");
        if (!lin) return "";
        return ` Atenção: esse índice foi escolhido entre ${s.Indices_testados} olhando as 12 parcelas. Escolhendo o índice só com as parcelas de treino, a reta fica em R² ${fmt(lin.R2, 2)}`
          + (hp[v] ? `; a melhor abordagem validada sem essa escolha é ${lower(hp[v].Abordagem)}, com R² ${fmt(hp[v].R2, 2)} (seção Rede neural).` : ".");
      })());

    $("#vg-rank-title", view).textContent = `Os 15 índices que melhor acompanham: ${short(v).toLowerCase()}`;
    const refs = [["Clorofila Total", s.SPAD_medido_R2_LOO], ["SPAD_previsto", s.SPAD_previsto_R2_LOO], ["Dose", s.Dose_R2_LOO], ["Canopy_Cover", s.Cobertura_R2_LOO]];
    registerChart("vg-rank", () => hbar($("#vg-rank", view), {
      xLabel: "R² de validação (deixando uma parcela de fora)", valueFmt: (x) => fmt(x, 2),
      items: [
        ...top.map((r, i) => ({ label: r.Variavel, value: r.R2_LOO, highlight: i === 0, color: i === 0 ? css("--series-1") : css("--neutral"), note: `r = ${fmt(r.r, 2)} · q = ${fmt(r.q, "p")}` })),
        ...refs.map(([k, val]) => ({ label: REF_LABEL[k], value: val, color: css("--series-2"), note: "referência" })),
      ],
    }), { columns: ["Rank", "Variavel", "r", "rho", "R2", "R2_LOO", "MAE_LOO", "q"], rows: top,
      formats: { r: 3, rho: 3, R2: 3, R2_LOO: 3, MAE_LOO: 2, q: "p" }, labels: { rho: "Spearman", R2_LOO: "R² validação", MAE_LOO: "erro validação" } });

    const vars = [...new Set(points.map((p) => p.Variavel))];
    const sel = $("#vg-var", view);
    sel.replaceChildren(...vars.map((x) => h("option", { value: x }, REF_LABEL[x] || x)));
    const drawScatter = () => {
      const x = sel.value, pts = points.filter((p) => p.Variavel === x);
      const info = top.find((r) => r.Variavel === x);
      registerChart("vg-scatter", () => scatter($("#vg-scatter", view), {
        points: pts.map((p) => ({ x: p.Valor, y: p.Y, color: doseColor(p.Dose), label: `parcela ${p.Ponto} (${p.Tratamento})`, mark: `P${p.Ponto}`, id: String(p.Ponto) })),
        xLabel: REF_LABEL[x] || x, yLabel: s.Rotulo_campo, fit: true, height: 330, onClick: (p) => goPlot(p.id),
        legendItems: [0, 50, 75, 100].map((d) => ({ name: `${d} kg N/ha`, color: doseColor(d) })).concat([{ name: "reta ajustada", color: css("--ink-2"), kind: "line" }]),
      }));
      $("#vg-scatter-text", view).textContent = (info ? `r = ${fmt(info.r, 2)} · R² de validação = ${fmt(info.R2_LOO, 2)} · erro médio = ${fmt(info.MAE_LOO, 2)} ${u}. ` : "")
        + "Cor mais escura = dose maior. Se as cores formam faixas separadas, parte da relação pode ser efeito da dose.";
    };
    sel.onchange = drawScatter;
    drawScatter();
  };
  options($("#vg-target", view), targets.map((v) => [v, short(v)]), drawTarget, targets[0]);

  // rede neural
  $("#vg-n-configs", view).textContent = fmt(vg.search_n);
  const mTargets = [...new Set(vg.models.map((m) => m.Variavel_campo))];
  const predSel = $("#vg-pred-kind", view);
  let curM = mTargets[0];
  const drawPred = () => {
    const v = curM, k = predSel.value, rows = vg.predictions.filter((r) => r.Variavel_campo === v);
    registerChart("vg-pred", () => scatter($("#vg-pred", view), {
      points: rows.map((r) => ({ x: r.Observado, y: r[k], color: doseColor(r.Dose), label: `parcela ${r.Ponto} (${r.Tratamento})`, id: String(r.Ponto) })),
      xLabel: `${short(v)} medido`, yLabel: "previsto", identity: true, square: true, height: 340, onClick: (p) => goPlot(p.id),
      legendItems: [0, 50, 75, 100].map((d) => ({ name: `${d} kg N/ha`, color: doseColor(d) })),
    }));
  };
  predSel.onchange = drawPred;
  const drawModels = (v) => {
    curM = v;
    const ms = vg.models.filter((m) => m.Variavel_campo === v), u = unit(v);
    $("#vg-models-title", view).textContent = `R² de validação por abordagem: ${short(v).toLowerCase()}`;
    registerChart("vg-models", () => hbar($("#vg-models", view), {
      xLabel: "R² fora da amostra", valueFmt: (x) => fmt(x, 2),
      items: ms.map((m) => ({ label: m.Abordagem, value: m.R2, color: m.Abordagem.includes("otimista") ? css("--neutral") : m.Abordagem.startsWith("MLP") ? css("--series-1") : m.Abordagem.includes("SPAD") ? css("--series-2") : css("--series-3"),
        note: `erro médio ${fmt(m.MAE, 2)} ${u}${m.Detalhe ? ` · ${m.Detalhe}` : ""}` })),
    }), { columns: ["Abordagem", "R2", "MAE", "RMSE", "Detalhe"], rows: ms, formats: { R2: 3, MAE: 2, RMSE: 2 } });
    const flat = model(v, "MLP: melhor"), nest = model(v, "MLP: busca aninhada"), lin = model(v, "Reta no melhor"), spad = model(v, "Reta no SPAD medido");
    $("#vg-models-text", view).replaceChildren(
      `A melhor das redes chega a R² ${fmt(flat.R2, 2)} (${flat.Detalhe}). Refazendo a busca sem a parcela testada, o R² cai para `,
      h("strong", {}, fmt(nest.R2, 2)), `: a diferença é o quanto a busca "decorou" estas 12 parcelas. `,
      nest.R2 > lin.R2 + 0.03 ? `A rede supera a reta no melhor índice (${fmt(lin.R2, 2)}), também escolhido só com o treino. `
        : lin.R2 > nest.R2 + 0.03 ? `Uma reta num único índice, escolhido só com o treino, faz melhor (${fmt(lin.R2, 2)}): com 12 parcelas, a rede não compensa. `
          : `A reta num único índice, escolhido só com o treino, empata (${fmt(lin.R2, 2)}). `,
      `Para comparação, o SPAD medido pelo Falker dá R² ${fmt(spad.R2, 2)}. `,
      "Nota: o R² da melhor rede na tabela abaixo usa uma semente; na figura, a mesma rede é a média de 5 sementes, e a diferença mostra o quanto o resultado depende da inicialização.");
    const search = vg.search_top.filter((r) => r.Variavel_campo === v);
    $("#vg-search", view).replaceChildren(localTable(["Rank", "Entradas", "Neuronios", "Ativacao", "Alpha", "R2_LOO", "MAE_LOO"], search,
      { formats: { R2_LOO: 3, MAE_LOO: 2 }, labels: { Neuronios: "neurônios", Ativacao: "ativação", Alpha: "α", R2_LOO: "R² LOO", MAE_LOO: "erro médio" } }));
    drawPred();
  };
  options($("#vg-mtarget", view), mTargets.map((v) => [v, short(v)]), drawModels, mTargets[0]);
};

// aba "Baixar dados": escolher tabelas e baixar CSV (uma) ou ZIP (várias)
async function renderDownloads(view) {
  const cats = await api("/api/downloads");
  const chosen = new Set();
  const boxes = new Map();
  const count = $("#dl-count", view), go = $("#dl-go", view);
  const sync = () => {
    for (const [name, b] of boxes) b.checked = chosen.has(name);
    for (const g of $$(".dl-group", view)) {
      const names = JSON.parse(g.dataset.names), n = names.filter((x) => chosen.has(x)).length;
      const all = g.querySelector(".dl-group-all");
      all.checked = n === names.length; all.indeterminate = n > 0 && n < names.length;
      g.querySelector(".dl-n").textContent = n ? `${n} de ${names.length} marcadas` : `${names.length} tabelas`;
    }
    count.textContent = chosen.size ? `${chosen.size} selecionada${chosen.size > 1 ? "s" : ""}` : "nenhuma selecionada";
    go.disabled = !chosen.size;
    go.textContent = chosen.size > 1 ? `Baixar ${chosen.size} tabelas (.zip)` : "Baixar (.csv)";
  };
  const toggle = (name, on) => { on ? chosen.add(name) : chosen.delete(name); sync(); };
  $("#dl-groups", view).replaceChildren(...cats.map((c) => {
    const names = c.tables.map((t) => t.name);
    const all = h("input", { type: "checkbox", class: "dl-group-all", "aria-label": `Marcar ${c.category}` });
    all.addEventListener("change", () => { names.forEach((n) => (all.checked ? chosen.add(n) : chosen.delete(n))); sync(); });
    const rows = c.tables.map((t) => {
      const b = h("input", { type: "checkbox" });
      boxes.set(t.name, b);
      b.addEventListener("change", () => toggle(t.name, b.checked));
      const size = t.virtual ? "montada na hora" : `${fmt(t.rows)} linhas × ${fmt(t.cols)} colunas${t.n_runs > 1 ? ` · ${t.n_runs} execuções` : ""}`;
      return h("label", { class: "dl-row" }, b,
        h("span", { class: "dl-name" }, h("code", {}, t.name)),
        h("span", { class: "dl-desc small muted" }, [t.description, size].filter(Boolean).join(" · ")));
    });
    const legacy = c.category.includes("legado");
    return h("details", { class: "dl-group", open: legacy ? null : "", "data-names": JSON.stringify(names) },
      h("summary", {}, all, h("strong", {}, ` ${c.category}`), h("span", { class: "dl-n small muted" }), h("div", { class: "small muted" }, c.note)),
      h("div", { class: "dl-list" }, rows));
  }));
  // o clique no checkbox do grupo não deve abrir/fechar o <details>
  $$(".dl-group-all", view).forEach((b) => b.addEventListener("click", (e) => e.stopPropagation()));
  $("#dl-all", view).onclick = () => { cats.filter((c) => !c.category.includes("legado")).forEach((c) => c.tables.forEach((t) => chosen.add(t.name))); sync(); };
  $("#dl-none", view).onclick = () => { chosen.clear(); sync(); };
  go.onclick = async () => {
    const fmtv = $("#dl-fmt", view).value, run = $("#dl-run", view).value;
    const names = [...chosen];
    if (STATIC) {
      go.disabled = true; go.textContent = "Preparando…";
      try {
        if (names.length === 1) {
          saveBlob(new Blob([await staticCsv(names[0], { run }, fmtv)], { type: "text/csv" }), `${names[0]}.csv`);
        } else {
          const catOf = Object.fromEntries(cats.flatMap((c) => c.tables.map((tb) => [tb.name, [c.category, tb.description]])));
          const files = [];
          for (const n of names) files.push({ name: `${catOf[n][0]}/${n}.csv`, data: await staticCsv(n, { run }, fmtv) });
          const readme = [`Exportado da versão estática do site (dados de database/tcc.sqlite).`,
            `Execução: ${run === "all" ? "todas (coluna run_id identifica cada uma)" : "a mais recente de cada tabela"}`,
            `Formato: ${fmtv === "br" ? "ponto e vírgula, vírgula decimal, UTF-8 com BOM (Excel em português)" : "vírgula, ponto decimal, UTF-8"}`, "",
            ...names.map((n) => `${catOf[n][0]}/${n}.csv: ${catOf[n][1] || ""}`)].join("\n") + "\n";
          files.push({ name: "LEIAME.txt", data: readme });
          const d = new Date(), stamp = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
          saveBlob(makeZip(files), `dados_tcc_${stamp}.zip`);
        }
      } finally { sync(); }
      return;
    }
    const url = names.length === 1
      ? `/api/table/${encodeURIComponent(names[0])}.csv?${qs({ fmt: fmtv, run })}`
      : `/api/export.zip?${names.map((n) => `t=${encodeURIComponent(n)}`).join("&")}&${qs({ fmt: fmtv, run })}`;
    const a = h("a", { href: url, download: "" });
    document.body.append(a); a.click(); a.remove();
  };
  // começa com as planilhas consolidadas marcadas
  cats.find((c) => c.category === "Planilhas consolidadas")?.tables.forEach((t) => chosen.add(t.name));
  sync();
}

// ─────────────────────────────────────────────────────────────────────────────
// Classes de clorofila (faixas de Beaufils sobre o SPAD medido)
// ─────────────────────────────────────────────────────────────────────────────
const CL_NAMES = ["Baixa", "Média", "Alta"];
const CL_STEP = [1, 3, 6];   // passos do ramp sequencial: mais escuro = mais clorofila
const clColor = (c) => SEQ()[CL_STEP[CL_NAMES.indexOf(c)]] ?? css("--neutral");
const clInk = (c) => (CL_NAMES.indexOf(c) >= 1 ? "#fff" : "#1b1b1a");

pageRender.classes = async (view) => {
  const cl = await cached("/api/classes");
  if (!cl.plots.length) { view.replaceChildren(h("p", { class: "muted" }, "Rode python run.py chl_classes para gerar esta análise.")); return; }
  const D = cl.distribution[0];
  const pct = (v) => `${fmt(v * 100, 0)}%`;
  const kappaWord = (k) => (k > 0.8 ? "quase perfeita" : k > 0.6 ? "substancial" : k > 0.4 ? "moderada" : k > 0.2 ? "razoável" : "fraca");
  const fact = (n, l) => h("div", {}, h("div", { class: "n" }, n), h("div", { class: "l" }, l));
  const A = (prefix) => cl.agreement.find((a) => a.Metodo.startsWith(prefix) && a.Escopo === "Todas as datas");
  const aBest = A("Melhor índice"), aNest = A("Índice escolhido"), aNet = A("Rede neural"), aMlp = A("MLP pequena") ?? aNest;
  const best = cl.index_ranking[0];
  const range = (r) => (r.SPAD_min == null ? `< ${fmt(r.SPAD_max, 1)}` : r.SPAD_max == null ? `> ${fmt(r.SPAD_min, 1)}` : `${fmt(r.SPAD_min, 1)} a ${fmt(r.SPAD_max, 1)}`);

  $("#cl-facts", view).replaceChildren(
    fact(`${fmt(D.Media, 1)} ± ${fmt(D.Desvio_padrao, 1)}`, "SPAD médio ± desvio das parcelas: base das classes"),
    fact(pct(D.Teto_exato), "o Falker concorda consigo mesmo ao medir a parcela de novo (teto)"),
    fact(pct(aMlp.Acerto_exato), `classe exata pela foto (rede pequena, escolhida sem ver a data testada; kappa ${fmt(aMlp.Kappa_ponderado, 2)})`),
    fact(pct(aMlp.Acerto_dentro_incerteza), "pela foto, dentro da incerteza do Falker"));

  // critério e distribuição
  $("#cl-n", view).textContent = fmt(D.n);
  $("#cl-normal", view).textContent = `O critério pressupõe distribuição aproximadamente normal. Os ${D.n} valores vão de ${fmt(D.Minimo, 1)} a ${fmt(D.Maximo, 1)}, `
    + `com média ${fmt(D.Media, 2)} e desvio padrão ${fmt(D.Desvio_padrao, 2)} (CV ${fmt(D.CV_pct, 1)}%). O teste de Shapiro-Wilk não rejeita a normalidade `
    + `(W = ${fmt(D.Shapiro_W, 3)}; p = ${fmt(D.Shapiro_p, 2)}), e a proporção de parcelas em cada classe fica perto da esperada para uma normal (tabela abaixo).`;
  $("#cl-scheme", view).replaceChildren(h("div", { class: "table-wrap" }, h("table", { class: "data" },
    h("thead", {}, h("tr", {}, ["classe", "faixa de Beaufils", "limites (z)", "SPAD", "parcelas", "observado", "esperado (normal)"].map((x, i) => h("th", { class: i >= 4 ? "num" : "" }, x)))),
    h("tbody", {}, cl.scheme.map((r) => h("tr", {},
      h("td", {}, h("span", { class: "cl-chip", style: `background:${clColor(r.Classe)};color:${clInk(r.Classe)}` }, r.Classe)),
      h("td", {}, r.Classe_Beaufils),
      h("td", { style: "white-space:nowrap" }, (() => { const z = (v) => `m ${v < 0 ? "−" : "+"} ${Math.abs(v) > 1 ? "4/3" : "2/3"} s`; return r.z_min == null ? `< ${z(r.z_max)}` : r.z_max == null ? `> ${z(r.z_min)}` : `${z(r.z_min)} a ${z(r.z_max)}`; })()),
      h("td", { class: "num", style: "white-space:nowrap" }, range(r)),
      h("td", { class: "num" }, fmt(r.n_parcelas)), h("td", { class: "num" }, pct(r.Proporcao_observada)), h("td", { class: "num" }, pct(r.Proporcao_esperada_normal))))))));

  registerChart("cl-dist", () => {
    const el = $("#cl-dist", view), height = 170, mg = { l: 16, r: 16, t: 26, b: 42 };
    const { svg, width } = frame(el, height);
    const lo = Math.floor(D.Minimo - 1), hi = Math.ceil(D.Maximo + 1);
    const x = scale(lo, hi, mg.l, width - mg.r);
    cl.scheme.forEach((r) => {
      const a = x(Math.max(lo, r.SPAD_min ?? lo)), b = x(Math.min(hi, r.SPAD_max ?? hi));
      svg.append(s("rect", { x: a, y: mg.t, width: b - a, height: height - mg.t - mg.b, fill: clColor(r.Classe), opacity: 0.16 }),
        s("line", { x1: a, x2: a, y1: mg.t, y2: height - mg.b, stroke: css("--ink-3"), "stroke-dasharray": "3 3", opacity: r.SPAD_min == null ? 0 : 0.7 }),
        s("text", { x: (a + b) / 2, y: mg.t - 8, "text-anchor": "middle", class: "tick-label" }, r.Classe));
    });
    for (const t of niceTicks(lo, hi, 8).filter((v) => v >= lo && v <= hi)) svg.append(s("text", { x: x(t), y: height - mg.b + 16, "text-anchor": "middle", class: "tick-label" }, fmt(t, 0)));
    svg.append(s("line", { x1: mg.l, x2: width - mg.r, y1: height - mg.b, y2: height - mg.b, class: "baseline" }),
      s("text", { x: (mg.l + width - mg.r) / 2, y: height - 6, "text-anchor": "middle", class: "axis-label" }, "SPAD médio da parcela (Falker)"));
    // pontos empilhados por faixa de 0,5 SPAD
    const stack = {};
    [...cl.plots].sort((a, b) => a.SPAD - b.SPAD).forEach((p) => {
      const bin = Math.round(p.SPAD * 2) / 2, k = (stack[bin] = (stack[bin] ?? 0) + 1);
      const c = s("circle", { cx: x(p.SPAD), cy: height - mg.b - 7 - (k - 1) * 13, r: 5.5, fill: doseColor(p.Dose), stroke: css("--ink-2"), "stroke-width": 1, style: "cursor:pointer" });
      c.addEventListener("pointermove", (e) => showTip(e, `parcela ${p.Ponto} · ${shortDate(p.Data)}`, [{ color: doseColor(p.Dose), value: `SPAD ${fmt(p.SPAD, 1)}`, label: `${p.Dose} kg N/ha` }, { value: p.Classe, label: "classe" }]));
      c.addEventListener("pointerleave", hideTip);
      c.addEventListener("click", () => goPlot(p.Ponto));
      svg.append(c);
    });
    legend(el, [0, 50, 75, 100].map((d) => ({ name: `${d} kg N/ha`, color: doseColor(d) })));
  });

  // validação agronômica
  $("#cl-valid-text", view).textContent = `Se as classes medem algo real, parcelas mais adubadas devem cair nas classes altas e ter mais altura e massa. `
    + `A classe acompanha a dose (Spearman ρ = ${fmt(D.rho_classe_dose, 2)}; p ${D.p_classe_dose < 0.001 ? "< 0,001" : `= ${fmt(D.p_classe_dose, 3)}`}). `
    + `Em 26/05, a única data com corte, acompanha a altura (ρ = ${fmt(D.rho_classe_altura, 2)}), a massa verde de folha (ρ = ${fmt(D.rho_classe_massa_verde_folha, 2)}) `
    + `e a massa seca total (ρ = ${fmt(D.rho_classe_massa_seca_total, 2)}). `
    + (() => {
      const n0hi = cl.validation.filter((r) => r.Classe === "Alta").reduce((a, r) => a + r.n_0kg, 0);
      const n100lo = cl.validation.filter((r) => r.Classe === "Baixa").reduce((a, r) => a + r.n_100kg, 0);
      return `${n0hi ? `${n0hi} parcela(s) sem N` : "Nenhuma parcela sem N"} ficou na classe alta, e ${n100lo ? `${n100lo} de 100 kg` : "nenhuma de 100 kg"} na classe baixa.`;
    })();
  $("#cl-valid", view).replaceChildren(localTable(["Classe", "n", "SPAD_medio", "n_0kg", "n_50kg", "n_75kg", "n_100kg", "Dose_media", "Altura_media_26_05", "Massa_verde_folha_26_05", "Massa_seca_total_26_05"],
    cl.validation, { formats: { SPAD_medio: 1, Dose_media: 0, Altura_media_26_05: 1, Massa_verde_folha_26_05: 2, Massa_seca_total_26_05: 2 },
      labels: { Classe: "classe", n: "parcelas", SPAD_medio: "SPAD médio", n_0kg: "0 kg", n_50kg: "50 kg", n_75kg: "75 kg", n_100kg: "100 kg", Dose_media: "dose média",
        Altura_media_26_05: "altura 26/05 (cm)", Massa_verde_folha_26_05: "m. verde folha 26/05 (t/ha)", Massa_seca_total_26_05: "m. seca total 26/05 (t/ha)" } }));

  // incerteza do Falker
  $("#cl-unc-text", view).textContent = `A classe de referência também tem erro. Cada parcela recebeu ${D.Leituras_por_parcela} leituras do Falker, em folhas diferentes, `
    + `e as leituras de uma mesma parcela variam em média ±${fmt(D.DP_entre_leituras, 1)} SPAD. A média de ${D.Leituras_por_parcela} leituras ainda tem um erro padrão de ${fmt(D.EP_medio, 2)} SPAD, `
    + `e o intervalo de confiança de 95 % (t de Student) vai, em média, ${fmt(D.IC95_meia_largura_media, 1)} SPAD para cada lado. Quando esse intervalo cruza um limite de classe, `
    + `outra medição da mesma parcela poderia dar a classe vizinha: a classe é incerta. Simulando a mesma parcela medida duas vezes, o Falker concorda consigo mesmo em ${pct(D.Teto_exato)} das parcelas `
    + `(kappa ${fmt(D.Teto_kappa, 2)}). Esse é o teto: nenhuma foto pode "acertar" mais do que a régua permite. Com ${D.Leituras_para_IC_1_SPAD} leituras por parcela, o intervalo cairia para ±1 SPAD `
    + `e o teto subiria para ${pct(D.Teto_exato_28_leituras)} (simulado com 28 leituras).`;
  $("#cl-unc-facts", view).replaceChildren(
    fact(`±${fmt(D.IC95_meia_largura_media, 1)}`, `SPAD de incerteza na média de ${D.Leituras_por_parcela} leituras (IC 95 %)`),
    fact(`${D.n_seguras} de ${D.n}`, "parcelas com classe segura no Falker"),
    fact(pct(D.Teto_exato), "teto: Falker × Falker na mesma parcela"),
    fact(`${D.Leituras_para_IC_1_SPAD}`, "leituras por parcela para uma incerteza de ±1 SPAD"));
  registerChart("cl-ci", () => {
    const el = $("#cl-ci", view), rows = [...cl.plots].sort((a, b) => a.SPAD - b.SPAD);
    const rowH = 11, mg = { l: 16, r: 16, t: 26, b: 42 }, height = mg.t + mg.b + rows.length * rowH;
    const { svg, width } = frame(el, height);
    const lo = Math.floor(Math.min(...rows.map((r) => r.IC95_min)) - 0.5), hi = Math.ceil(Math.max(...rows.map((r) => r.IC95_max)) + 0.5);
    const x = scale(lo, hi, mg.l, width - mg.r);
    cl.scheme.forEach((r) => {
      const a = x(Math.max(lo, r.SPAD_min ?? lo)), b = x(Math.min(hi, r.SPAD_max ?? hi));
      svg.append(s("rect", { x: a, y: mg.t, width: b - a, height: height - mg.t - mg.b, fill: clColor(r.Classe), opacity: 0.16 }),
        s("text", { x: (a + b) / 2, y: mg.t - 8, "text-anchor": "middle", class: "tick-label" }, r.Classe));
      if (r.SPAD_min != null) svg.append(s("line", { x1: a, x2: a, y1: mg.t, y2: height - mg.b, stroke: css("--ink-3"), "stroke-dasharray": "3 3" }));
    });
    for (const tk of niceTicks(lo, hi, 8).filter((v) => v >= lo && v <= hi)) svg.append(s("text", { x: x(tk), y: height - mg.b + 16, "text-anchor": "middle", class: "tick-label" }, fmt(tk, 0)));
    svg.append(s("line", { x1: mg.l, x2: width - mg.r, y1: height - mg.b, y2: height - mg.b, class: "baseline" }),
      s("text", { x: (mg.l + width - mg.r) / 2, y: height - 6, "text-anchor": "middle", class: "axis-label" }, "SPAD médio da parcela e IC 95 % (Falker, 7 leituras)"));
    const sure = css("--ink-2"), unsure = css("--series-2");
    rows.forEach((p, i) => {
      const cy = mg.t + i * rowH + rowH / 2, col = p.Classe_segura ? sure : unsure;
      const g = s("g", { style: "cursor:pointer" });
      g.append(s("line", { x1: x(p.IC95_min), x2: x(p.IC95_max), y1: cy, y2: cy, stroke: col, "stroke-width": 2, "stroke-linecap": "round" }),
        s("circle", { cx: x(p.SPAD), cy, r: 3.5, fill: col }),
        s("rect", { x: mg.l, y: cy - rowH / 2, width: width - mg.l - mg.r, height: rowH, fill: "transparent" }));
      g.addEventListener("pointermove", (e) => showTip(e, `parcela ${p.Ponto} · ${shortDate(p.Data)} · ${p.Dose} kg N`, [
        { color: col, value: `${fmt(p.SPAD, 1)} (IC ${fmt(p.IC95_min, 1)} a ${fmt(p.IC95_max, 1)})`, label: "SPAD" },
        { value: p.Classe_segura ? p.Classe : `${p.Classe_IC_min} a ${p.Classe_IC_max}`, label: p.Classe_segura ? "classe segura" : "classe incerta" }]));
      g.addEventListener("pointerleave", hideTip);
      g.addEventListener("click", () => goPlot(p.Ponto));
      svg.append(g);
    });
    legend(el, [{ name: "classe segura (intervalo dentro de uma classe)", color: sure, kind: "line" }, { name: "classe incerta (intervalo cruza um limite)", color: unsure, kind: "line" }]);
  }, { columns: ["Data", "Ponto", "Dose", "n_leituras", "SPAD", "SPAD_dp", "SPAD_ep", "IC95_min", "IC95_max", "Classe", "Classe_IC_min", "Classe_IC_max", "Classe_segura"],
    rows: cl.plots.map((p) => ({ ...p, Classe_segura: p.Classe_segura ? "sim" : "não" })),
    formats: { SPAD: 2, SPAD_dp: 2, SPAD_ep: 2, IC95_min: 1, IC95_max: 1 },
    labels: { n_leituras: "leituras", SPAD_dp: "desvio", SPAD_ep: "erro padrão", IC95_min: "IC mín.", IC95_max: "IC máx.", Classe_IC_min: "classe no IC mín.", Classe_IC_max: "classe no IC máx.", Classe_segura: "segura" } });
  $("#cl-ci-text", view).textContent = `Cada linha é uma parcela em uma data, ordenadas pelo SPAD. As linhas em laranja atravessam um limite de classe: `
    + `para essas ${D.n - D.n_seguras} parcelas, a classe do Falker não é garantida. Com três classes, as faixas têm ${fmt(D.Largura_classe_media, 1)} SPAD de largura; com cinco, teriam só 2,7, e quase nenhuma parcela teria classe segura.`;
  const SC = cl.scheme_comparison || [];
  $("#cl-schemes", view).replaceChildren(localTable(["Esquema", "Limites_SPAD", "Parcelas_por_classe", "n_seguras", "Teto_Falker_exato", "MLP_exato", "MLP_kappa", "MLP_dentro_incerteza", "Rede_exato", "Rede_kappa"],
    SC.map((r) => ({ ...r, Teto_Falker_exato: r.Teto_Falker_exato * 100, MLP_exato: r.MLP_exato * 100, MLP_dentro_incerteza: r.MLP_dentro_incerteza * 100, Rede_exato: r.Rede_exato * 100 })),
    { formats: { Teto_Falker_exato: 0, MLP_exato: 0, MLP_kappa: 2, MLP_dentro_incerteza: 0, Rede_exato: 0, Rede_kappa: 2 },
      labels: { Limites_SPAD: "limites (SPAD)", Parcelas_por_classe: "parcelas por classe", n_seguras: "fotos com classe segura", Teto_Falker_exato: "teto Falker (%)",
        MLP_exato: "rede pequena: exata (%)", MLP_kappa: "kappa", MLP_dentro_incerteza: "dentro da incerteza (%)", Rede_exato: "rede definitiva: exata (%)", Rede_kappa: "kappa" } }));
  const s3 = SC.find((r) => r.n_classes === 3), s5 = SC.find((r) => r.n_classes === 5);
  if (s3 && s5) $("#cl-schemes-text", view).textContent = `Com três classes o teto do Falker sobe de ${pct(s5.Teto_Falker_exato)} para ${pct(s3.Teto_Falker_exato)}, e a rede pequena (escolhida sem ver a data testada) `
    + `vai de ${pct(s5.MLP_exato)} para ${pct(s3.MLP_exato)} de classe exata, ${s3.MLP_exato >= s3.Teto_Falker_exato - 0.03 ? "praticamente no nível do próprio Falker" : "perto do nível do próprio Falker"}. `
    + `As versões com quatro classes ficam no meio do caminho. Três classes também correspondem à decisão de campo: adubar, observar, manter.`;

  // grades por data
  const dates = [...new Set(cl.plots.map((p) => p.Data))];
  const blocks = [...new Set(cl.plots.map((p) => p.Bloco))].sort();
  const doses = [100, 75, 50, 0];
  const SRC = {
    falker: { cls: "Classe", val: "SPAD", label: "Medida (Falker)", text: "Classe pelo SPAD médio do Falker (cerca de 7 leituras por parcela)." },
    indice: { cls: "Classe_melhor_indice", val: "SPAD_melhor_indice", label: `Pelo índice ${best.Variavel}`, text: `Classe pelo SPAD estimado com o índice ${best.Variavel}, por uma reta ajustada nas outras duas datas. Células vazias: parcela sem foto nesta data (18/05 tem 7).` },
    aninhado: { cls: "Classe_indice_aninhado", val: "SPAD_indice_aninhado", label: "Pelo índice escolhido sem a data", text: "Classe pelo índice escolhido só com as outras duas datas (a escolha muda por data: veja a tabela de concordância)." },
    mlp: { cls: "Classe_mlp", val: "SPAD_mlp", label: "Pela rede pequena", text: "Classe pelo SPAD da rede neural pequena, com a configuração escolhida sem ver a data testada: o desempenho esperado numa coleta nova." },
    rede: { cls: "Classe_rede", val: "SPAD_rede", label: "Pela rede neural", text: "Classe pelo SPAD previsto pela rede neural definitiva, para uma data fora do treino." },
  };
  const drawGrid = (src) => {
    const { cls, val, text } = SRC[src];
    $("#cl-source-text", view).textContent = text;
    $("#cl-grid", view).replaceChildren(...dates.map((d) => h("div", { class: "cl-grid" },
      h("div", { class: "cl-date" }, shortDate(d)),
      h("div", { class: "cl-cells", style: `grid-template-columns: 4.2em repeat(${blocks.length}, minmax(0,1fr))` },
        h("div"), ...blocks.map((b) => h("div", { class: "cl-head small muted" }, b.replace("Bloco ", "B"))),
        ...doses.flatMap((dose) => [h("div", { class: "cl-head small muted" }, `${dose} kg`),
          ...blocks.map((b) => {
            const p = cl.plots.find((x) => x.Data === d && x.Bloco === b && x.Dose === dose);
            const c = p?.[cls];
            if (!p || !c) return h("div", { class: "cl-cell empty" }, "–");
            const wrong = src !== "falker" && c !== p.Classe;
            const cell = h("button", { type: "button", class: `cl-cell${wrong ? " miss" : ""}`, style: `background:${clColor(c)};color:${clInk(c)}`, onclick: () => goPlot(p.Ponto) },
              h("strong", {}, `P${p.Ponto}`), h("span", {}, c), h("span", { class: "isn" }, `SPAD ${fmt(p[val], 1)}`),
              src === "falker" && !p.Classe_segura ? h("span", { class: "isn" }, `incerta: ${p.Classe_IC_min} a ${p.Classe_IC_max}`) : null);
            cell.addEventListener("pointermove", (e) => showTip(e, `parcela ${p.Ponto} · ${shortDate(d)}`, [
              { color: clColor(p.Classe), value: `${p.Classe} (SPAD ${fmt(p.SPAD, 1)})`, label: "Falker" },
              ...(src !== "falker" ? [{ color: clColor(c), value: `${c} (SPAD ${fmt(p[val], 1)})`, label: SRC[src].label }] : [])]));
            cell.addEventListener("pointerleave", hideTip);
            return cell;
          })])))));
    const rows = cl.plots.filter((p) => p[cls]);
    const miss = src === "falker" ? 0 : rows.filter((p) => p[cls] !== p.Classe).length;
    $("#cl-grid-text", view).textContent = src === "falker"
      ? `A classe sobe com a dose nas três datas. Em 26/05 o nível geral é mais baixo. ${cl.plots.filter((p) => !p.Classe_segura).length} das ${cl.plots.length} parcelas têm classe incerta: o intervalo de confiança da média cruza um limite.`
      : `${rows.length - miss} de ${rows.length} parcelas com a mesma classe do Falker. As células com contorno tracejado são as que erraram.`;
  };
  legend($("#cl-legend", view), CL_NAMES.map((c) => ({ name: c, color: clColor(c), kind: "rect" })));
  options($("#cl-source", view), Object.entries(SRC).map(([k, v]) => [k, v.label]), drawGrid, "falker");

  // ranking dos índices
  const top = cl.index_ranking.slice(0, 15);
  registerChart("cl-rank", () => hbar($("#cl-rank", view), {
    xLabel: "kappa ponderado (reta ajustada em duas datas, testada na terceira)", valueFmt: (v) => fmt(v, 2),
    items: top.map((r, i) => ({ label: r.Variavel, value: r.Kappa_ponderado, highlight: i === 0, color: i === 0 ? css("--series-1") : css("--neutral"),
      note: `exata ${pct(r.Acerto_exato)} · até 1 classe ${pct(r.Acerto_1_classe)} · R² do SPAD ${fmt(r.R2_SPAD_LODO, 2)}` })),
  }), { columns: ["Rank", "Variavel", "rho_SPAD", "R2_SPAD_LODO", "MAE_SPAD_LODO", "Acerto_exato", "Acerto_1_classe", "Kappa_ponderado"], rows: cl.index_ranking.slice(0, 30),
    formats: { rho_SPAD: 3, R2_SPAD_LODO: 3, MAE_SPAD_LODO: 2, Acerto_exato: 3, Acerto_1_classe: 3, Kappa_ponderado: 3 },
    labels: { rho_SPAD: "ρ com SPAD", R2_SPAD_LODO: "R² do SPAD", MAE_SPAD_LODO: "erro SPAD", Acerto_exato: "exata", Acerto_1_classe: "até 1 classe", Kappa_ponderado: "kappa" } });
  const fam = [...new Set(top.map((r) => r.Indice))];
  $("#cl-rank-text", view).textContent = `O melhor, ${best.Variavel}, acerta a classe exata em ${pct(best.Acerto_exato)} das ${best.n} parcelas com foto e erra no máximo uma classe em ${pct(best.Acerto_1_classe)} (kappa ${fmt(best.Kappa_ponderado, 2)}, ${kappaWord(best.Kappa_ponderado)}). `
    + `Os primeiros são variações da mesma informação de cor (${fam.slice(0, 7).join(", ")}): a proporção de azul em relação ao vermelho e a saturação da folha. `
    + `Esse número é otimista, porque o índice foi escolhido olhando o resultado; o valor honesto está na linha "escolhido sem a data testada" abaixo.`;

  // concordância
  const pc = (v) => (v == null ? null : v * 100);
  $("#cl-agree", view).replaceChildren(localTable(["Metodo", "Escopo", "Indice", "n", "Acerto_exato", "Kappa_ponderado", "Acerto_seguras", "Acerto_dentro_incerteza", "Previsto_acima", "Previsto_abaixo"],
    cl.agreement.map((a) => ({ ...a, Escopo: a.Escopo === "Todas as datas" ? a.Escopo : shortDate(a.Escopo), Acerto_exato: pc(a.Acerto_exato),
      Acerto_seguras: pc(a.Acerto_seguras), Acerto_dentro_incerteza: pc(a.Acerto_dentro_incerteza) })),
    { formats: { Acerto_exato: 0, Kappa_ponderado: 2, Acerto_seguras: 0, Acerto_dentro_incerteza: 0 },
      labels: { Metodo: "método", Escopo: "data", Indice: "índice escolhido", Acerto_exato: "classe exata (%)", Kappa_ponderado: "kappa",
        Acerto_seguras: "só classes seguras (%)", Acerto_dentro_incerteza: "dentro da incerteza (%)", Previsto_acima: "acima", Previsto_abaixo: "abaixo" } }));

  const methods = [...new Set(cl.confusion.map((c) => c.Metodo))];
  const drawConf = (mth) => {
    const conf = cl.confusion.filter((c) => c.Metodo === mth);
    const n = (a, b) => conf.find((c) => c.Classe_medida === a && c.Classe_prevista === b)?.n ?? 0;
    const max = Math.max(...conf.map((c) => c.n));
    $("#cl-confusion", view).replaceChildren(h("div", { class: "table-wrap" }, h("table", { class: "data cl-conf" },
      h("thead", {}, h("tr", {}, h("th", {}, "medida ↓ · prevista →"), CL_NAMES.map((c) => h("th", { class: "num" }, c)))),
      h("tbody", {}, CL_NAMES.map((a) => h("tr", {}, h("td", {}, a), CL_NAMES.map((b) => {
        const v = n(a, b);
        return h("td", { class: `num${a === b ? " diag" : ""}`, style: v ? `background:color-mix(in srgb, ${css("--series-1")} ${Math.round(12 + 60 * v / max)}%, transparent)` : "" }, v || "·");
      })))))));
    const ag = cl.agreement.find((a) => a.Metodo === mth && a.Escopo === "Todas as datas");
    $("#cl-confusion-text", view).textContent = `Diagonal = classe certa. ${pct(ag.Acerto_exato)} na diagonal e ${pct(ag.Acerto_1_classe)} na diagonal ou numa casa vizinha. `
      + `Quando erra, a previsão fica ${ag.Previsto_acima > ag.Previsto_abaixo ? "mais vezes acima" : ag.Previsto_acima < ag.Previsto_abaixo ? "mais vezes abaixo" : "tanto acima quanto abaixo"} da classe medida (${ag.Previsto_acima} acima, ${ag.Previsto_abaixo} abaixo).`;
  };
  options($("#cl-conf-method", view), methods.map((m) => [m, m.replace(/ \(.*\)$/, "")]), drawConf, methods[0]);

  serverTable($("#cl-table", view), "chl_class_plots", {
    limit: 12, sort: "Ponto",
    hide: ["Classe_idx", "Tem_foto", "SPAD_dp", "Bloco", "Tratamento", "z"],
    formats: { SPAD: 1, SPAD_rede: 1, SPAD_melhor_indice: 1, SPAD_indice_aninhado: 1 },
    labels: { n_leituras: "leituras", SPAD_rede: "SPAD rede", SPAD_melhor_indice: "SPAD melhor índice", SPAD_indice_aninhado: "SPAD aninhado", Indice_aninhado: "índice aninhado",
      Classe_melhor_indice: "classe melhor índice", Classe_indice_aninhado: "classe aninhado", Classe_rede: "classe rede" },
  });

  $("#cl-refs", view).replaceChildren(...cl.references.map((r) => h("p", {},
    r.Referencia, r.URL ? [" ", h("a", { href: r.URL, target: "_blank", rel: "noopener" }, "link")] : null,
    h("br"), h("span", { class: "small muted" }, `Usado para: ${r.Uso}`))));
};
