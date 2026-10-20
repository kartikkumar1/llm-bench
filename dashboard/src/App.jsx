import React, { useEffect, useMemo, useState } from "react";
import { BACKENDS, useTheme } from "./theme.js";
import { ConcurrencyLines, GroupedBars, LiveLine, QualityScatter } from "./charts.jsx";

const QUANT_ORDER = ["bf16", "fp16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"];
const fmt = (v, d = 1) => (v == null ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

function useLive() {
  const [live, setLive] = useState(null);
  const [health, setHealth] = useState(null);
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try { setLive(await getJSON("/api/live")); setHealth(await getJSON("/health")); }
      catch { setHealth(null); }
      if (!stop) setTimeout(tick, 2000);
    };
    tick();
    return () => { stop = true; };
  }, []);
  return { live, health };
}

/** Pivot runs into chart rows: one row per quant, one column per backend. */
function pivot(runs, pick) {
  const byQuant = {};
  for (const run of runs) {
    const v = pick(run);
    if (v == null) continue;
    (byQuant[run.quant] ??= { quant: run.quant })[run.backend] = v;
  }
  return Object.values(byQuant).sort((a, b) => QUANT_ORDER.indexOf(a.quant) - QUANT_ORDER.indexOf(b.quant));
}

function Tile({ label, value, note }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

function Card({ title, sub, wide, children }) {
  return (
    <section className={`card${wide ? " wide" : ""}`}>
      <h2>{title}</h2>
      {sub && <p className="sub">{sub}</p>}
      {children}
    </section>
  );
}

export default function App() {
  const theme = useTheme();
  const { live, health } = useLive();
  const [runIds, setRunIds] = useState([]);
  const [runId, setRunId] = useState("latest");
  const [report, setReport] = useState(null);
  const [hw, setHw] = useState(null);
  const [error, setError] = useState("");
  const [promptLen, setPromptLen] = useState(null);
  const [conc, setConc] = useState(null);
  const [quant, setQuant] = useState(null);

  useEffect(() => {
    getJSON("/api/results").then(setRunIds).catch(() => {});
    getJSON("/api/hardware").then(setHw).catch(() => {});
  }, []);

  useEffect(() => {
    setError("");
    getJSON(`/api/results/${runId}`).then(setReport).catch((e) => { setReport(null); setError(e.message); });
  }, [runId]);

  const runs = useMemo(() => (report?.runs ?? []).filter((r) => r.rows?.length), [report]);
  const promptLens = useMemo(() => [...new Set(runs.flatMap((r) => r.rows.map((x) => x.prompt_tokens_target)))], [runs]);
  const concs = useMemo(() => [...new Set(runs.flatMap((r) => r.rows.map((x) => x.concurrency)))].sort((a, b) => a - b), [runs]);
  const quants = useMemo(() => [...new Set(runs.map((r) => r.quant))]
    .sort((a, b) => QUANT_ORDER.indexOf(a) - QUANT_ORDER.indexOf(b)), [runs]);

  // Default filters once a report arrives; keep the user's choice if it still exists.
  useEffect(() => {
    if (!promptLens.includes(promptLen)) setPromptLen(promptLens[0] ?? null);
    if (!concs.includes(conc)) setConc(concs[0] ?? null);
    if (!quants.includes(quant)) setQuant(quants.includes("q4_k_m") ? "q4_k_m" : quants[0] ?? null);
  }, [promptLens, concs, quants]);

  const series = BACKENDS.filter((b) => runs.some((r) => r.backend === b.key));
  const row = (run) => run.rows.find((x) => x.prompt_tokens_target === promptLen && x.concurrency === conc);

  const decode = pivot(runs, (r) => row(r)?.decode_tps_mean);
  const ttft = pivot(runs, (r) => row(r)?.ttft_ms_p50);
  const memory = pivot(runs, (r) => r.memory?.peak_rss_mb);
  const throughputByConc = concs.map((c) => {
    const out = { concurrency: c };
    for (const r of runs.filter((r) => r.quant === quant || r.backend === "hf_transformers")) {
      const x = r.rows.find((x) => x.prompt_tokens_target === promptLen && x.concurrency === c);
      if (x?.ok) out[r.backend] = x.throughput_tps;
    }
    return out;
  });
  const quality = runs
    .filter((r) => r.backend === "llamacpp_server" && r.perplexity != null)
    .map((r) => ({ quant: r.quant, ppl: r.perplexity, tps: row(r)?.decode_tps_mean, size: r.model_file_mb }))
    .filter((d) => d.tps != null);

  const best = runs.map((r) => ({ r, x: row(r) })).filter((d) => d.x?.ok)
    .sort((a, b) => b.x.decode_tps_mean - a.x.decode_tps_mean)[0];
  const leanest = [...runs].filter((r) => r.memory).sort((a, b) => a.memory.peak_rss_mb - b.memory.peak_rss_mb)[0];

  const hardware = report?.hardware ?? hw?.hardware;
  const resolved = report?.config ?? hw?.config;
  const label = (key) => BACKENDS.find((b) => b.key === key)?.label ?? key;

  const liveSeries = (live?.recent ?? []).map((r, i) => ({
    i, time: new Date(r.ts * 1000).toLocaleTimeString(), e2e: r.e2e_ms, tps: r.decode_tps ?? null,
  }));
  const upstreamOk = health?.upstream_status === "ok";

  return (
    <div className="app">
      <header>
        <div>
          <h1>LLM inference benchmark</h1>
          <p className="sub">
            {hardware ? `${hardware.cpu.name} · ${hardware.ram.total_gb} GB RAM · ` : ""}
            {resolved ? `device ${resolved.hardware.device} · ${resolved.hardware.threads} threads · model ${resolved.model.name}` : "no hardware info yet"}
          </p>
        </div>
        <span className="status">
          <i className="dot" style={{ background: upstreamOk ? theme["--good"] : theme["--critical"] }} />
          {health ? `gateway ok · upstream ${health.upstream_status}` : "gateway unreachable"}
        </span>
      </header>

      <div className="filters">
        <label>Run
          <select value={runId} onChange={(e) => setRunId(e.target.value)}>
            <option value="latest">latest</option>
            {runIds.map((r) => <option key={r.id} value={r.id}>{r.id}</option>)}
          </select>
        </label>
        <label>Prompt tokens
          <select value={promptLen ?? ""} onChange={(e) => setPromptLen(Number(e.target.value))}>
            {promptLens.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label>Concurrency
          <select value={conc ?? ""} onChange={(e) => setConc(Number(e.target.value))}>
            {concs.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
      </div>

      {!runs.length ? (
        <div className="empty">
          <h2>No benchmark results yet</h2>
          <p className="sub">{error && `(${error}) `}Run <code>python -m bench.run</code>, then refresh this page.</p>
        </div>
      ) : (
        <>
          <div className="tiles">
            <Tile label="Fastest decode" value={best ? `${fmt(best.x.decode_tps_mean)} tok/s` : "–"}
              note={best && `${label(best.r.backend)} · ${best.r.quant}`} />
            <Tile label="Lowest peak RAM" value={leanest ? `${fmt(leanest.memory.peak_rss_mb, 0)} MB` : "–"}
              note={leanest && `${label(leanest.backend)} · ${leanest.quant}`} />
            <Tile label="Configurations" value={runs.length} note={`${quants.length} quants × ${series.length} backends`} />
            <Tile label="Run duration" value={`${fmt(report.duration_s / 60)} min`}
              note={new Date(report.started_at).toLocaleString()} />
          </div>

          <div className="grid">
            <Card title="Decode speed" sub="Mean tokens per second after the first token, per request. Higher is better.">
              <GroupedBars data={decode} series={series} theme={theme} unit="tok/s" />
            </Card>
            <Card title="Time to first token (p50)" sub="Prompt processing plus queueing, in milliseconds. Lower is better.">
              <GroupedBars data={ttft} series={series} theme={theme} unit="ms" digits={0} />
            </Card>
            <Card title="Peak memory (RSS)" sub="Resident memory of the server process tree, in MB. On an integrated GPU, shared GPU memory is reported separately in the table.">
              <GroupedBars data={memory} series={series} theme={theme} unit="MB" digits={0} />
            </Card>
            <Card title={`Throughput vs concurrency · ${quant ?? ""}`}
              sub="Total tokens per second across all in-flight requests.">
              <div className="filters" style={{ margin: "0 0 8px" }}>
                <label>Quant
                  <select value={quant ?? ""} onChange={(e) => setQuant(e.target.value)}>
                    {quants.filter((q) => q !== "bf16").map((q) => <option key={q} value={q}>{q}</option>)}
                  </select>
                </label>
              </div>
              <ConcurrencyLines data={throughputByConc} series={series} theme={theme} unit="tok/s" />
            </Card>
            <Card wide title="Quality vs speed (llama.cpp server)"
              sub="Each point is one quantization level. Down and to the right is better: faster and closer to full-precision quality.">
              {quality.length ? <QualityScatter data={quality} theme={theme} />
                : <p className="muted">No perplexity scores in this run. Run <code>python -m bench.run --perplexity</code>.</p>}
            </Card>
          </div>
        </>
      )}

      <div className="grid" style={{ marginTop: 16 }}>
        <Card title="Live gateway: end-to-end latency" sub={`Last ${live?.window ?? 0} requests · in flight ${live?.inflight ?? 0} · errors ${live?.errors ?? 0}`}>
          {liveSeries.length ? <LiveLine data={liveSeries} dataKey="e2e" theme={theme} unit="ms" />
            : <p className="muted">No traffic yet. Send a request to <code>/v1/chat/completions</code>.</p>}
        </Card>
        <Card title="Live gateway: decode speed" sub={`Average ${fmt(live?.decode_tps_avg)} tok/s · TTFT ${fmt(live?.ttft_ms_avg, 0)} ms`}>
          {liveSeries.length ? <LiveLine data={liveSeries} dataKey="tps" theme={theme} unit="tok/s" />
            : <p className="muted">Waiting for streamed requests.</p>}
        </Card>
      </div>

      {runs.length > 0 && (
        <section className="card wide" style={{ marginTop: 16 }}>
          <h2>All results</h2>
          <p className="sub">Every configuration in this run, at every prompt length and concurrency.</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Backend</th><th>Quant</th><th>Prompt</th><th>Conc.</th><th>TTFT p50</th><th>TTFT p95</th>
                  <th>E2E p50</th><th>ITL</th><th>Decode tok/s</th><th>Throughput</th><th>RSS MB</th><th>GPU MB</th>
                  <th>File MB</th><th>PPL</th></tr>
              </thead>
              <tbody>
                {runs.flatMap((r) => r.rows.map((x) => (
                  <tr key={`${r.backend}-${r.quant}-${x.prompt_tokens_target}-${x.concurrency}`}>
                    <td>{label(r.backend)}</td><td>{r.quant}</td><td>{x.prompt_tokens_target}</td><td>{x.concurrency}</td>
                    <td>{fmt(x.ttft_ms_p50, 0)}</td><td>{fmt(x.ttft_ms_p95, 0)}</td><td>{fmt(x.e2e_ms_p50, 0)}</td>
                    <td>{fmt(x.itl_ms_mean)}</td><td>{fmt(x.decode_tps_mean)}</td><td>{fmt(x.throughput_tps)}</td>
                    <td>{fmt(r.memory?.peak_rss_mb, 0)}</td><td>{fmt(r.memory?.peak_gpu_mb, 0)}</td>
                    <td>{fmt(r.model_file_mb, 0)}</td><td>{fmt(r.perplexity, 3)}</td>
                  </tr>
                )))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
