import React from "react";
import {
  Bar, BarChart, CartesianGrid, LabelList, Line, LineChart, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";

const fmt = (v, digits = 1) =>
  v == null || Number.isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: digits });

function axisProps(t) {
  return {
    stroke: t["--axis"],
    tick: { fill: t["--text-muted"], fontSize: 11 },
    tickLine: false,
  };
}

export function Legend({ series, theme }) {
  return (
    <div className="legend">
      {series.map((s) => (
        <span key={s.key}>
          <i className="swatch" style={{ background: theme[s.color] }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

function SeriesTooltip({ active, payload, label, series, theme, unit, digits, labelPrefix = "" }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip">
      <div className="t">{labelPrefix}{label}</div>
      {series.map((s) => {
        const p = payload.find((x) => x.dataKey === s.key);
        if (!p || p.value == null) return null;
        return (
          <div className="row" key={s.key}>
            <span><i className="swatch" style={{ background: theme[s.color] }} /> {s.label}</span>
            <b>{fmt(p.value, digits)} {unit}</b>
          </div>
        );
      })}
    </div>
  );
}

/** Grouped bars: one group per quant, one bar per backend. */
export function GroupedBars({ data, series, theme, unit, digits = 1, height = 260 }) {
  return (
    <>
      {series.length > 1 && <Legend series={series} theme={theme} />}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} barGap={2} barCategoryGap="22%" margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke={theme["--grid"]} />
          <XAxis dataKey="quant" {...axisProps(theme)} />
          <YAxis {...axisProps(theme)} axisLine={false} width={48} />
          <Tooltip cursor={{ fill: theme["--grid"], opacity: 0.5 }}
            content={<SeriesTooltip series={series} theme={theme} unit={unit} digits={digits} />} />
          {series.map((s) => (
            <Bar key={s.key} dataKey={s.key} fill={theme[s.color]} radius={[4, 4, 0, 0]} maxBarSize={36}
              isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}

/** Lines over concurrency, one per backend. */
export function ConcurrencyLines({ data, series, theme, unit, height = 260 }) {
  return (
    <>
      {series.length > 1 && <Legend series={series} theme={theme} />}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke={theme["--grid"]} />
          <XAxis dataKey="concurrency" {...axisProps(theme)}
            label={{ value: "concurrent requests", position: "insideBottomRight", offset: -2, fill: theme["--text-muted"], fontSize: 11 }} />
          <YAxis {...axisProps(theme)} axisLine={false} width={48} />
          <Tooltip cursor={{ stroke: theme["--axis"] }}
            content={<SeriesTooltip series={series} theme={theme} unit={unit} labelPrefix="concurrency " />} />
          {series.map((s) => (
            <Line key={s.key} dataKey={s.key} stroke={theme[s.color]} strokeWidth={2} isAnimationActive={false}
              dot={{ r: 4, strokeWidth: 2, stroke: theme["--surface-1"], fill: theme[s.color] }}
              activeDot={{ r: 5, strokeWidth: 2, stroke: theme["--surface-1"] }} connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </>
  );
}

function QualityTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="tooltip">
      <div className="t">{d.quant}</div>
      <div className="row"><span>decode speed</span><b>{fmt(d.tps)} tok/s</b></div>
      <div className="row"><span>perplexity</span><b>{fmt(d.ppl, 3)}</b></div>
      <div className="row"><span>file size</span><b>{fmt(d.size, 0)} MB</b></div>
    </div>
  );
}

/** Quality vs speed: each point is one quant. Lower perplexity is better, further right is faster. */
export function QualityScatter({ data, theme, height = 280 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 16, right: 24, left: 0, bottom: 8 }}>
        <CartesianGrid stroke={theme["--grid"]} />
        <XAxis type="number" dataKey="tps" name="tok/s" {...axisProps(theme)} domain={["auto", "auto"]}
          label={{ value: "decode tok/s →", position: "insideBottomRight", offset: -4, fill: theme["--text-muted"], fontSize: 11 }} />
        <YAxis type="number" dataKey="ppl" name="perplexity" {...axisProps(theme)} axisLine={false} width={52}
          domain={["auto", "auto"]}
          label={{ value: "perplexity (lower is better)", angle: -90, position: "insideLeft", offset: 12, fill: theme["--text-muted"], fontSize: 11, style: { textAnchor: "middle" } }} />
        <ZAxis range={[90, 90]} />
        <Tooltip cursor={false} content={<QualityTooltip />} />
        <Scatter data={data} fill={theme["--series-1"]} stroke={theme["--surface-1"]} strokeWidth={2}
          isAnimationActive={false}>
          <LabelList dataKey="quant" position="top" offset={8} fill={theme["--text-secondary"]} fontSize={11} />
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/** Small single-series sparkline for the live panel. */
export function LiveLine({ data, dataKey, theme, unit }) {
  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke={theme["--grid"]} />
        <XAxis dataKey="i" hide />
        <YAxis {...axisProps(theme)} axisLine={false} width={48} />
        <Tooltip cursor={{ stroke: theme["--axis"] }}
          content={({ active, payload }) => active && payload?.length ? (
            <div className="tooltip"><div className="row"><span>{payload[0].payload.time}</span>
              <b>{fmt(payload[0].value)} {unit}</b></div></div>) : null} />
        <Line dataKey={dataKey} stroke={theme["--series-1"]} strokeWidth={2} dot={false}
          isAnimationActive={false} connectNulls />
      </LineChart>
    </ResponsiveContainer>
  );
}
