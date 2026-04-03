---
title: Holdout Predictions vs Reality
sidebar: false
toc: true
---

# Holdout Predictions vs Reality

This notebook compares holdout prediction probabilities with realized outcomes using the current backtest output.

```js
import * as Plot from "npm:@observablehq/plot";
import * as Inputs from "npm:@observablehq/inputs";
import * as d3 from "npm:d3";
```

```js
raw = FileAttachment("data/backtest_results.csv").csv({ typed: true })
```

```js
holdout = raw
  .filter((d) => d.pred_prob_class_1 != null && d.target != null)
  .map((d) => ({
    ...d,
    pred_prob_class_1: +d.pred_prob_class_1,
    target: +d.target,
    market_prob_mean: +d.market_prob_mean,
    edge: +d.edge,
    event_start_ts_utc: d.event_start_ts_utc ? new Date(d.event_start_ts_utc) : null
  }))
```

```js
summary = {
  rows: holdout.length,
  realized_home_win_rate: d3.mean(holdout, (d) => d.target),
  average_model_probability: d3.mean(holdout, (d) => d.pred_prob_class_1),
  average_market_probability: d3.mean(holdout, (d) => d.market_prob_mean)
}
```

```js
summary
```

## Distribution of Model Probabilities

```js
Plot.plot({
  width: 900,
  height: 340,
  x: { label: "Predicted home-win probability" },
  y: { label: "Count" },
  marks: [
    Plot.rectY(
      holdout,
      Plot.binX({ y: "count" }, { x: "pred_prob_class_1", thresholds: 30, fill: "#0f766e" })
    ),
    Plot.ruleY([0])
  ]
})
```

## Reliability (Calibration): Prediction vs Reality

```js
viewof bins = Inputs.range([5, 30], { value: 12, step: 1, label: "Calibration bins" })
```

```js
binWidth = 1 / bins
calibration = d3
  .range(bins)
  .map((i) => {
    const lo = i * binWidth
    const hi = lo + binWidth
    const rows = holdout.filter((d) => d.pred_prob_class_1 >= lo && (i === bins - 1 ? d.pred_prob_class_1 <= hi : d.pred_prob_class_1 < hi))
    return {
      bin: i,
      x: lo + binWidth / 2,
      n: rows.length,
      predicted: rows.length ? d3.mean(rows, (d) => d.pred_prob_class_1) : null,
      actual: rows.length ? d3.mean(rows, (d) => d.target) : null
    }
  })
  .filter((d) => d.n > 0)
```

```js
Plot.plot({
  width: 900,
  height: 380,
  x: { label: "Predicted probability (bin center)", domain: [0, 1] },
  y: { label: "Observed frequency", domain: [0, 1] },
  color: { legend: true },
  marks: [
    Plot.line([{ x: 0, y: 0 }, { x: 1, y: 1 }], { x: "x", y: "y", stroke: "#9ca3af", strokeDasharray: "4,4" }),
    Plot.lineY(calibration, { x: "predicted", y: "actual", stroke: "#1d4ed8" }),
    Plot.dot(calibration, { x: "predicted", y: "actual", r: "n", fill: "#1d4ed8", tip: true })
  ]
})
```

## Classification Outcomes at a Probability Cutoff

```js
viewof cutoff = Inputs.range([0.1, 0.9], { value: 0.5, step: 0.01, label: "Home-win decision cutoff" })
```

```js
scored = holdout.map((d) => ({
  ...d,
  predicted_label: d.pred_prob_class_1 >= cutoff ? 1 : 0
}))

confusion = [
  { bucket: "True Positive", value: scored.filter((d) => d.predicted_label === 1 && d.target === 1).length },
  { bucket: "False Positive", value: scored.filter((d) => d.predicted_label === 1 && d.target === 0).length },
  { bucket: "True Negative", value: scored.filter((d) => d.predicted_label === 0 && d.target === 0).length },
  { bucket: "False Negative", value: scored.filter((d) => d.predicted_label === 0 && d.target === 1).length }
]
```

```js
Plot.plot({
  width: 900,
  height: 320,
  y: { label: "Rows" },
  x: { label: null },
  marks: [
    Plot.barY(confusion, { x: "bucket", y: "value", fill: "bucket", tip: true }),
    Plot.ruleY([0])
  ]
})
```

## Signal and PnL Sensitivity by Edge Threshold

```js
viewof minEdge = Inputs.range([0, 0.2], { value: 0.02, step: 0.005, label: "Minimum edge threshold" })
```

```js
strategy = holdout.map((d, i) => {
  const signal = d.edge >= minEdge ? 1 : 0
  const pnl = signal ? ((d.target * (1 / d.market_prob_mean)) - 1) : 0
  return {
    i,
    signal,
    pnl,
    cum_pnl: 0
  }
})

let running = 0
for (const r of strategy) {
  running += r.pnl
  r.cum_pnl = running
}

strategySummary = {
  signals: d3.sum(strategy, (d) => d.signal),
  total_rows: strategy.length,
  signal_rate: d3.mean(strategy, (d) => d.signal),
  final_cum_pnl_units: strategy.length ? strategy[strategy.length - 1].cum_pnl : 0,
  avg_pnl_per_signal: d3.mean(strategy.filter((d) => d.signal === 1), (d) => d.pnl)
}
```

```js
strategySummary
```

```js
Plot.plot({
  width: 900,
  height: 340,
  x: { label: "Holdout row index" },
  y: { label: "Cumulative PnL (units)" },
  marks: [
    Plot.lineY(strategy, { x: "i", y: "cum_pnl", stroke: "#b91c1c" }),
    Plot.ruleY([0])
  ]
})
```

## Inspect Individual Rows

```js
viewof rowsToShow = Inputs.range([10, 200], { value: 30, step: 5, label: "Rows in table" })
```

```js
Inputs.table(
  holdout.slice(0, rowsToShow).map((d) => ({
    event: d.event_name,
    predicted_home_win_prob: d.pred_prob_class_1,
    market_home_win_prob: d.market_prob_mean,
    realized_home_win: d.target,
    edge: d.edge
  }))
)
```
