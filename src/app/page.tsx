"use client";

import React, { useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from "recharts";
import { motion } from "framer-motion";

/*
  Emotion Likelihood Explorer (probability-normalized)
  ------------------------------------------------------------
  横軸: θ (deg)
  縦軸: 各感情 (Anger / Sad / Neutral / Joy) の確率 P(Emotion | θ, ...)

  変更点（重要）:
  - emotionScoresFromS() で得た4カテゴリのスコアを
      p_k = (s_k + ε) / Σ_j (s_j + ε)
    に正規化して確率として使用
  - Y軸ラベルを "probability" に変更
*/

const deg2rad = (d: number) => (Math.PI / 180) * d;

function enumerateCandX(q: number[]): number[][] {
  const xs: number[][] = [];
  for (let x1 = 0; x1 <= q[0]; x1++)
    for (let x2 = 0; x2 <= q[1]; x2++)
      for (let x3 = 0; x3 <= q[2]; x3++)
        for (let x4 = 0; x4 <= q[3]; x4++) xs.push([x1, x2, x3, x4]);
  return xs;
}

function dot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function add(a: number[], b: number[], sign = 1) {
  return a.map((ai, i) => ai + sign * b[i]);
}

function utility(thetaRad: number, w: number[], x: number[], q: number[]) {
  const xOther = add(q, x, -1);
  return Math.cos(thetaRad) * dot(w, x) + Math.sin(thetaRad) * dot(w, xOther);
}

function satisfaction(thetaRad: number, w: number[], x: number[], q: number[], beta: number, candX: number[][]) {
  let umax = -Infinity;
  for (const xx of candX) umax = Math.max(umax, utility(thetaRad, w, xx, q));
  const u = utility(thetaRad, w, x, q);
  return Math.exp(beta * (u - umax)); // in (0,1]
}

function emotionScoresFromS(S: number, tau1: number, tau2: number, sadBand: number) {
  const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
  const anger = clamp((tau1 - S) / tau1);
  const sad = Math.abs(S - tau1) <= sadBand ? 1 - Math.abs(S - tau1) / sadBand : 0;
  let neutral = 0;
  if (S > tau1 && S < tau2) {
    const mid = (tau1 + tau2) / 2;
    const width = (tau2 - tau1) / 2;
    neutral = clamp(1 - Math.abs(S - mid) / width);
  }
  const joy = clamp((S - tau2) / (1 - tau2));
  return { Anger: anger, Sad: sad, Neutral: neutral, Joy: joy };
}

// --- NEW: normalize 4-category scores into probabilities ---
function scoresToProbs(scores: { [k: string]: number }, eps = 1e-9) {
  const total =
    (scores.Anger + eps) +
    (scores.Sad + eps) +
    (scores.Neutral + eps) +
    (scores.Joy + eps);
  return {
    Anger: (scores.Anger + eps) / total,
    Sad: (scores.Sad + eps) / total,
    Neutral: (scores.Neutral + eps) / total,
    Joy: (scores.Joy + eps) / total,
  };
}

const EMO_COLORS: Record<string, string> = {
  Anger: "#d62728",
  Sad: "#9467bd",
  Neutral: "#7f7f7f",
  Joy: "#2ca02c",
};

function NumberField({
  label, value, onChange, step = 0.01, min, max, suffix,
}: {
  label: string; value: number; onChange: (v: number) => void;
  step?: number; min?: number; max?: number; suffix?: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <Label className="w-28 text-sm text-muted-foreground">{label}</Label>
      <Input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-28"
      />
      {suffix ? <span className="text-sm text-muted-foreground">{suffix}</span> : null}
    </div>
  );
}

function SliderField({
  label, value, onChange, min = 0, max = 1, step = 0.01,
}: {
  label: string; value: number; onChange: (v: number[]) => void;
  min?: number; max?: number; step?: number;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between">
        <Label className="text-sm text-muted-foreground">{label}</Label>
        <span className="text-xs tabular-nums">{value.toFixed(3)}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={(arr) => onChange(arr)} />
    </div>
  );
}

function normalize(w: number[]) {
  const s = w.reduce((a, b) => a + b, 0) || 1;
  return w.map((v) => v / s);
}

export default function LikelihoodExplorer() {
  // --- Parameters ---
  const [q, setQ] = useState<number[]>([7, 5, 5, 5]);
  const [x, setX] = useState<number[]>([3, 2, 2, 1]);
  const [w, setW] = useState<number[]>([0.5, 0.2, 0.2, 0.1]);
  const [beta, setBeta] = useState<number>(0.5);
  const [tau1, setTau1] = useState<number>(0.4);
  const [tau2, setTau2] = useState<number>(0.7);
  const [sadBand, setSadBand] = useState<number>(0.02);
  const [thetaStep, setThetaStep] = useState<number>(1);

  const wNorm = useMemo(() => normalize(w), [w]);
  const candX = useMemo(() => enumerateCandX(q.map((v) => Math.round(v))), [q]);

  const data = useMemo(() => {
    const rows: Array<{ [k: string]: number }> = [];
    for (let th = -90; th <= 90; th += thetaStep) {
      const S = satisfaction(deg2rad(th), wNorm, x, q, beta, candX);
      const emoScores = emotionScoresFromS(S, tau1, tau2, sadBand);
      const emoProbs = scoresToProbs(emoScores); // <-- normalize to probabilities
      rows.push({ theta: th, ...emoProbs });
    }
    return rows;
  }, [wNorm, x, q, beta, tau1, tau2, sadBand, thetaStep, candX]);

  const reset = () => {
    setQ([7, 5, 5, 5]);
    setX([3, 2, 2, 1]);
    setW([0.5, 0.2, 0.2, 0.1]);
    setBeta(0.5);
    setTau1(0.4);
    setTau2(0.7);
    setSadBand(0.02);
    setThetaStep(1);
  };

  return (
    <div className="p-6 grid gap-6 lg:grid-cols-2">
      <motion.h1 initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className="text-2xl font-semibold">
        Emotion Likelihood Explorer
      </motion.h1>

      <Card className="lg:row-span-2 shadow-md">
        <CardContent className="pt-6">
          <div className="h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
                <XAxis dataKey="theta" type="number" domain={[-90, 90]} tickCount={13}
                  label={{ value: "θ (deg)", position: "insideBottom", dy: 10 }} />
                <YAxis domain={[0, 1]} tickCount={6}
                  label={{ value: "probability", angle: -90, position: "insideLeft" }} />
                <Tooltip formatter={(v: number) => v.toFixed(3)} />
                <Legend />
                <ReferenceLine x={0} strokeDasharray="3 3" />
                <Line type="monotone" dataKey="Anger" stroke={EMO_COLORS.Anger} dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="Sad" stroke={EMO_COLORS.Sad} dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="Neutral" stroke={EMO_COLORS.Neutral} dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="Joy" stroke={EMO_COLORS.Joy} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card className="shadow-md">
        <CardContent className="pt-6 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium">Model params</h2>
            <Button variant="outline" onClick={reset}>Reset</Button>
          </div>
          <SliderField label="β (inverse temperature)" value={beta} min={0} max={4} step={0.01} onChange={([v]) => setBeta(v)} />
          <SliderField label="τ1 (Anger↔Neutral boundary)" value={tau1} min={0.05} max={0.9} step={0.005} onChange={([v]) => setTau1(v)} />
          <SliderField label="τ2 (Neutral↔Joy boundary)" value={tau2} min={0.1} max={0.98} step={0.005} onChange={([v]) => setTau2(v)} />
          <SliderField label="sad_band (Sad spike width)" value={sadBand} min={0.005} max={0.1} step={0.001} onChange={([v]) => setSadBand(v)} />
          <SliderField label="θ grid step (deg)" value={thetaStep} min={1} max={10} step={1} onChange={([v]) => setThetaStep(v)} />
        </CardContent>
      </Card>

      <Card className="shadow-md">
        <CardContent className="pt-6 space-y-5">
          <h2 className="text-lg font-medium">Scenario</h2>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-3">
              <Label className="text-sm text-muted-foreground">q (total quantities)</Label>
              <div className="grid grid-cols-4 gap-2">
                {q.map((qi, i) => (
                  <Input key={i} type="number" min={0} step={1} value={qi} onChange={(e) => {
                    const v = Math.max(0, Math.round(parseFloat(e.target.value)) || 0);
                    const next = [...q];
                    next[i] = v;
                    setQ(next);
                  }} />
                ))}
              </div>
            </div>
            <div className="space-y-3">
              <Label className="text-sm text-muted-foreground">x (proposal)</Label>
              <div className="grid grid-cols-4 gap-2">
                {x.map((xi, i) => (
                  <Input key={i} type="number" min={0} step={1} value={xi} onChange={(e) => {
                    const v = Math.max(0, Math.round(parseFloat(e.target.value)) || 0);
                    const next = [...x];
                    next[i] = v;
                    setX(next);
                  }} />
                ))}
              </div>
              <p className="text-xs text-muted-foreground">0 ≤ x_i ≤ q_i を満たすように手で調整してください。</p>
            </div>
          </div>

          <div className="space-y-2">
            <Label className="text-sm text-muted-foreground">w (issue weights, auto-normalized)</Label>
            <div className="grid grid-cols-4 gap-2">
              {w.map((wi, i) => (
                <Input key={i} type="number" step={0.01} value={wi} onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  const next = [...w];
                  next[i] = isNaN(v) ? 0 : v;
                  setW(next);
                }} />
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              現在の合計: {(w.reduce((a, b) => a + b, 0)).toFixed(3)} → 正規化後: [{wNorm.map((v) => v.toFixed(3)).join(", ")}]
            </p>
          </div>
        </CardContent>
      </Card>

      <Card className="lg:col-span-2 shadow-sm">
        <CardContent className="pt-5 text-sm text-muted-foreground space-y-2">
          <p>Tips:</p>
          <ul className="list-disc pl-5 space-y-1">
            <li>β を大きくすると S=exp(β(U−Umax)) が鋭くなり、Joy/Anger の確率が極端になりがちです。</li>
            <li>τ1, τ2 を動かして境界条件の影響を確認できます（Sad は τ1±sad_band の帯域）。</li>
            <li>q を大きくし過ぎると candX の組合せ数が増えて描画が重くなります。</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
