"use client";

import React, { useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ScatterChart,
  Scatter,
  CartesianGrid,
} from "recharts";
import { motion } from "framer-motion";

/*
  Emotion Likelihood Explorer (self/other, probability-normalized)
  ----------------------------------------------------------------
  - 役割分離: U^other(x) = cosθ <w_other, q − x> + sinθ <w_self, x>
  - 感情スコア g_k(S) を正規化して確率: P_k = (g_k + ε) / Σ(g_j + ε)
  - θスキャンのラインチャートに加え、θ固定で全候補xを散布図に分類（色は表出感情）
  - Division テーブル（x_i スライダー）と q エディタはそのまま
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

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function clampWeights(w: number[], wmax: number) {
  return w.map((wi) => clamp(wi, -wmax, wmax));
}

// 効用（other 視点）
function utility(thetaRad: number, wSelf: number[], wOther: number[], x: number[], q: number[]) {
  const xOther = add(q, x, -1);
  return Math.cos(thetaRad) * dot(wOther, xOther) + Math.sin(thetaRad) * dot(wSelf, x);
}

// 満足度 S = exp(β (U − Umax))
function satisfaction(
  thetaRad: number,
  wSelf: number[],
  wOther: number[],
  x: number[],
  q: number[],
  beta: number,
  candX: number[][]
) {
  let umax = -Infinity;
  for (const xx of candX) umax = Math.max(umax, utility(thetaRad, wSelf, wOther, xx, q));
  const u = utility(thetaRad, wSelf, wOther, x, q);
  return Math.exp(beta * (u - umax));
}

// 区分線形スコア（未正規化）
function emotionScoresFromS(S: number, tau1: number, tau2: number, sadBand: number) {
  const clamp01 = (v: number) => clamp(v, 0, 1);
  const anger = clamp01((tau1 - S) / tau1);
  const sad = Math.abs(S - tau1) <= sadBand ? 1 - Math.abs(S - tau1) / sadBand : 0;
  let neutral = 0;
  if (S > tau1 && S < tau2) {
    const mid = (tau1 + tau2) / 2;
    const width = (tau2 - tau1) / 2;
    neutral = clamp01(1 - Math.abs(S - mid) / width);
  }
  const joy = clamp01((S - tau2) / (1 - tau2));
  return { Anger: anger, Sad: sad, Neutral: neutral, Joy: joy };
}

// スコア → 確率正規化
function scoresToProbs(scores: { [k: string]: number }, eps = 1e-9) {
  const total = (scores.Anger + eps) + (scores.Sad + eps) + (scores.Neutral + eps) + (scores.Joy + eps);
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

function SliderField({ label, value, onChange, min = 0, max = 1, step = 0.01 }: { label: string; value: number; onChange: (v: number[]) => void; min?: number; max?: number; step?: number; }) {
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

export default function LikelihoodExplorer() {
  const [q, setQ] = useState<number[]>([7, 5, 5, 5]);
  const [x, setX] = useState<number[]>([3, 2, 2, 1]);
  const [wSelf, setWSelf] = useState<number[]>([0.6, 0.2, 0.1, 0.1]);
  const [wOther, setWOther] = useState<number[]>([0.5, -0.2, 0.3, 0.1]);
  const [wMax, setWMax] = useState<number>(4);
  const [beta, setBeta] = useState<number>(0.8);
  const [tau1, setTau1] = useState<number>(0.4);
  const [tau2, setTau2] = useState<number>(0.7);
  const [sadBand, setSadBand] = useState<number>(0.02);
  const [thetaStep, setThetaStep] = useState<number>(1);
  const [thetaDeg, setThetaDeg] = useState<number>(45);

  const wSelfClamped = useMemo(() => clampWeights(wSelf, wMax), [wSelf, wMax]);
  const wOtherClamped = useMemo(() => clampWeights(wOther, wMax), [wOther, wMax]);
  const candX = useMemo(() => enumerateCandX(q.map((v) => Math.round(v))), [q]);

  // θ→感情尤度ライン
  const lineData = useMemo(() => {
    const rows: Array<{ [k: string]: number }> = [];
    for (let th = -90; th <= 90; th += thetaStep) {
      const S = satisfaction(deg2rad(th), wSelfClamped, wOtherClamped, x, q, beta, candX);
      const probs = scoresToProbs(emotionScoresFromS(S, tau1, tau2, sadBand));
      rows.push({ theta: th, ...probs });
    }
    return rows;
  }, [wSelfClamped, wOtherClamped, x, q, beta, tau1, tau2, sadBand, thetaStep, candX]);

  // 散布図（θ固定で全候補xを分類）
  const scatterGroups = useMemo(() => {
    const groups: Record<string, Array<{ sx: number; oy: number }>> = { Joy: [], Neutral: [], Sad: [], Anger: [] };
    const thetaRad = deg2rad(thetaDeg);
    let umax = -Infinity;
    for (const xx of candX) {
      const u = utility(thetaRad, wSelfClamped, wOtherClamped, xx, q);
      if (u > umax) umax = u;
    }
    for (const xx of candX) {
      const u = utility(thetaRad, wSelfClamped, wOtherClamped, xx, q);
      const S = Math.exp(beta * (u - umax));
      const probs = scoresToProbs(emotionScoresFromS(S, tau1, tau2, sadBand));
      let best: keyof typeof probs = "Joy";
      let bestVal = probs[best];
      (Object.keys(probs) as Array<keyof typeof probs>).forEach((k) => {
        if (probs[k] > bestVal) { best = k; bestVal = probs[k]; }
      });
      const sx = dot(wSelfClamped, xx);
      const oy = dot(wOtherClamped, add(q, xx, -1));
      groups[best].push({ sx, oy });
    }
    return groups;
  }, [thetaDeg, wSelfClamped, wOtherClamped, q, candX, beta, tau1, tau2, sadBand]);

  const reset = () => {
    setQ([7, 5, 5, 5]);
    setX([3, 2, 2, 1]);
    setWSelf([0.6, 0.2, 0.1, 0.1]);
    setWOther([0.5, -0.2, 0.3, 0.1]);
    setWMax(4);
    setBeta(0.8);
    setTau1(0.4);
    setTau2(0.7);
    setSadBand(0.02);
    setThetaStep(1);
    setThetaDeg(45);
  };

  // 合計ポイント: self = Σ x_i w_self,i, other = Σ (q_i − x_i) w_other,i
  const totalSelf = x.reduce((s, xi, i) => s + xi * wSelfClamped[i], 0);
  const totalOther = x.reduce((s, xi, i) => s + (q[i] - xi) * wOtherClamped[i], 0);

  return (
    <div className="p-6 grid gap-6 xl:grid-cols-2">
      <motion.h1 initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className="text-2xl font-semibold">
        Emotion Likelihood Explorer (self/other)
      </motion.h1>

      {/* ラインチャート（θ→感情尤度） */}
      <Card className="shadow-md">
        <CardContent className="pt-6">
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lineData} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
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

      {/* 散布図（θ固定の全候補x） */}
      <Card className="shadow-md">
        <CardContent className="pt-6">
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="sx" domain={["auto", "auto"]}
                  label={{ value: "Σ x_i w_self,i", position: "insideBottom", dy: 10 }} />
                <YAxis type="number" dataKey="oy" domain={["auto", "auto"]}
                  label={{ value: "Σ (q_i − x_i) w_other,i", angle: -90, position: "insideLeft" }} />
                <Legend />
                <Tooltip formatter={(v: number) => (typeof v === "number" ? v.toFixed(2) : String(v))} />
                <Scatter name="Joy" data={scatterGroups.Joy} fill={EMO_COLORS.Joy} />
                <Scatter name="Neutral" data={scatterGroups.Neutral} fill={EMO_COLORS.Neutral} />
                <Scatter name="Sad" data={scatterGroups.Sad} fill={EMO_COLORS.Sad} />
                <Scatter name="Anger" data={scatterGroups.Anger} fill={EMO_COLORS.Anger} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 text-sm">θ = {thetaDeg}° 固定での全候補 x の感情分類</div>
        </CardContent>
      </Card>

      {/* θ 調整 */}
      <Card className="shadow-md">
        <CardContent className="pt-6 space-y-4">
          <div className="flex items-center gap-3">
            <Label className="w-28 text-sm">θ (deg)</Label>
            <Slider value={[thetaDeg]} min={-90} max={90} step={1} onValueChange={([v]) => setThetaDeg(v)} className="flex-1" />
            <span className="text-sm tabular-nums w-10 text-right">{thetaDeg}</span>
          </div>
        </CardContent>
      </Card>

      {/* q 編集 + Division（スライダー） */}
      <Card className="shadow-md xl:col-span-2">
        <CardContent className="pt-6 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium">Division (x is self share, q − x is other share)</h2>
            <Button variant="outline" onClick={reset}>Reset</Button>
          </div>

          {/* q editor */}
          <div className="space-y-3">
            <Label className="text-sm text-muted-foreground">q (total quantities) — change to adjust slider ranges</Label>
            <div className="grid grid-cols-4 gap-2">
              {q.map((qi, i) => (
                <Input key={i} type="number" min={0} step={1} value={qi} onChange={(e) => {
                  const v = Math.max(0, Math.round(parseFloat(e.target.value)) || 0);
                  const next = [...q];
                  // x_i が q_i を超えないように同期
                  if (x[i] > v) {
                    const nx = [...x];
                    nx[i] = v;
                    setX(nx);
                  }
                  next[i] = v;
                  setQ(next);
                }} />
              ))}
            </div>
          </div>

          {/* Division table */}
          <div className="grid grid-cols-12 gap-2 text-sm font-medium">
            <div className="col-span-3">Your Item</div>
            <div className="col-span-3">Your Point = q_i × w_self,i</div>
            <div className="col-span-4">Division (x_i / q_i)</div>
            <div className="col-span-2">Opponent Point = q_i × w_other,i</div>
          </div>

          {[0,1,2,3].map((i) => (
            <div key={i} className="grid grid-cols-12 items-center gap-2 text-sm">
              <div className="col-span-3">Item {i+1}</div>
              <div className="col-span-3">{q[i]} × {wSelfClamped[i].toFixed(1)} pt</div>
              <div className="col-span-4">
                <Slider value={[x[i]]} min={0} max={q[i]} step={1} onValueChange={([v]) => {
                  const next = [...x];
                  next[i] = Math.round(v);
                  setX(next);
                }} />
              </div>
              <div className="col-span-2">{q[i]} × {wOtherClamped[i].toFixed(1)} pt</div>
            </div>
          ))}

          <div className="pt-4 border-t" />
          <div className="flex items-center justify-between text-sm font-semibold">
            <span>Total Point</span>
            <div className="flex items-center gap-8">
              <span className="text-blue-600">self: {totalSelf.toFixed(1)}</span>
              <span className="text-blue-600">other: {totalOther.toFixed(1)}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 重み編集 */}
      <Card className="shadow-md xl:col-span-2">
        <CardContent className="pt-6 space-y-4">
          <h2 className="text-lg font-medium">Issue weights (signed; |w_i| ≤ w_max)</h2>
          <div className="flex items-center gap-3">
            <Label className="w-28 text-sm text-muted-foreground">w_max</Label>
            <Input type="number" step={0.1} min={0} value={wMax} onChange={(e) => setWMax(Math.max(0, parseFloat(e.target.value) || 0))} className="w-28" />
          </div>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label className="text-sm">w_self (self / proposer)</Label>
              <div className="grid grid-cols-4 gap-2">
                {wSelf.map((wi, i) => (
                  <Input key={i} type="number" step={0.1} value={wi} onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    const next = [...wSelf];
                    next[i] = isNaN(v) ? 0 : v;
                    setWSelf(next);
                  }} />
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <Label className="text-sm">w_other (other / emotion expresser)</Label>
              <div className="grid grid-cols-4 gap-2">
                {wOther.map((wi, i) => (
                  <Input key={i} type="number" step={0.1} value={wi} onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    const next = [...wOther];
                    next[i] = isNaN(v) ? 0 : v;
                    setWOther(next);
                  }} />
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
