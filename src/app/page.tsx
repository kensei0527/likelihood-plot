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
  Emotion Likelihood Explorer (self/other, probability-normalized)
  — weights-as-points variant —

  変更点（あなたの要望）
  - "各論点のポイント" は固定値ではなく **嗜好重み w × アイテム数** として表示・計算する
    → Your Point 列は q_i × w_self,i（self側の価値基準）
      Opponent Point は q_i × w_other,i（other側の価値基準）
  - Division スライダーで x_i を 0..q_i（1刻み）
  - w_self / w_other は ±w_max を **0.1刻み**スライダーで編集
  - グラフは other の感情尤度 P_other(E | θ, x, w_self, w_other)
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

// --- other の効用 ---
function utility(thetaRad: number, wSelf: number[], wOther: number[], x: number[], q: number[]) {
  const xOther = add(q, x, -1); // other の取り分 = q − x
  return Math.cos(thetaRad) * dot(wOther, xOther) + Math.sin(thetaRad) * dot(wSelf, x);
}

// S = exp(β (U − Umax))
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

export default function LikelihoodExplorer() {
  // --- Params ---
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

  const wSelfClamped = useMemo(() => clampWeights(wSelf, wMax), [wSelf, wMax]);
  const wOtherClamped = useMemo(() => clampWeights(wOther, wMax), [wOther, wMax]);
  const candX = useMemo(() => enumerateCandX(q.map((v) => Math.round(v))), [q]);

  // --- θ 走査: other の感情尤度 ---
  const data = useMemo(() => {
    const rows: Array<{ [k: string]: number }> = [];
    for (let th = -90; th <= 90; th += thetaStep) {
      const S = satisfaction(deg2rad(th), wSelfClamped, wOtherClamped, x, q, beta, candX);
      const emoScores = emotionScoresFromS(S, tau1, tau2, sadBand);
      const emoProbs = scoresToProbs(emoScores);
      rows.push({ theta: th, ...emoProbs });
    }
    return rows;
  }, [wSelfClamped, wOtherClamped, x, q, beta, tau1, tau2, sadBand, thetaStep, candX]);

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
  };

  // --- 合計ポイントは重み × 個数に基づく（要望どおり） ---
  const totalSelf = x.reduce((s, xi, i) => s + xi * wSelfClamped[i], 0);
  const totalOpp = x.reduce((s, xi, i) => s + (q[i] - xi) * wOtherClamped[i], 0);

  return (
    <div className="p-6 grid gap-6 xl:grid-cols-2">
      <motion.h1 initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className="text-2xl font-semibold">
        Emotion Likelihood Explorer (other's emotion likelihood)
      </motion.h1>

      {/* --- グラフ --- */}
      <Card className="xl:row-span-2 shadow-md">
        <CardContent className="pt-6">
          <div className="h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
                <XAxis dataKey="theta" type="number" domain={[-90, 90]} tickCount={13}
                  label={{ value: "θ (SVO angle, deg)", position: "insideBottom", dy: 10 }} />
                <YAxis domain={[0, 1]} tickCount={6}
                  label={{ value: "probability P_other(E | θ, x, w_self, w_other)", angle: -90, position: "insideLeft" }} />
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

      {/* --- モデル・ハイパラ --- */}
      <Card className="shadow-md">
        <CardContent className="pt-6 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium">Model params</h2>
            <Button variant="outline" onClick={reset}>Reset</Button>
          </div>
          <div className="grid gap-3">
            <div className="flex items-center gap-3">
              <Label className="w-40 text-sm text-muted-foreground">β (inverse temperature)</Label>
              <Slider value={[beta]} min={0} max={4} step={0.01} onValueChange={([v]) => setBeta(v)} className="flex-1" />
              <span className="text-xs tabular-nums w-12 text-right">{beta.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-3">
              <Label className="w-40 text-sm text-muted-foreground">τ1 (Anger↔Neutral)</Label>
              <Slider value={[tau1]} min={0.05} max={0.9} step={0.005} onValueChange={([v]) => setTau1(v)} className="flex-1" />
              <span className="text-xs tabular-nums w-12 text-right">{tau1.toFixed(3)}</span>
            </div>
            <div className="flex items-center gap-3">
              <Label className="w-40 text-sm text-muted-foreground">τ2 (Neutral↔Joy)</Label>
              <Slider value={[tau2]} min={0.1} max={0.98} step={0.005} onValueChange={([v]) => setTau2(v)} className="flex-1" />
              <span className="text-xs tabular-nums w-12 text-right">{tau2.toFixed(3)}</span>
            </div>
            <div className="flex items-center gap-3">
              <Label className="w-40 text-sm text-muted-foreground">sad_band</Label>
              <Slider value={[sadBand]} min={0.005} max={0.1} step={0.001} onValueChange={([v]) => setSadBand(v)} className="flex-1" />
              <span className="text-xs tabular-nums w-12 text-right">{sadBand.toFixed(3)}</span>
            </div>
            <div className="flex items-center gap-3">
              <Label className="w-40 text-sm text-muted-foreground">θ grid step (deg)</Label>
              <Slider value={[thetaStep]} min={1} max={10} step={1} onValueChange={([v]) => setThetaStep(v)} className="flex-1" />
              <span className="text-xs tabular-nums w-12 text-right">{thetaStep.toFixed(0)}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* --- 配分スライダー & ポイント（w×個数） --- */}
      <Card className="shadow-md">
        <CardContent className="pt-6 space-y-6">
          <h2 className="text-lg font-medium">Division (x is self share, q − x is other share)</h2>

          <div className="grid grid-cols-12 gap-2 text-sm font-medium">
            <div className="col-span-3">Your Item</div>
            <div className="col-span-3">Your Point = q_i × w_self,i</div>
            <div className="col-span-3">Division (x_i / q_i)</div>
            <div className="col-span-3">Opponent Point = q_i × w_other,i</div>
          </div>

          {[0,1,2,3].map((i) => (
            <div key={i} className="grid grid-cols-12 items-center gap-2 text-sm">
              <div className="col-span-3">Item {i+1}</div>
              <div className="col-span-3">{q[i]} × {wSelfClamped[i].toFixed(1)} pt</div>
              <div className="col-span-3">
                <Slider value={[x[i]]} min={0} max={q[i]} step={1} onValueChange={([v]) => {
                  const next = [...x];
                  next[i] = Math.round(v);
                  setX(next);
                }} />
              </div>
              <div className="col-span-3">{q[i]} × {wOtherClamped[i].toFixed(1)} pt</div>
            </div>
          ))}

          <div className="grid grid-cols-12 gap-2 text-sm font-semibold pt-2 border-t">
            <div className="col-span-3">Total Point</div>
            <div className="col-span-3 text-blue-600">self: {totalSelf.toFixed(1)}</div>
            <div className="col-span-3" />
            <div className="col-span-3 text-blue-600">other: {totalOpp.toFixed(1)}</div>
          </div>
        </CardContent>
      </Card>

      {/* --- 重みスライダー（0.1刻み） --- */}
      <Card className="xl:col-span-2 shadow-md">
        <CardContent className="pt-6 space-y-6">
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
                  <Slider key={i} value={[wi]} min={-wMax} max={wMax} step={0.1} onValueChange={([v]) => {
                    const next = [...wSelf];
                    next[i] = v;
                    setWSelf(next);
                  }} />
                ))}
              </div>
              <p className="text-xs text-muted-foreground">clamped: [{wSelfClamped.map((v) => v.toFixed(1)).join(", ")}]</p>
            </div>

            <div className="space-y-2">
              <Label className="text-sm">w_other (other / emotion expresser)</Label>
              <div className="grid grid-cols-4 gap-2">
                {wOther.map((wi, i) => (
                  <Slider key={i} value={[wi]} min={-wMax} max={wMax} step={0.1} onValueChange={([v]) => {
                    const next = [...wOther];
                    next[i] = v;
                    setWOther(next);
                  }} />
                ))}
              </div>
              <p className="text-xs text-muted-foreground">clamped: [{wOtherClamped.map((v) => v.toFixed(1)).join(", ")}]</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
