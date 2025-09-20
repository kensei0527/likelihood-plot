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
  ----------------------------------------------------------------
  ◇ 横軸: θ (deg)
  ◇ 縦軸: 各感情 (Anger / Sad / Neutral / Joy) の確率 P(Emotion | θ, ...)

  この版の主な更新点:
  - 効用 U^{other}(x) = cosθ * <w_other, q-x> + sinθ * <w_self, x>
      → 役割分離（self/other）・複数論点の最新版の式に対応
  - w_self / w_other は符号付き。各成分は |w_i| ≤ w_max にクリップ（L1正規化は行わない）
  - 満足度は softmax 由来の S = exp(β (U - Umax)) を使用（0< S ≤ 1）
  - 感情スコア g_k(S) を最後に確率へ正規化: P_k = (g_k + ε) / Σ_j (g_j + ε)
  - UI を self/other で明示化
*/

const deg2rad = (d: number) => (Math.PI / 180) * d;

function enumerateCandX(q: number[]): number[][] {
  const xs: number[][] = [];
  // 注意: q が大きいと組合せが爆発します。検証用の小規模ケースを想定。
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

// --- 最新の効用: self/other 役割分離 ---
function utility(thetaRad: number, wSelf: number[], wOther: number[], x: number[], q: number[]) {
  const xOther = add(q, x, -1); // other の取り分 = q - x
  return Math.cos(thetaRad) * dot(wOther, xOther) + Math.sin(thetaRad) * dot(wSelf, x);
}

// 満足度（softmax 由来）: S = exp(β (U - Umax)) ∈ (0,1]
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

export default function LikelihoodExplorer() {
  // --- Parameters ---
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

  const data = useMemo(() => {
    const rows: Array<{ [k: string]: number }> = [];
    for (let th = -90; th <= 90; th += thetaStep) {
      const S = satisfaction(deg2rad(th), wSelfClamped, wOtherClamped, x, q, beta, candX);
      const emoScores = emotionScoresFromS(S, tau1, tau2, sadBand);
      const emoProbs = scoresToProbs(emoScores); // ← 正規化して確率
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

  return (
    <div className="p-6 grid gap-6 lg:grid-cols-2">
      <motion.h1 initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className="text-2xl font-semibold">
        Emotion Likelihood Explorer (self/other)
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
        <CardContent className="pt-6 space-y-6">
          <h2 className="text-lg font-medium">Scenario</h2>

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
            <Label className="text-sm text-muted-foreground">x (proposal: self share)</Label>
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

          <div className="space-y-4">
            <Label className="text-sm text-muted-foreground">w_self (issue weights of self; signed, |w_i| ≤ w_max)</Label>
            <div className="grid grid-cols-4 gap-2">
              {wSelf.map((wi, i) => (
                <Input key={i} type="number" step={0.01} value={wi} onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  const next = [...wSelf];
                  next[i] = isNaN(v) ? 0 : v;
                  setWSelf(next);
                }} />
              ))}
            </div>

            <Label className="text-sm text-muted-foreground">w_other (issue weights of other; signed, |w_i| ≤ w_max)</Label>
            <div className="grid grid-cols-4 gap-2">
              {wOther.map((wi, i) => (
                <Input key={i} type="number" step={0.01} value={wi} onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  const next = [...wOther];
                  next[i] = isNaN(v) ? 0 : v;
                  setWOther(next);
                }} />
              ))}
            </div>

            <div className="flex items-center gap-3">
              <Label className="w-28 text-sm text-muted-foreground">w_max</Label>
              <Input type="number" step={0.1} min={0} value={wMax} onChange={(e) => setWMax(Math.max(0, parseFloat(e.target.value) || 0))} className="w-28" />
              <span className="text-xs text-muted-foreground">入力値は描画時に [−w_max, w_max] にクリップされます。</span>
            </div>

            <p className="text-xs text-muted-foreground">
              現在の w_self (clamped): [{wSelfClamped.map((v) => v.toFixed(3)).join(", ")}] / w_other (clamped): [{wOtherClamped.map((v) => v.toFixed(3)).join(", ")}]
            </p>
          </div>
        </CardContent>
      </Card>

      <Card className="lg:col-span-2 shadow-sm">
        <CardContent className="pt-5 text-sm text-muted-foreground space-y-2">
          <p>Tips:</p>
          <ul className="list-disc pl-5 space-y-1">
            <li>β を大きくすると S = exp(β (U − Umax)) が鋭くなり、Joy/Anger の確率が極端になりがちです。</li>
            <li>τ1, τ2 を動かして境界条件の影響を確認できます（Sad は τ1±sad_band の帯域）。</li>
            <li>q を大きくし過ぎると candX の組合せ数が増えて描画が重くなります。</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
