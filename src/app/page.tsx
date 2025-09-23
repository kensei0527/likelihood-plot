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
  Emotion Belt Scatter Explorer (self/other, probability-normalized)
  -----------------------------------------------------------------
  要件:
  - w は 1 刻みで調整（±w_max の範囲、符号付き）
  - 横軸 = Σ_i x_i * w_self,i（self の価値）
  - 縦軸 = Σ_i (q_i − x_i) * w_other,i（other の価値）
  - θ を固定して、全候補 x の組み合わせを点でプロット
  - 各点の色は other の表出感情（満足度 S に基づく g_k を正規化 → argmax）
  - 上部には従来の θ→感情尤度のラインチャートも残して比較可能に
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
  return w.map((wi) => Math.round(clamp(wi, -wmax, wmax))); // ← 1刻み想定
}

// 効用（other 視点; self/other 役割分離）
function utility(thetaRad: number, wSelf: number[], wOther: number[], x: number[], q: number[]) {
  const xOther = add(q, x, -1); // other の取り分 = q − x
  return Math.cos(thetaRad) * dot(wOther, xOther) + Math.sin(thetaRad) * dot(wSelf, x);
}

// 追加: 効用の下限クリップ
function clampUtility(u: number, uMin: number) {
  return Math.max(u, uMin);
}


// 満足度 S = exp(β (U − Umax))
function satisfactionGivenUmax(
  u: number,
  umax: number,
  beta: number,
) {
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
  Anger: "#f28e8e",   // pinkish red
  Sad: "#7d7aa6",     // muted purple
  Neutral: "#bfecc5", // pale green
  Joy: "#f5c04a",     // warm yellow
};

// ラベル順序
const EMO_ORDER = ["Joy", "Neutral", "Sad", "Anger"] as const;

function WeightSliderRow({ label, values, setValues, wMax }: { label: string; values: number[]; setValues: (v: number[]) => void; wMax: number; }) {
  return (
    <div className="space-y-2">
      <Label className="text-sm">{label} (step=1)</Label>
      <div className="grid grid-cols-4 gap-2">
        {values.map((wi, i) => (
          <Slider key={i} value={[wi]} min={-wMax} max={wMax} step={1} onValueChange={([v]) => {
            const next = [...values];
            next[i] = Math.round(v);
            setValues(next);
          }} />
        ))}
      </div>
      <p className="text-xs text-muted-foreground">current: [{values.map((v) => v.toFixed(0)).join(", ")}]</p>
    </div>
  );
}

export default function EmotionBeltScatterExplorer() {
  // --- Parameters ---
  const [q, setQ] = useState<number[]>([7, 5, 5, 5]);
  const [x, setX] = useState<number[]>([3, 2, 2, 1]);
  const [wSelf, setWSelf] = useState<number[]>([2, 1, 0, -1]);
  const [wOther, setWOther] = useState<number[]>([2, 0, -1, 1]);
  const [wMax, setWMax] = useState<number>(4);
  const [beta, setBeta] = useState<number>(0.8);
  const [tau1, setTau1] = useState<number>(0.4);
  const [tau2, setTau2] = useState<number>(0.7);
  const [sadBand, setSadBand] = useState<number>(0.02);
  const [thetaDeg, setThetaDeg] = useState<number>(45);
  const [thetaStep, setThetaStep] = useState<number>(1); // for line chart only
  // --- Parameters ---
  const [uMin, setUMin] = useState<number>(0); // ← 追加: 最低効用。デフォルトは 0 を推奨


  // クリップ（±w_max、1刻み）
  const wSelfClamped = useMemo(() => clampWeights(wSelf, wMax), [wSelf, wMax]);
  const wOtherClamped = useMemo(() => clampWeights(wOther, wMax), [wOther, wMax]);
  
  // Total points calculation
  const totalSelf = useMemo(() => {
    return x.reduce((sum, xi, i) => sum + xi * wSelfClamped[i], 0);
  }, [x, wSelfClamped]);
  
  const totalOther = useMemo(() => {
    return q.reduce((sum, qi, i) => sum + (qi - x[i]) * wOtherClamped[i], 0);
  }, [q, x, wOtherClamped]);

  const candX = useMemo(() => enumerateCandX(q.map((v) => Math.round(v))), [q]);

  // ---- ラインチャート（θ→感情尤度）: 参考用 ----
  const lineData = useMemo(() => {
    const rows: Array<{ [k: string]: number }> = [];
    for (let th = -90; th <= 90; th += thetaStep) {
      // Umax を一度だけ計算
      const thetaRad = deg2rad(th);
      let umax = -Infinity;
      // 1) まず Umax を計算（uMin で下限クリップしてから最大を取る）
      for (const xx of candX) {
        const uRaw = utility(thetaRad, wSelfClamped, wOtherClamped, xx, q);
        const uClamped = clampUtility(uRaw, uMin);
        if (uClamped > umax) umax = uClamped;
      }
      // 2) 提案 x の満足度も“クリップ後”の U で
      const uRaw = utility(thetaRad, wSelfClamped, wOtherClamped, x, q);
      const uClamped = clampUtility(uRaw, uMin);
      const S = satisfactionGivenUmax(uClamped, umax, beta);

      const probs = scoresToProbs(emotionScoresFromS(S, tau1, tau2, sadBand));
      rows.push({ theta: th, ...probs });
    }
    return rows;
  }, [wSelfClamped, wOtherClamped, x, q, beta, tau1, tau2, sadBand, thetaStep, candX, uMin]);

  // ---- スキャッター: 全候補 x の (selfValue, otherValue) と感情分類 ----
  const scatterGroups = useMemo(() => {
    const groups: Record<string, Array<{ sx: number; oy: number }>> = {
      Joy: [], Neutral: [], Sad: [], Anger: []
    };

    const thetaRad = deg2rad(thetaDeg);
    let umax = -Infinity;
    // Umax（クリップ後の値で最大）
    for (const xx of candX) {
      const uRaw = utility(thetaRad, wSelfClamped, wOtherClamped, xx, q);
      const uClamped = clampUtility(uRaw, uMin);
      if (uClamped > umax) umax = uClamped;
    }

    // 各候補を分類（クリップ後の U を使用）
    for (const xx of candX) {
      const uRaw = utility(thetaRad, wSelfClamped, wOtherClamped, xx, q);
      const uClamped = clampUtility(uRaw, uMin);
      const S = satisfactionGivenUmax(uClamped, umax, beta);
      const probs = scoresToProbs(emotionScoresFromS(S, tau1, tau2, sadBand));

      // argmax は現状のまま
      let best = "Joy" as keyof typeof probs;
      let bestVal = probs[best];
      (Object.keys(probs) as Array<keyof typeof probs>).forEach((k) => {
        if (probs[k] > bestVal) { best = k; bestVal = probs[k]; }
      });

      const selfVal = dot(wSelfClamped, xx);
      const otherVal = dot(wOtherClamped, add(q, xx, -1));
      groups[best].push({ sx: selfVal, oy: otherVal });
    }
    return groups;
  }, [thetaDeg, wSelfClamped, wOtherClamped, q, candX, beta, tau1, tau2, sadBand, uMin]);

  // 軸の範囲を自動で
  const xyExtent = useMemo(() => {
    const all = ([] as Array<{ sx: number; oy: number }>).concat(
      scatterGroups.Joy,
      scatterGroups.Neutral,
      scatterGroups.Sad,
      scatterGroups.Anger,
    );
    const xs = all.map((d) => d.sx);
    const ys = all.map((d) => d.oy);
    const xmin = Math.min(...xs, 0);
    const xmax = Math.max(...xs, 1);
    const ymin = Math.min(...ys, 0);
    const ymax = Math.max(...ys, 1);
    return { xmin, xmax, ymin, ymax };
  }, [scatterGroups]);

  const reset = () => {
    setQ([7, 5, 5, 5]);
    setX([3, 2, 2, 1]);
    setWSelf([2, 1, 0, -1]);
    setWOther([2, 0, -1, 1]);
    setWMax(4);
    setBeta(0.8);
    setTau1(0.4);
    setTau2(0.7);
    setSadBand(0.02);
    setThetaDeg(45);
    setThetaStep(1);
    setUMin(0);
  };

  return (
    <div className="p-6 grid gap-6 2xl:grid-cols-2">
      <motion.h1 initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className="text-2xl font-semibold">
        Emotion Belt Explorer (scatter region + θ-scan)
      </motion.h1>

      {/* ---- θ→感情尤度（参考） ---- */}
      <Card className="shadow-md">
        <CardContent className="pt-6">
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lineData} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
                <XAxis dataKey="theta" type="number" domain={[-90, 90]} tickCount={13}
                  label={{ value: "θ (deg)", position: "insideBottom", dy: 10 }} />
                <YAxis domain={[0, 1]} tickCount={6}
                  label={{ value: "P_other(E | θ, x, w_self, w_other)", angle: -90, position: "insideLeft" }} />
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

      {/* ---- スキャッター（候補 x の全点） ---- */}
      <Card className="shadow-md">
        <CardContent className="pt-6">
          <div className="h-[520px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="sx" domain={[xyExtent.xmin, xyExtent.xmax]}
                  label={{ value: "Self value Σ x_i w_self,i", position: "insideBottom", dy: 10 }} />
                <YAxis type="number" dataKey="oy" domain={[xyExtent.ymin, xyExtent.ymax]}
                  label={{ value: "Other value Σ (q_i − x_i) w_other,i", angle: -90, position: "insideLeft" }} />
                <Legend />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v: number, n: string) => v.toFixed(2)} />
                {EMO_ORDER.map((emo) => (
                  <Scatter key={emo} name={emo} data={scatterGroups[emo]} fill={EMO_COLORS[emo]} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* ---- 操作パネル ---- */}
      <Card className="2xl:col-span-2 shadow-md">
        <CardContent className="pt-6 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium">Controls</h2>
            <Button variant="outline" onClick={reset}>Reset</Button>
          </div>

          {/* --- 配分スライダー & ポイント（w×個数） --- */}
          <Card className="shadow-md">
            <CardContent className="pt-6 space-y-6">
              <h2 className="text-lg font-medium">Division (x is self share, q − x is other share)</h2>
              <div className="grid grid-cols-12 gap-2 text-sm font-medium">
                <div className="col-span-3">Your Item</div>
                <div className="col-span-3">Your Point = x_i × w_self,i</div>
                <div className="col-span-3">Division (x_i / q_i)</div>
                <div className="col-span-3">Opponent Point = (q_i - x_i) × w_other,i</div>
              </div>
              {[0,1,2,3].map((i) => (
                <div key={i} className="grid grid-cols-12 items-center gap-2 text-sm">
                  <div className="col-span-3">Item {i+1}</div>
                  <div className="col-span-3">{x[i]} × {wSelfClamped[i].toFixed(0)} = {(x[i] * wSelfClamped[i]).toFixed(0)} pt</div>
                  <div className="col-span-3">
                    <Slider 
                      value={[x[i]]} 
                      min={0} 
                      max={q[i]} 
                      step={1} 
                      onValueChange={([v]) => {
                        const next = [...x];
                        next[i] = Math.round(v);
                        setX(next);
                      }} 
                    />
                    <div className="text-xs text-center mt-1">{x[i]} / {q[i]}</div>
                  </div>
                  <div className="col-span-3">{q[i] - x[i]} × {wOtherClamped[i].toFixed(0)} = {((q[i] - x[i]) * wOtherClamped[i]).toFixed(0)} pt</div>
                </div>
              ))}
              <div className="grid grid-cols-12 gap-2 text-sm font-semibold pt-2 border-t">
                <div className="col-span-3">Total Point</div>
                <div className="col-span-3 text-blue-600">self: {totalSelf.toFixed(0)}</div>
                <div className="col-span-3" />
                <div className="col-span-3 text-blue-600">other: {totalOther.toFixed(0)}</div>
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <Label className="w-28 text-sm text-muted-foreground">θ (deg)</Label>
                <Slider value={[thetaDeg]} min={-90} max={90} step={1} onValueChange={([v]) => setThetaDeg(Math.round(v))} className="flex-1" />
                <span className="text-xs tabular-nums w-10 text-right">{thetaDeg}</span>
              </div>
              <div className="flex items-center gap-3">
                <Label className="w-28 text-sm text-muted-foreground">β</Label>
                <Slider value={[beta]} min={0} max={4} step={0.05} onValueChange={([v]) => setBeta(v)} className="flex-1" />
                <span className="text-xs tabular-nums w-10 text-right">{beta.toFixed(2)}</span>
              </div>
              <div className="flex items-center gap-3">
                <Label className="w-28 text-sm text-muted-foreground">τ1</Label>
                <Slider value={[tau1]} min={0.05} max={0.9} step={0.005} onValueChange={([v]) => setTau1(v)} className="flex-1" />
                <span className="text-xs tabular-nums w-10 text-right">{tau1.toFixed(3)}</span>
              </div>
              <div className="flex items-center gap-3">
                <Label className="w-28 text-sm text-muted-foreground">τ2</Label>
                <Slider value={[tau2]} min={0.1} max={0.98} step={0.005} onValueChange={([v]) => setTau2(v)} className="flex-1" />
                <span className="text-xs tabular-nums w-10 text-right">{tau2.toFixed(3)}</span>
              </div>
              <div className="flex items-center gap-3">
                <Label className="w-28 text-sm text-muted-foreground">sad_band</Label>
                <Slider value={[sadBand]} min={0.005} max={0.1} step={0.001} onValueChange={([v]) => setSadBand(v)} className="flex-1" />
                <span className="text-xs tabular-nums w-10 text-right">{sadBand.toFixed(3)}</span>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <Label className="w-28 text-sm text-muted-foreground">w_max</Label>
                <Input type="number" step={1} min={0} value={wMax} onChange={(e) => setWMax(Math.max(0, parseInt(e.target.value) || 0))} className="w-28" />
              </div>
              <div className="flex items-center gap-3">
                <Label className="w-28 text-sm text-muted-foreground">u_min</Label>
                <Input
                  type="number"
                  step={1}
                  value={uMin}
                  onChange={(e) => setUMin(Number(e.target.value))}
                  className="w-28"
                />
                <span className="text-xs text-muted-foreground">
                  Floor for utility U (clip: U = max(U, u_min))
                </span>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <WeightSliderRow label="w_self (proposer)" values={wSelf} setValues={setWSelf} wMax={wMax} />
              <WeightSliderRow label="w_other (emotion expresser)" values={wOther} setValues={setWOther} wMax={wMax} />
            </div>

            <div className="space-y-2 text-sm text-muted-foreground">
              <p>Notes:</p>
              <ul className="list-disc pl-5 space-y-1">
                <li>横軸 = 自分の価値 Σ x_i w_self,i、縦軸 = 相手の価値 Σ (q_i − x_i) w_other,i。</li>
                <li>各点は候補 x の組（整数格子）。色は other の表出感情（Joy/Neutral/Sad/Anger）。</li>
                <li>q を大きくすると点の数が指数的に増えるので注意（現在 4 次元）。</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}