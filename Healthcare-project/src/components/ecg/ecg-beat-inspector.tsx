"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { SegmentedBeat } from "@/types/ecg-analysis";

interface ECGBeatInspectorProps {
  beat: SegmentedBeat;
}

export function ECGBeatInspector({
  beat,
}: ECGBeatInspectorProps) {
  const firstTime =
    beat.waveform?.[0]?.timeSeconds ??
    beat.startTimeSeconds;

  const data =
    beat.waveform?.map((point) => ({
      timeMilliseconds:
        (point.timeSeconds - firstTime) * 1000,
      amplitude: point.amplitude,
    })) ?? [];

  const rPeakMilliseconds =
    (beat.rPeakTimeSeconds - firstTime) * 1000;

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b bg-slate-50 px-4 py-3">
        <div>
          <p className="text-sm font-semibold text-slate-900">
            Enlarged segmented beat
          </p>

          <p className="text-xs text-slate-500">
            Window centred on detected R peak
          </p>
        </div>

        <span className="rounded-full bg-violet-100 px-2.5 py-1 text-xs font-semibold text-violet-700">
          R peak
        </span>
      </div>

      <div
        className="h-56 w-full"
        style={{
          backgroundImage:
            "linear-gradient(rgba(244,63,94,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(244,63,94,0.07) 1px, transparent 1px), linear-gradient(rgba(244,63,94,0.13) 1px, transparent 1px), linear-gradient(90deg, rgba(244,63,94,0.13) 1px, transparent 1px)",
          backgroundSize:
            "8px 8px, 8px 8px, 40px 40px, 40px 40px",
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{
              top: 16,
              right: 18,
              bottom: 16,
              left: 0,
            }}
          >
            <CartesianGrid
              vertical={false}
              stroke="rgba(148,163,184,0.14)"
            />

            <XAxis
              dataKey="timeMilliseconds"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(value) =>
                `${Math.round(Number(value))} ms`
              }
              tick={{
                fontSize: 10,
                fill: "#64748b",
              }}
            />

            <YAxis
              width={42}
              tick={{
                fontSize: 10,
                fill: "#64748b",
              }}
            />

            <Tooltip
              formatter={(value) => [
                typeof value === "number"
                  ? `${value.toFixed(4)} mV`
                  : String(value),
                "Amplitude",
              ]}
              labelFormatter={(label) =>
                `${Math.round(Number(label))} ms`
              }
            />

            <ReferenceLine
              x={rPeakMilliseconds}
              stroke="#7c3aed"
              strokeWidth={2}
              strokeDasharray="4 3"
              label={{
                value: "R",
                position: "top",
                fill: "#7c3aed",
                fontSize: 11,
              }}
            />

            <Line
              type="linear"
              dataKey="amplitude"
              dot={false}
              stroke="#0f766e"
              strokeWidth={2}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
