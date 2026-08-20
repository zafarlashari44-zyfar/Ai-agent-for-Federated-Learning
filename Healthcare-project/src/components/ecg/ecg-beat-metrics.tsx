"use client";

import {
  Activity,
  Gauge,
  HeartPulse,
  ScanLine,
} from "lucide-react";

import type { SegmentedBeat } from "@/types/ecg-analysis";

interface ECGBeatMetricsProps {
  beat: SegmentedBeat;
  previousBeat?: SegmentedBeat | null;
}

function formatMilliseconds(seconds: number) {
  return `${Math.round(seconds * 1000)} ms`;
}

export function ECGBeatMetrics({
  beat,
  previousBeat = null,
}: ECGBeatMetricsProps) {
  const rrIntervalSeconds = previousBeat
    ? beat.rPeakTimeSeconds - previousBeat.rPeakTimeSeconds
    : null;

  const estimatedHeartRate =
    rrIntervalSeconds && rrIntervalSeconds > 0
      ? Math.round(60 / rrIntervalSeconds)
      : null;

  const segmentDuration =
    beat.endTimeSeconds - beat.startTimeSeconds;

  const metrics = [
    {
      label: "RR interval",
      value:
        rrIntervalSeconds !== null
          ? formatMilliseconds(rrIntervalSeconds)
          : "Unavailable",
      Icon: HeartPulse,
      accent: "text-violet-600",
      surface: "bg-violet-50",
    },
    {
      label: "Estimated heart rate",
      value:
        estimatedHeartRate !== null
          ? `${estimatedHeartRate} bpm`
          : "Unavailable",
      Icon: Activity,
      accent: "text-blue-600",
      surface: "bg-blue-50",
    },
    {
      label: "Segment duration",
      value: formatMilliseconds(segmentDuration),
      Icon: ScanLine,
      accent: "text-emerald-600",
      surface: "bg-emerald-50",
    },
    {
      label: "Model confidence",
      value: `${Math.round(
        beat.prediction.confidence * 100,
      )}%`,
      Icon: Gauge,
      accent: "text-amber-600",
      surface: "bg-amber-50",
    },
  ];

  return (
    <section className="rounded-3xl border bg-white p-5 shadow-sm">
      <div>
        <h3 className="text-lg font-semibold text-slate-900">
          Beat measurements
        </h3>

        <p className="mt-1 text-sm text-slate-500">
          Automatically calculated metrics for the selected beat.
        </p>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3">
        {metrics.map((metric) => {
          const Icon = metric.Icon;

          return (
            <div
              key={metric.label}
              className="rounded-2xl border border-slate-200 p-4"
            >
              <div
                className={`inline-flex rounded-xl p-2 ${metric.surface}`}
              >
                <Icon
                  className={`h-4 w-4 ${metric.accent}`}
                />
              </div>

              <p className="mt-3 text-xs font-medium uppercase tracking-wide text-slate-400">
                {metric.label}
              </p>

              <p className="mt-1 text-lg font-semibold text-slate-900">
                {metric.value}
              </p>
            </div>
          );
        })}
      </div>

      <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-4">
        <div className="flex items-center justify-between gap-3">
          <span className="text-sm text-slate-500">
            Beat class
          </span>

          <span className="rounded-full bg-slate-950 px-3 py-1 text-xs font-semibold text-white">
            {beat.prediction.classCode}
          </span>
        </div>

        <p className="mt-2 text-sm font-medium text-slate-900">
          {beat.prediction.className}
        </p>
      </div>
    </section>
  );
}
