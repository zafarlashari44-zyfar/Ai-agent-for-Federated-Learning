"use client";

import {
  AlertTriangle,
  CircleDot,
  ShieldAlert,
} from "lucide-react";

import type {
  AttributionRegion,
  SegmentedBeat,
} from "@/types/ecg-analysis";

interface ECGEventTimelineProps {
  durationSeconds: number;
  beats: SegmentedBeat[];
  regions: AttributionRegion[];
  selectedBeatIndex: number | null;
  onSelectBeat: (beatIndex: number) => void;
}

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);

  return `${minutes}:${remainingSeconds
    .toString()
    .padStart(2, "0")}`;
}

function getEventColor(classCode: string) {
  if (classCode === "V") return "bg-red-500";
  if (classCode === "S") return "bg-orange-500";
  if (classCode === "F") return "bg-violet-500";
  if (classCode === "Q") return "bg-slate-500";

  return "bg-emerald-500";
}

export function ECGEventTimeline({
  durationSeconds,
  beats,
  regions,
  selectedBeatIndex,
  onSelectBeat,
}: ECGEventTimelineProps) {
  const abnormalBeats = beats.filter(
    (beat) => beat.isAbnormal,
  );

  return (
    <section className="rounded-3xl border bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-lg font-semibold text-slate-900">
            Recording event timeline
          </h3>

          <p className="mt-1 text-sm text-slate-500">
            Abnormal beats and attribution hotspots across the
            complete recording.
          </p>
        </div>

        <div className="flex flex-wrap gap-2 text-xs">
          <span className="inline-flex items-center gap-2 rounded-full bg-red-50 px-3 py-1 text-red-700">
            <CircleDot className="h-3.5 w-3.5" />
            {abnormalBeats.length} abnormal beats
          </span>

          <span className="inline-flex items-center gap-2 rounded-full bg-amber-50 px-3 py-1 text-amber-700">
            <AlertTriangle className="h-3.5 w-3.5" />
            {regions.length} attribution regions
          </span>
        </div>
      </div>

      <div className="mt-6">
        <div className="relative h-28 overflow-hidden rounded-2xl border border-slate-200 bg-slate-950">
          <div className="absolute inset-x-4 top-1/2 h-px -translate-y-1/2 bg-slate-700" />

          {regions.map((region) => {
            const left =
              (region.startTimeSeconds / durationSeconds) * 100;

            const width =
              ((region.endTimeSeconds -
                region.startTimeSeconds) /
                durationSeconds) *
              100;

            return (
              <div
                key={region.id}
                className="absolute top-4 h-20 rounded-lg border border-amber-300/60 bg-amber-400/20"
                style={{
                  left: `${left}%`,
                  width: `${Math.max(width, 0.8)}%`,
                }}
                title={`${region.label ?? "Attribution region"} · ${Math.round(
                  region.intensity * 100,
                )}%`}
              />
            );
          })}

          {abnormalBeats.map((beat) => {
            const left =
              (beat.rPeakTimeSeconds / durationSeconds) * 100;

            const selected =
              beat.beatIndex === selectedBeatIndex;

            return (
              <button
                key={beat.beatIndex}
                type="button"
                onClick={() => onSelectBeat(beat.beatIndex)}
                className="group absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
                style={{
                  left: `${left}%`,
                }}
                title={`Beat ${beat.beatIndex + 1} · ${formatTime(
                  beat.rPeakTimeSeconds,
                )}`}
              >
                <span
                  className={`block h-4 w-4 rounded-full border-2 border-white shadow-lg transition ${
                    selected
                      ? "scale-125 ring-4 ring-blue-400/40"
                      : "group-hover:scale-110"
                  } ${getEventColor(
                    beat.prediction.classCode,
                  )}`}
                />
              </button>
            );
          })}

          <div className="absolute bottom-2 left-3 text-[10px] text-slate-400">
            00:00
          </div>

          <div className="absolute bottom-2 left-1/4 text-[10px] text-slate-500">
            {formatTime(durationSeconds * 0.25)}
          </div>

          <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] text-slate-500">
            {formatTime(durationSeconds * 0.5)}
          </div>

          <div className="absolute bottom-2 left-3/4 text-[10px] text-slate-500">
            {formatTime(durationSeconds * 0.75)}
          </div>

          <div className="absolute bottom-2 right-3 text-[10px] text-slate-400">
            {formatTime(durationSeconds)}
          </div>
        </div>

        <div className="mt-4 flex flex-wrap gap-4 text-xs text-slate-500">
          <span className="inline-flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
            Ventricular event
          </span>

          <span className="inline-flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-orange-500" />
            Supraventricular event
          </span>

          <span className="inline-flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-violet-500" />
            Fusion event
          </span>

          <span className="inline-flex items-center gap-2">
            <span className="h-3 w-6 rounded-sm border border-amber-400 bg-amber-300/40" />
            Attribution hotspot
          </span>

          <span className="inline-flex items-center gap-2">
            <ShieldAlert className="h-3.5 w-3.5 text-slate-500" />
            Click a marker to inspect the beat
          </span>
        </div>
      </div>
    </section>
  );
}
