"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ChevronLeft,
  ChevronRight,
  Minus,
  Plus,
  RotateCcw,
} from "lucide-react";

import type {
  AttributionRegion,
  ECGSignalPoint,
  RPeak,
} from "@/types/ecg-analysis";

interface ECGWaveformViewerProps {
  points: ECGSignalPoint[];
  rPeaks?: RPeak[];
  attributionRegions?: AttributionRegion[];
  title?: string;
  leadName?: string;
  samplingRateHz?: number;
  visibleWindowSeconds?: number;
  focusTimeSeconds?: number | null;
}

const MINIMUM_WINDOW_SECONDS = 2;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}

function formatTime(seconds: number) {
  const safeSeconds = Math.max(seconds, 0);
  const minutes = Math.floor(safeSeconds / 60);
  const remaining = Math.floor(safeSeconds % 60);

  return `${minutes}:${remaining.toString().padStart(2, "0")}`;
}

function attributionStyle(intensity: number) {
  if (intensity >= 0.75) {
    return {
      fill: "rgba(239, 68, 68, 0.28)",
      stroke: "rgba(220, 38, 38, 0.9)",
    };
  }

  if (intensity >= 0.45) {
    return {
      fill: "rgba(249, 115, 22, 0.24)",
      stroke: "rgba(234, 88, 12, 0.85)",
    };
  }

  return {
    fill: "rgba(234, 179, 8, 0.2)",
    stroke: "rgba(202, 138, 4, 0.8)",
  };
}

export function ECGWaveformViewer({
  points,
  rPeaks = [],
  attributionRegions = [],
  title = "ECG recording",
  leadName = "Lead II",
  samplingRateHz,
  visibleWindowSeconds = 10,
  focusTimeSeconds = null,
}: ECGWaveformViewerProps) {
  const recordingStart = points[0]?.timeSeconds ?? 0;
  const recordingEnd =
    points[points.length - 1]?.timeSeconds ?? recordingStart;

  const recordingDuration = Math.max(
    recordingEnd - recordingStart,
    MINIMUM_WINDOW_SECONDS,
  );

  const initialWindow = clamp(
    visibleWindowSeconds,
    MINIMUM_WINDOW_SECONDS,
    recordingDuration,
  );

  const [windowSeconds, setWindowSeconds] =
    useState(initialWindow);

  const [windowStart, setWindowStart] =
    useState(recordingStart);

  const dragStartX = useRef<number | null>(null);
  const dragStartWindow = useRef(recordingStart);

  const effectiveWindow = Math.min(
    windowSeconds,
    recordingDuration,
  );

  const maximumWindowStart = Math.max(
    recordingStart,
    recordingEnd - effectiveWindow,
  );

  const safeWindowStart = clamp(
    windowStart,
    recordingStart,
    maximumWindowStart,
  );

  const windowEnd = Math.min(
    recordingEnd,
    safeWindowStart + effectiveWindow,
  );

  useEffect(() => {
    if (focusTimeSeconds === null) return;

    setWindowStart(
      clamp(
        focusTimeSeconds - effectiveWindow / 2,
        recordingStart,
        maximumWindowStart,
      ),
    );
  }, [
    focusTimeSeconds,
    effectiveWindow,
    recordingStart,
    maximumWindowStart,
  ]);

  const visiblePoints = useMemo(
    () =>
      points.filter(
        (point) =>
          point.timeSeconds >= safeWindowStart &&
          point.timeSeconds <= windowEnd,
      ),
    [points, safeWindowStart, windowEnd],
  );

  const visiblePeaks = useMemo(
    () =>
      rPeaks.filter(
        (peak) =>
          peak.timeSeconds >= safeWindowStart &&
          peak.timeSeconds <= windowEnd,
      ),
    [rPeaks, safeWindowStart, windowEnd],
  );

  const visibleRegions = useMemo(
    () =>
      attributionRegions.filter(
        (region) =>
          region.endTimeSeconds >= safeWindowStart &&
          region.startTimeSeconds <= windowEnd,
      ),
    [attributionRegions, safeWindowStart, windowEnd],
  );

  const overviewPoints = useMemo(() => {
    if (points.length <= 1200) return points;

    const step = Math.ceil(points.length / 1200);

    return points.filter((_, index) => index % step === 0);
  }, [points]);

  const amplitudes = visiblePoints.map(
    (point) => point.amplitude,
  );

  const minimumAmplitude =
    amplitudes.length > 0 ? Math.min(...amplitudes) : -1;

  const maximumAmplitude =
    amplitudes.length > 0 ? Math.max(...amplitudes) : 1;

  const amplitudePadding = Math.max(
    (maximumAmplitude - minimumAmplitude) * 0.18,
    0.2,
  );

  function changeZoom(nextWindow: number) {
    const clampedWindow = clamp(
      nextWindow,
      MINIMUM_WINDOW_SECONDS,
      recordingDuration,
    );

    const centre =
      safeWindowStart + effectiveWindow / 2;

    const nextMaximumStart = Math.max(
      recordingStart,
      recordingEnd - clampedWindow,
    );

    setWindowSeconds(clampedWindow);
    setWindowStart(
      clamp(
        centre - clampedWindow / 2,
        recordingStart,
        nextMaximumStart,
      ),
    );
  }

  function moveWindow(direction: number) {
    setWindowStart((current) =>
      clamp(
        current + effectiveWindow * 0.8 * direction,
        recordingStart,
        maximumWindowStart,
      ),
    );
  }

  function resetView() {
    setWindowSeconds(initialWindow);
    setWindowStart(recordingStart);
  }

  function handlePointerDown(
    event: ReactPointerEvent<HTMLDivElement>,
  ) {
    dragStartX.current = event.clientX;
    dragStartWindow.current = safeWindowStart;
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(
    event: ReactPointerEvent<HTMLDivElement>,
  ) {
    if (dragStartX.current === null) return;

    const width = event.currentTarget.clientWidth;
    if (width <= 0) return;

    const pixelDelta =
      event.clientX - dragStartX.current;

    const timeDelta =
      (pixelDelta / width) * effectiveWindow;

    setWindowStart(
      clamp(
        dragStartWindow.current - timeDelta,
        recordingStart,
        maximumWindowStart,
      ),
    );
  }

  function handlePointerUp(
    event: ReactPointerEvent<HTMLDivElement>,
  ) {
    dragStartX.current = null;

    if (
      event.currentTarget.hasPointerCapture(
        event.pointerId,
      )
    ) {
      event.currentTarget.releasePointerCapture(
        event.pointerId,
      );
    }
  }

  function handleOverviewClick(
    event: ReactPointerEvent<HTMLDivElement>,
  ) {
    const bounds =
      event.currentTarget.getBoundingClientRect();

    const ratio = clamp(
      (event.clientX - bounds.left) / bounds.width,
      0,
      1,
    );

    const selectedTime =
      recordingStart + ratio * recordingDuration;

    setWindowStart(
      clamp(
        selectedTime - effectiveWindow / 2,
        recordingStart,
        maximumWindowStart,
      ),
    );
  }

  if (points.length === 0) {
    return (
      <section className="rounded-2xl border bg-white p-6">
        <p className="text-sm text-slate-500">
          No ECG waveform is available.
        </p>
      </section>
    );
  }

  const viewportLeft =
    ((safeWindowStart - recordingStart) /
      recordingDuration) *
    100;

  const viewportWidth =
    (effectiveWindow / recordingDuration) * 100;

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-200 bg-slate-950 shadow-xl">
      <header className="flex flex-col gap-4 border-b border-white/10 px-5 py-4 text-white lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>

          <p className="mt-1 text-xs text-slate-400">
            {leadName}
            {samplingRateHz
              ? ` · ${samplingRateHz} Hz`
              : ""}
            {` · ${formatTime(recordingDuration)}`}
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => moveWindow(-1)}
            disabled={safeWindowStart <= recordingStart}
            className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-200 transition hover:bg-white/10 disabled:opacity-30"
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </button>

          <button
            type="button"
            onClick={() => moveWindow(1)}
            disabled={
              safeWindowStart >= maximumWindowStart
            }
            className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-200 transition hover:bg-white/10 disabled:opacity-30"
          >
            Next
            <ChevronRight className="h-4 w-4" />
          </button>

          <button
            type="button"
            onClick={() =>
              changeZoom(effectiveWindow * 1.5)
            }
            disabled={
              effectiveWindow >= recordingDuration
            }
            className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-200 transition hover:bg-white/10 disabled:opacity-30"
          >
            <Minus className="h-4 w-4" />
            Zoom out
          </button>

          <button
            type="button"
            onClick={() =>
              changeZoom(effectiveWindow / 1.5)
            }
            disabled={
              effectiveWindow <= MINIMUM_WINDOW_SECONDS
            }
            className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-200 transition hover:bg-white/10 disabled:opacity-30"
          >
            <Plus className="h-4 w-4" />
            Zoom in
          </button>

          <button
            type="button"
            onClick={resetView}
            className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-200 transition hover:bg-white/10"
          >
            <RotateCcw className="h-4 w-4" />
            Reset
          </button>
        </div>
      </header>

      <div className="p-4">
        <div
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          className="cursor-grab touch-none overflow-hidden rounded-2xl border border-red-200 bg-[#fffafa] active:cursor-grabbing"
          style={{
            backgroundImage:
              "linear-gradient(rgba(244,63,94,0.09) 1px, transparent 1px), linear-gradient(90deg, rgba(244,63,94,0.09) 1px, transparent 1px), linear-gradient(rgba(244,63,94,0.16) 1px, transparent 1px), linear-gradient(90deg, rgba(244,63,94,0.16) 1px, transparent 1px)",
            backgroundSize:
              "10px 10px, 10px 10px, 50px 50px, 50px 50px",
          }}
        >
          <div className="h-[430px] w-full">
            <ResponsiveContainer
              width="100%"
              height="100%"
            >
              <ComposedChart
                data={visiblePoints}
                margin={{
                  top: 20,
                  right: 26,
                  bottom: 18,
                  left: 8,
                }}
              >
                <CartesianGrid
                  stroke="rgba(148,163,184,0.12)"
                  vertical={false}
                />

                <XAxis
                  dataKey="timeSeconds"
                  type="number"
                  domain={[
                    safeWindowStart,
                    windowEnd,
                  ]}
                  allowDataOverflow
                  tickFormatter={formatTime}
                  tick={{
                    fontSize: 11,
                    fill: "#64748b",
                  }}
                  axisLine={{
                    stroke: "#cbd5e1",
                  }}
                />

                <YAxis
                  type="number"
                  domain={[
                    minimumAmplitude -
                      amplitudePadding,
                    maximumAmplitude +
                      amplitudePadding,
                  ]}
                  allowDataOverflow
                  width={48}
                  tick={{
                    fontSize: 11,
                    fill: "#64748b",
                  }}
                  axisLine={{
                    stroke: "#cbd5e1",
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
                    `Time ${formatTime(
                      Number(label),
                    )}`
                  }
                />

                {visibleRegions.map((region) => {
                  const style = attributionStyle(
                    region.intensity,
                  );

                  return (
                    <ReferenceArea
                      key={region.id}
                      x1={Math.max(
                        region.startTimeSeconds,
                        safeWindowStart,
                      )}
                      x2={Math.min(
                        region.endTimeSeconds,
                        windowEnd,
                      )}
                      fill={style.fill}
                      stroke={style.stroke}
                      strokeOpacity={0.9}
                    />
                  );
                })}

                <Line
                  type="linear"
                  dataKey="amplitude"
                  dot={false}
                  stroke="#0f766e"
                  strokeWidth={1.7}
                  isAnimationActive={false}
                />

                {visiblePeaks.map((peak) => (
                  <ReferenceDot
                    key={`${peak.index}-${peak.sampleIndex}`}
                    x={peak.timeSeconds}
                    y={
                      peak.amplitude ??
                      maximumAmplitude
                    }
                    r={3.5}
                    fill="#7c3aed"
                    stroke="#ffffff"
                    strokeWidth={1.5}
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="mt-4 flex items-center justify-between text-xs text-slate-400">
          <span>
            Window {formatTime(safeWindowStart)} to{" "}
            {formatTime(windowEnd)}
          </span>

          <span>
            {visiblePeaks.length} R peaks visible
          </span>
        </div>

        <div
          onPointerDown={handleOverviewClick}
          className="relative mt-3 h-24 cursor-crosshair overflow-hidden rounded-2xl border border-white/10 bg-slate-900"
        >
          <ResponsiveContainer
            width="100%"
            height="100%"
          >
            <ComposedChart
              data={overviewPoints}
              margin={{
                top: 10,
                right: 0,
                bottom: 10,
                left: 0,
              }}
            >
              <XAxis
                dataKey="timeSeconds"
                type="number"
                domain={[
                  recordingStart,
                  recordingEnd,
                ]}
                hide
              />

              <YAxis hide />

              <Line
                type="linear"
                dataKey="amplitude"
                dot={false}
                stroke="#38bdf8"
                strokeWidth={1}
                opacity={0.8}
                isAnimationActive={false}
              />

              {attributionRegions.map((region) => (
                <ReferenceArea
                  key={`overview-${region.id}`}
                  x1={region.startTimeSeconds}
                  x2={region.endTimeSeconds}
                  fill="rgba(239,68,68,0.55)"
                  stroke="rgba(248,113,113,0.9)"
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>

          <div
            className="pointer-events-none absolute inset-y-1 rounded-lg border-2 border-cyan-400 bg-cyan-400/10 shadow-[0_0_18px_rgba(34,211,238,0.35)]"
            style={{
              left: `${viewportLeft}%`,
              width: `${Math.max(
                viewportWidth,
                0.8,
              )}%`,
            }}
          />

          <div className="pointer-events-none absolute bottom-2 left-3 text-[10px] text-slate-400">
            00:00
          </div>

          <div className="pointer-events-none absolute bottom-2 right-3 text-[10px] text-slate-400">
            {formatTime(recordingDuration)}
          </div>
        </div>

        <div className="mt-4 flex flex-wrap gap-5 text-xs text-slate-400">
          <span className="inline-flex items-center gap-2">
            <span className="h-0.5 w-6 bg-teal-500" />
            ECG signal
          </span>

          <span className="inline-flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-violet-500" />
            R peak
          </span>

          <span className="inline-flex items-center gap-2">
            <span className="h-3 w-6 rounded-sm bg-red-500/60" />
            High attribution
          </span>

          <span className="inline-flex items-center gap-2">
            <span className="h-3 w-6 rounded-sm bg-orange-500/60" />
            Moderate attribution
          </span>
        </div>
      </div>
    </section>
  );
}
