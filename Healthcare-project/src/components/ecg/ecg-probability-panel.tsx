"use client";

import type { ClassProbability } from "@/types/ecg-analysis";

interface ECGProbabilityPanelProps {
  probabilities?: ClassProbability[];
  predictedClassCode: string;
}

function getBarClass(classCode: string, isPredicted: boolean) {
  if (isPredicted) return "bg-blue-600";

  if (classCode === "V") return "bg-red-400";
  if (classCode === "S") return "bg-orange-400";
  if (classCode === "F") return "bg-violet-400";
  if (classCode === "Q") return "bg-slate-400";

  return "bg-emerald-400";
}

export function ECGProbabilityPanel({
  probabilities = [],
  predictedClassCode,
}: ECGProbabilityPanelProps) {
  const sortedProbabilities = [...probabilities].sort(
    (first, second) => second.probability - first.probability,
  );

  return (
    <section className="rounded-3xl border bg-white p-5 shadow-sm">
      <div>
        <h3 className="text-lg font-semibold text-slate-900">
          Class probabilities
        </h3>

        <p className="mt-1 text-sm text-slate-500">
          Recording level model output across the AAMI classes.
        </p>
      </div>

      <div className="mt-5 space-y-4">
        {sortedProbabilities.map((item) => {
          const percentage = Math.round(item.probability * 100);
          const isPredicted =
            item.classCode === predictedClassCode;

          return (
            <div key={item.classCode}>
              <div className="flex items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <span
                    className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-bold ${
                      isPredicted
                        ? "bg-blue-600 text-white"
                        : "bg-slate-100 text-slate-600"
                    }`}
                  >
                    {item.classCode}
                  </span>

                  <span className="truncate text-sm font-medium text-slate-700">
                    {item.className}
                  </span>
                </div>

                <span
                  className={`text-sm font-semibold ${
                    isPredicted
                      ? "text-blue-700"
                      : "text-slate-600"
                  }`}
                >
                  {percentage}%
                </span>
              </div>

              <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${getBarClass(
                    item.classCode,
                    isPredicted,
                  )}`}
                  style={{
                    width: `${Math.max(percentage, 1)}%`,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-5 rounded-2xl border border-blue-100 bg-blue-50 p-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">
          Selected output
        </p>

        <p className="mt-1 text-sm text-blue-900">
          Class {predictedClassCode} has the strongest recording level
          probability.
        </p>
      </div>
    </section>
  );
}
