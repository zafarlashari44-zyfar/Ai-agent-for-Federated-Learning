"use client";

import {
  AlertTriangle,
  CheckCircle2,
  ClipboardCopy,
  FileText,
  Printer,
  ShieldAlert,
  Stethoscope,
} from "lucide-react";

import type {
  ECGAnalysisResult,
  ECGSignalPoint,
} from "@/types/ecg-analysis";

interface ECGClinicalReportProps {
  analysis: ECGAnalysisResult;
}

function percentage(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function createWaveformPath(
  points: ECGSignalPoint[],
  width: number,
  height: number,
) {
  if (points.length === 0) {
    return "";
  }

  const amplitudes = points.map((point) => point.amplitude);
  const minimum = Math.min(...amplitudes);
  const maximum = Math.max(...amplitudes);
  const range = Math.max(maximum - minimum, 0.0001);

  return points
    .map((point, index) => {
      const x =
        points.length === 1
          ? 0
          : (index / (points.length - 1)) * width;

      const normalized =
        (point.amplitude - minimum) / range;

      const y = height - normalized * height;

      return `${index === 0 ? "M" : "L"} ${x.toFixed(
        2,
      )} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function getReportStrip(analysis: ECGAnalysisResult) {
  const abnormalBeat =
    analysis.beats.find((beat) => beat.isAbnormal) ??
    analysis.beats[0];

  if (!abnormalBeat) {
    return [];
  }

  const centre = abnormalBeat.rPeakTimeSeconds;
  const start = Math.max(0, centre - 4);
  const end = Math.min(
    analysis.metadata.durationSeconds,
    centre + 4,
  );

  return analysis.waveform.points.filter(
    (point) =>
      point.timeSeconds >= start &&
      point.timeSeconds <= end,
  );
}

export function ECGClinicalReport({
  analysis,
}: ECGClinicalReportProps) {
  const supportingEvidence =
    analysis.reasoning.evidence.filter(
      (item) => item.direction === "supports",
    );

  const contradictingEvidence =
    analysis.reasoning.evidence.filter(
      (item) => item.direction === "contradicts",
    );

  const probabilities = [
    ...(analysis.prediction.classProbabilities ?? []),
  ].sort(
    (first, second) =>
      second.probability - first.probability,
  );

  const stripPoints = getReportStrip(analysis);
  const waveformPath = createWaveformPath(
    stripPoints,
    760,
    150,
  );

  const analysisDate = formatDate(
    analysis.generatedAt,
  );

  const reportText = [
    "MediCare Pro",
    "AI Assisted ECG Interpretation Report",
    "",
    `Analysis ID: ${analysis.analysisId}`,
    `Record ID: ${analysis.metadata.recordId}`,
    `Generated: ${analysisDate}`,
    "",
    `Prediction: ${analysis.prediction.className}`,
    `Confidence: ${percentage(
      analysis.prediction.confidence,
    )}`,
    `Uncertainty: ${percentage(
      analysis.uncertainty.predictionUncertainty,
    )}`,
    "",
    "Clinical summary",
    analysis.reasoning.conclusion,
    "",
    "Supporting evidence",
    ...supportingEvidence.map(
      (item) =>
        `${item.featureName}: ${item.explanation}`,
    ),
    "",
    "Contradicting evidence",
    ...contradictingEvidence.map(
      (item) =>
        `${item.featureName}: ${item.explanation}`,
    ),
    "",
    "Limitations",
    ...analysis.uncertainty.limitations,
  ].join("\n");

  async function copyReport() {
    await navigator.clipboard.writeText(reportText);
  }

  function printReport() {
    window.print();
  }

  return (
    <section
      id="ecg-clinical-report"
      className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm print:rounded-none print:border-0 print:shadow-none"
    >
      <header className="flex flex-col gap-4 border-b bg-slate-950 px-6 py-5 text-white sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-blue-500/20 p-2.5">
            <FileText className="h-5 w-5 text-blue-300" />
          </div>

          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-300">
              MediCare Pro
            </p>

            <h3 className="mt-1 text-lg font-semibold">
              AI Assisted ECG Interpretation Report
            </h3>

            <p className="mt-1 text-xs text-slate-400">
              Structured report for clinician review
            </p>
          </div>
        </div>

        <div className="flex gap-2 print:hidden">
          <button
            type="button"
            onClick={copyReport}
            className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs font-medium text-slate-200 transition hover:bg-white/10"
          >
            <ClipboardCopy className="h-4 w-4" />
            Copy report
          </button>

          <button
            type="button"
            onClick={printReport}
            className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-xs font-medium text-white transition hover:bg-blue-500"
          >
            <Printer className="h-4 w-4" />
            Print or save PDF
          </button>
        </div>
      </header>

      <div className="space-y-6 p-6">
        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 print:grid-cols-4">
          {[
            [
              "Analysis ID",
              analysis.analysisId.slice(0, 12),
            ],
            [
              "Record ID",
              analysis.metadata.recordId,
            ],
            [
              "Patient ID",
              analysis.metadata.patientExternalId ??
                "Not provided",
            ],
            [
              "Generated",
              analysisDate,
            ],
          ].map(([label, value]) => (
            <div
              key={label}
              className="rounded-2xl border border-slate-200 bg-slate-50 p-4 print:rounded-lg"
            >
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                {label}
              </p>

              <p className="mt-2 break-words text-sm font-semibold text-slate-900">
                {value}
              </p>
            </div>
          ))}
        </section>

        <section>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Primary interpretation
          </p>

          <div className="mt-3 rounded-2xl border border-red-200 bg-red-50 p-5 print:rounded-lg">
            <div className="flex items-start gap-3">
              <Stethoscope className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />

              <div>
                <p className="text-xl font-semibold text-red-900">
                  {analysis.prediction.className}
                </p>

                <p className="mt-2 text-sm leading-6 text-red-800">
                  {analysis.reasoning.conclusion}
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-3 sm:grid-cols-3 print:grid-cols-3">
          <div className="rounded-2xl border p-4 print:rounded-lg">
            <p className="text-xs uppercase tracking-wide text-slate-400">
              Confidence
            </p>

            <p className="mt-2 text-2xl font-semibold text-emerald-600">
              {percentage(
                analysis.prediction.confidence,
              )}
            </p>
          </div>

          <div className="rounded-2xl border p-4 print:rounded-lg">
            <p className="text-xs uppercase tracking-wide text-slate-400">
              Uncertainty
            </p>

            <p className="mt-2 text-2xl font-semibold text-amber-600">
              {percentage(
                analysis.uncertainty
                  .predictionUncertainty,
              )}
            </p>
          </div>

          <div className="rounded-2xl border p-4 print:rounded-lg">
            <p className="text-xs uppercase tracking-wide text-slate-400">
              Signal quality
            </p>

            <p className="mt-2 text-2xl font-semibold capitalize text-blue-600">
              {analysis.signalQuality.status}
            </p>
          </div>
        </section>

        <section className="break-inside-avoid">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-slate-900">
                Representative ECG strip
              </p>

              <p className="mt-1 text-xs text-slate-500">
                {analysis.metadata.selectedLead} with an
                abnormal beat centred in the window
              </p>
            </div>

            <span className="rounded-full bg-red-50 px-3 py-1 text-xs font-semibold text-red-600">
              Abnormal region
            </span>
          </div>

          <div
            className="mt-3 overflow-hidden rounded-2xl border border-red-200 bg-[#fffafa] p-3 print:rounded-lg"
            style={{
              backgroundImage:
                "linear-gradient(rgba(244,63,94,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(244,63,94,0.08) 1px, transparent 1px), linear-gradient(rgba(244,63,94,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(244,63,94,0.15) 1px, transparent 1px)",
              backgroundSize:
                "10px 10px, 10px 10px, 50px 50px, 50px 50px",
            }}
          >
            <svg
              viewBox="0 0 760 150"
              className="h-40 w-full"
              role="img"
              aria-label="Representative ECG waveform"
            >
              <rect
                x="345"
                y="0"
                width="70"
                height="150"
                fill="rgba(239,68,68,0.12)"
              />

              <path
                d={waveformPath}
                fill="none"
                stroke="#0f766e"
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />

              <line
                x1="380"
                x2="380"
                y1="0"
                y2="150"
                stroke="#7c3aed"
                strokeWidth="2"
                strokeDasharray="5 4"
              />
            </svg>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-2 print:grid-cols-2">
          <div className="space-y-5">
            <div>
              <p className="text-sm font-semibold text-slate-900">
                Supporting evidence
              </p>

              <div className="mt-3 space-y-3">
                {supportingEvidence.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-start gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 p-4 print:rounded-lg"
                  >
                    <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />

                    <div>
                      <p className="font-medium text-emerald-950">
                        {item.featureName}
                      </p>

                      <p className="mt-1 text-sm leading-6 text-emerald-800">
                        {item.explanation}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <p className="text-sm font-semibold text-slate-900">
                Contradicting evidence
              </p>

              <div className="mt-3 space-y-3">
                {contradictingEvidence.length > 0 ? (
                  contradictingEvidence.map((item) => (
                    <div
                      key={item.id}
                      className="flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 print:rounded-lg"
                    >
                      <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />

                      <div>
                        <p className="font-medium text-red-950">
                          {item.featureName}
                        </p>

                        <p className="mt-1 text-sm leading-6 text-red-800">
                          {item.explanation}
                        </p>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="rounded-2xl border bg-slate-50 p-4 text-sm text-slate-500">
                    No strong contradicting evidence was
                    identified.
                  </p>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-5">
            <section className="break-inside-avoid">
              <p className="text-sm font-semibold text-slate-900">
                Class probabilities
              </p>

              <div className="mt-3 space-y-3 rounded-2xl border bg-white p-4 print:rounded-lg">
                {probabilities.map((item) => {
                  const value = Math.round(
                    item.probability * 100,
                  );

                  return (
                    <div key={item.classCode}>
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium text-slate-700">
                          {item.classCode} ·{" "}
                          {item.className}
                        </span>

                        <span className="font-semibold text-slate-900">
                          {value}%
                        </span>
                      </div>

                      <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full rounded-full bg-blue-600"
                          style={{
                            width: `${Math.max(
                              value,
                              1,
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            <section className="break-inside-avoid">
              <p className="text-sm font-semibold text-slate-900">
                Recording information
              </p>

              <dl className="mt-3 divide-y rounded-2xl border bg-white px-4 print:rounded-lg">
                {[
                  [
                    "Dataset",
                    analysis.metadata.datasetName,
                  ],
                  [
                    "Format",
                    analysis.metadata.sourceFormat,
                  ],
                  [
                    "Lead",
                    analysis.metadata.selectedLead,
                  ],
                  [
                    "Sampling rate",
                    `${analysis.metadata.samplingRateHz} Hz`,
                  ],
                  [
                    "Duration",
                    `${Math.round(
                      analysis.metadata.durationSeconds /
                        60,
                    )} minutes`,
                  ],
                  [
                    "Model",
                    analysis.metadata.modelVersion ??
                      "Unavailable",
                  ],
                  [
                    "Reasoning",
                    analysis.reasoning.reasoningVersion ??
                      "Unavailable",
                  ],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="flex justify-between gap-4 py-3 text-sm"
                  >
                    <dt className="text-slate-500">
                      {label}
                    </dt>

                    <dd className="text-right font-medium text-slate-900">
                      {value}
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
          </div>
        </section>

        <section className="grid gap-6 border-t pt-6 lg:grid-cols-[1fr_320px] print:grid-cols-[1fr_280px]">
          <div>
            <p className="text-sm font-semibold text-slate-900">
              Limitations
            </p>

            <div className="mt-3 space-y-3">
              {analysis.uncertainty.limitations.map(
                (limitation) => (
                  <div
                    key={limitation}
                    className="flex items-start gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4 print:rounded-lg"
                  >
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />

                    <p className="text-sm leading-6 text-amber-900">
                      {limitation}
                    </p>
                  </div>
                ),
              )}
            </div>
          </div>

          <aside className="space-y-4">
            <div className="rounded-2xl border border-blue-200 bg-blue-50 p-4 print:rounded-lg">
              <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">
                Clinical review required
              </p>

              <p className="mt-2 text-sm leading-6 text-blue-900">
                This analysis is decision support only and
                requires review by a qualified clinician.
              </p>
            </div>

            <div className="rounded-2xl border p-4 print:rounded-lg">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Generated by
              </p>

              <p className="mt-2 font-semibold text-slate-900">
                MediCare Pro Explainable AI Platform
              </p>

              <p className="mt-2 text-sm text-slate-500">
                {analysis.metadata.modelVersion ??
                  "Model unavailable"}
              </p>

              <p className="text-sm text-slate-500">
                {analysis.reasoning.reasoningVersion ??
                  "Reasoning version unavailable"}
              </p>

              <div className="mt-8 border-t pt-3">
                <p className="text-xs text-slate-400">
                  Clinician signature
                </p>
              </div>
            </div>
          </aside>
        </section>
      </div>
    </section>
  );
}
