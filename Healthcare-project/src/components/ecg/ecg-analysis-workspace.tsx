"use client";
import { ChangeEvent, useMemo, useState } from "react";
import { useECGAnalysis } from "@/hooks/use-ecg-analysis";
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  FileHeart,
  HeartPulse,
  Loader2,
  ScanLine,
  ShieldAlert,
  Upload,
  Waves,
} from "lucide-react";

import { ECGBeatInspector } from "@/components/ecg/ecg-beat-inspector";
import { ECGBeatMetrics } from "@/components/ecg/ecg-beat-metrics";
import { ECGEventTimeline } from "@/components/ecg/ecg-event-timeline";
import { ECGLeadSelector } from "@/components/ecg/ecg-lead-selector";
import { ECGClinicalReport } from "@/components/ecg/ecg-clinical-report";
import { ECGWaveformViewer } from "@/components/ecg/ecg-waveform-viewer";
import type {
  AttributionRegion,
  ClinicalEvidenceItem,
  ECGAnalysisResult,
  ECGSignalPoint,
  RPeak,
  SegmentedBeat,
} from "@/types/ecg-analysis";

const SAMPLE_RATE_HZ = 25;
const RECORDING_DURATION_SECONDS = 30 * 60;

function generateECGAmplitude(timeSeconds: number) {
  const beatInterval = 0.82;
  const phase = timeSeconds % beatInterval;

  const baseline =
    0.025 * Math.sin(timeSeconds * Math.PI * 0.4) +
    0.01 * Math.sin(timeSeconds * Math.PI * 5);

  const pWave =
    0.12 * Math.exp(-Math.pow((phase - 0.14) / 0.035, 2));

  const qWave =
    -0.16 * Math.exp(-Math.pow((phase - 0.28) / 0.014, 2));

  const rWave =
    1.05 * Math.exp(-Math.pow((phase - 0.3) / 0.012, 2));

  const sWave =
    -0.28 * Math.exp(-Math.pow((phase - 0.33) / 0.016, 2));

  const tWave =
    0.3 * Math.exp(-Math.pow((phase - 0.52) / 0.07, 2));

  return baseline + pWave + qWave + rWave + sWave + tWave;
}

function isAbnormalTime(timeSeconds: number) {
  return (
    (timeSeconds >= 312 && timeSeconds <= 325) ||
    (timeSeconds >= 945 && timeSeconds <= 960) ||
    (timeSeconds >= 1470 && timeSeconds <= 1485)
  );
}

function formatRecordingTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);

  return `${minutes}:${remainingSeconds
    .toString()
    .padStart(2, "0")}`;
}

function createDemoAnalysis(fileName: string): ECGAnalysisResult {
  const totalSamples =
    SAMPLE_RATE_HZ * RECORDING_DURATION_SECONDS;

  const waveform: ECGSignalPoint[] = Array.from(
    { length: totalSamples },
    (_, sampleIndex) => {
      const timeSeconds = sampleIndex / SAMPLE_RATE_HZ;
      let amplitude = generateECGAmplitude(timeSeconds);

      if (isAbnormalTime(timeSeconds)) {
        amplitude =
          amplitude * 1.3 +
          0.12 * Math.sin(timeSeconds * Math.PI * 10);
      }

      return {
        sampleIndex,
        timeSeconds,
        amplitude,
      };
    },
  );

  const rPeaks: RPeak[] = [];
  const beats: SegmentedBeat[] = [];

  let beatIndex = 0;

  for (
    let peakTime = 0.3;
    peakTime < RECORDING_DURATION_SECONDS;
    peakTime += 0.82
  ) {
    const sampleIndex = Math.round(
      peakTime * SAMPLE_RATE_HZ,
    );

    const point = waveform[sampleIndex];
    const abnormal = isAbnormalTime(peakTime);

    const startTimeSeconds = Math.max(0, peakTime - 0.25);
    const endTimeSeconds = Math.min(
      RECORDING_DURATION_SECONDS,
      peakTime + 0.45,
    );

    rPeaks.push({
      index: beatIndex,
      sampleIndex,
      timeSeconds: peakTime,
      amplitude: point?.amplitude ?? 1,
      confidence: abnormal ? 0.9 : 0.97,
    });

    beats.push({
      beatIndex,
      rPeakSample: sampleIndex,
      rPeakTimeSeconds: peakTime,
      startSample: Math.round(
        startTimeSeconds * SAMPLE_RATE_HZ,
      ),
      endSample: Math.round(
        endTimeSeconds * SAMPLE_RATE_HZ,
      ),
      startTimeSeconds,
      endTimeSeconds,
      waveform: waveform.slice(
        Math.round(startTimeSeconds * SAMPLE_RATE_HZ),
        Math.round(endTimeSeconds * SAMPLE_RATE_HZ) + 1,
      ),
      prediction: {
        classCode: abnormal ? "V" : "N",
        className: abnormal
          ? "Ventricular ectopic beat"
          : "Normal beat",
        confidence: abnormal ? 0.89 : 0.96,
      },
      attributionRegions: [],
      isAbnormal: abnormal,
    });

    beatIndex += 1;
  }

  const attributionRegions: AttributionRegion[] = [
    {
      id: "region-1",
      method: "grad-cam",
      startSample: 312 * SAMPLE_RATE_HZ,
      endSample: 325 * SAMPLE_RATE_HZ,
      startTimeSeconds: 312,
      endTimeSeconds: 325,
      intensity: 0.91,
      label: "High ventricular influence",
      description:
        "Strong model attribution around abnormal QRS morphology.",
    },
    {
      id: "region-2",
      method: "grad-cam",
      startSample: 945 * SAMPLE_RATE_HZ,
      endSample: 960 * SAMPLE_RATE_HZ,
      startTimeSeconds: 945,
      endTimeSeconds: 960,
      intensity: 0.72,
      label: "Rhythm irregularity",
      description:
        "Moderate attribution around irregular beat timing.",
    },
  ];

  const evidence: ClinicalEvidenceItem[] = [
    {
      id: "evidence-1",
      featureName: "QRS morphology",
      observedValue: "Widened ventricular complexes",
      expectedRange: "Narrow and stable QRS morphology",
      direction: "supports",
      explanation:
        "Multiple beats contain ventricular morphology supporting the predicted class.",
      confidence: 0.91,
      startTimeSeconds: 312,
      endTimeSeconds: 325,
    },
    {
      id: "evidence-2",
      featureName: "RR variability",
      observedValue: 0.68,
      expectedRange: "Below 0.30",
      direction: "supports",
      explanation:
        "Elevated beat-to-beat variability supports rhythm abnormality.",
      confidence: 0.84,
      startTimeSeconds: 945,
      endTimeSeconds: 960,
    },
    {
      id: "evidence-3",
      featureName: "Average heart rate",
      observedValue: "73 bpm",
      expectedRange: "60 to 100 bpm",
      direction: "contradicts",
      explanation:
        "The average heart rate remains within a typical resting range.",
      confidence: 0.79,
    },
  ];

  const abnormalBeatCount = beats.filter(
    (beat) => beat.isAbnormal,
  ).length;

  return {
    analysisId: crypto.randomUUID(),
    generatedAt: new Date().toISOString(),

    metadata: {
      datasetName: "Uploaded ECG",
      recordId: fileName.replace(/\.[^/.]+$/, ""),
      sourceFormat:
        fileName.split(".").pop()?.toUpperCase() ?? "UNKNOWN",
      sourceFileNames: [fileName],
      patientExternalId: "DEMO-PATIENT-001",
      leadNames: ["Lead II"],
      selectedLead: "Lead II",
      samplingRateHz: SAMPLE_RATE_HZ,
      durationSeconds: RECORDING_DURATION_SECONDS,
      totalSamples,
      preprocessingVersion: "dashboard-demo-v2",
      modelVersion: "federated-cnn-v1",
    },

    signalQuality: {
      status: "good",
      score: 0.93,
      explanation:
        "The signal has acceptable baseline stability and low noise.",
      warnings: [],
    },

    waveform: {
      startSample: 0,
      endSample: totalSamples - 1,
      startTimeSeconds: 0,
      endTimeSeconds: RECORDING_DURATION_SECONDS,
      points: waveform,
    },

    rPeaks,
    beats,
    attributionMethod: "grad-cam",
    attributionRegions,

    prediction: {
      classCode: "V",
      className: "Ventricular arrhythmia",
      confidence: 0.89,
      dominantClass: "N",
      abnormalBeatCount,
      abnormalBeatPercentage:
        (abnormalBeatCount / beats.length) * 100,
      totalBeatCount: beats.length,
      classCounts: {
        N: beats.length - abnormalBeatCount,
        S: 0,
        V: abnormalBeatCount,
        F: 0,
        Q: 0,
      },
      classProbabilities: [
        {
          classCode: "N",
          className: "Normal",
          probability: 0.08,
        },
        {
          classCode: "S",
          className: "Supraventricular",
          probability: 0.01,
        },
        {
          classCode: "V",
          className: "Ventricular",
          probability: 0.89,
        },
        {
          classCode: "F",
          className: "Fusion",
          probability: 0.01,
        },
        {
          classCode: "Q",
          className: "Unknown",
          probability: 0.01,
        },
      ],
    },

    reasoning: {
      conclusion:
        "The recording contains ventricular ectopic activity with several influential abnormal regions.",
      evidence,
      consistencyStatus: "strongly_supported",
      reasoningConfidence: 0.87,
      limitations: [
        "This frontend stage currently uses generated ECG data.",
        "The attribution values are demonstration values.",
        "Clinical review remains required.",
      ],
      reasoningVersion: "reasoning-demo-v2",
    },

    uncertainty: {
      predictionUncertainty: 0.11,
      reasoningConfidence: 0.87,
      consistencyStatus: "strongly_supported",
      limitations: [
        "The uploaded file is not yet sent to the FastAPI backend.",
        "Only one ECG lead is represented.",
        "Cross-dataset validation is not active in this demo.",
      ],
      warnings: [],
    },
  };
}

function getEvidenceStyle(
  direction: ClinicalEvidenceItem["direction"],
) {
  if (direction === "supports") {
    return {
      Icon: CheckCircle2,
      className:
        "border-emerald-200 bg-emerald-50 text-emerald-900",
      iconClassName: "text-emerald-600",
      label: "Supporting",
    };
  }

  if (direction === "contradicts") {
    return {
      Icon: ShieldAlert,
      className: "border-red-200 bg-red-50 text-red-900",
      iconClassName: "text-red-600",
      label: "Contradicting",
    };
  }

  return {
    Icon: AlertTriangle,
    className:
      "border-amber-200 bg-amber-50 text-amber-900",
    iconClassName: "text-amber-600",
    label: "Neutral",
  };
}

export function ECGAnalysisWorkspace() {
  const [selectedFiles, setSelectedFiles] =
    useState<File[]>([]);

  const {
    analyse,
    analysis,
    loading: isAnalysing,
    error,
    reset,
  } = useECGAnalysis();

  const [selectedBeatIndex, setSelectedBeatIndex] =
    useState<number | null>(null);

  const availableLeads = [
    "Lead I",
    "Lead II",
    "Lead III",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
  ];

  const [selectedLead, setSelectedLead] =
    useState("Lead II");

  const [activeExplanation, setActiveExplanation] = useState<
    "grad-cam" | "integrated-gradients"
  >("grad-cam");

  const abnormalBeats = useMemo(
    () =>
      analysis?.beats.filter((beat) => beat.isAbnormal) ?? [],
    [analysis],
  );

  const selectedLeadPoints = useMemo(() => {
    if (!analysis) {
      return [];
    }

    const leadIndex = Math.max(
      availableLeads.indexOf(selectedLead),
      0,
    );

    return analysis.waveform.points.map((point) => {
      const phaseShift = leadIndex * 0.015;
      const gain = 1 + leadIndex * 0.025;
      const polarity =
        selectedLead === "V1" || selectedLead === "V2"
          ? -0.75
          : 1;

      return {
        ...point,
        amplitude:
          point.amplitude * gain * polarity +
          0.025 *
            Math.sin(
              point.timeSeconds *
                Math.PI *
                (0.6 + phaseShift),
            ),
      };
    });
  }, [analysis, selectedLead]);

  const filteredAttributionRegions = useMemo(
    () =>
      analysis?.attributionRegions.filter(
        (region) => region.method === activeExplanation,
      ) ?? [],
    [analysis, activeExplanation],
  );

  const previousSelectedBeat = useMemo(() => {
    if (!analysis || selectedBeatIndex === null) {
      return null;
    }

    const currentIndex = analysis.beats.findIndex(
      (beat) => beat.beatIndex === selectedBeatIndex,
    );

    if (currentIndex <= 0) {
      return null;
    }

    return analysis.beats[currentIndex - 1] ?? null;
  }, [analysis, selectedBeatIndex]);

  const selectedBeat = useMemo(
    () =>
      analysis?.beats.find(
        (beat) => beat.beatIndex === selectedBeatIndex,
      ) ?? null,
    [analysis, selectedBeatIndex],
  );

  function handleFileChange(
    event: ChangeEvent<HTMLInputElement>,
  ) {
    const files = Array.from(event.target.files ?? []);

    setSelectedFiles(files);
    reset();
    setSelectedBeatIndex(null);
  }

  async function analyseFile() {
    if (selectedFiles.length === 0) {
      return;
    }

    try {
      const result = await analyse({
        files: selectedFiles,
        metadata: {
          selectedLead:
            selectedFiles.some((file) => file.name.toLowerCase().endsWith(".hea"))
              ? "MLII"
              : selectedLead,
        },
        includeExplanations: true,
        includeOverlay: true,
      });

      setSelectedBeatIndex(
        result.beats.find((beat) => beat.isAbnormal)?.beatIndex ?? null,
      );
    } catch {
      // Error state is handled by useECGAnalysis.
    }
  }

  if (!analysis) {
    return (
      <section className="overflow-hidden rounded-3xl border bg-white shadow-sm">
        <div className="border-b bg-slate-950 px-6 py-7 text-white">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl bg-blue-500/20 p-3">
              <FileHeart className="h-7 w-7 text-blue-400" />
            </div>

            <div>
              <h2 className="text-2xl font-semibold">
                ECG Analysis Workstation
              </h2>

              <p className="mt-1 text-sm text-slate-400">
                Upload a recording to begin signal analysis and
                explainable AI review.
              </p>
            </div>
          </div>
        </div>

        <div className="p-6">
          <label className="flex min-h-64 cursor-pointer flex-col items-center justify-center rounded-3xl border-2 border-dashed border-slate-300 bg-slate-50 p-8 text-center transition hover:border-blue-400 hover:bg-blue-50/40">
            <div className="rounded-2xl bg-blue-100 p-4">
              <Upload className="h-8 w-8 text-blue-600" />
            </div>

            <span className="mt-4 text-lg font-semibold">
              {selectedFiles.length > 0
                ? selectedFiles.map((file) => file.name).join(", ")
                : "Choose an ECG recording"}
            </span>

            <span className="mt-2 max-w-lg text-sm text-slate-500">
              CSV, NPY, TXT and WFDB HEA plus DAT recordings are supported.
            </span>

            <input
              type="file"
              multiple
              accept=".csv,.npy,.txt,.hea,.dat"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

          {error && (
            <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              {error}
            </div>
          )}

          <button
            type="button"
            onClick={analyseFile}
            disabled={selectedFiles.length === 0 || isAnalysing}
            className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-blue-600 px-6 py-4 font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isAnalysing ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Analysing recording
              </>
            ) : (
              <>
                <Brain className="h-5 w-5" />
                Start ECG analysis
              </>
            )}
          </button>
        </div>
      </section>
    );
  }

  return (
    <div className="space-y-5">
      <section className="overflow-hidden rounded-3xl border bg-slate-950 text-white shadow-xl">
        <div className="grid lg:grid-cols-[1fr_auto]">
          <div className="p-6">
            <div className="flex flex-wrap items-center gap-3">
              <span className="rounded-full bg-red-500/15 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-red-300">
                Abnormal rhythm detected
              </span>

              <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-300">
                {analysis.metadata.selectedLead}
              </span>

              <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-300">
                {analysis.metadata.samplingRateHz} Hz
              </span>
            </div>

            <h2 className="mt-4 text-3xl font-semibold">
              {analysis.prediction.className}
            </h2>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              {analysis.reasoning.conclusion}
            </p>
          </div>

          <div className="min-w-56 border-t border-white/10 lg:border-l lg:border-t-0">
            <div className="p-5">
              <p className="text-xs uppercase tracking-wide text-slate-500">
                Confidence
              </p>
              <p className="mt-2 text-3xl font-semibold text-emerald-400">
                {Math.round(
                  analysis.prediction.confidence * 100,
                )}
                %
              </p>
            </div>

            <div className="p-5">
              <p className="text-xs uppercase tracking-wide text-slate-500">
                Uncertainty
              </p>
              <p className="mt-2 text-3xl font-semibold text-amber-400">
                {Math.round(
                  analysis.uncertainty
                    .predictionUncertainty * 100,
                )}
                %
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-2xl border bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <Activity className="h-5 w-5 text-blue-600" />
            <span className="text-xs text-slate-400">
              Recording
            </span>
          </div>
          <p className="mt-4 text-2xl font-semibold">
            30 minutes
          </p>
          <p className="mt-1 text-sm text-slate-500">
            {analysis.metadata.totalSamples.toLocaleString()} samples
          </p>
        </div>

        <div className="rounded-2xl border bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <HeartPulse className="h-5 w-5 text-violet-600" />
            <span className="text-xs text-slate-400">
              R peaks
            </span>
          </div>
          <p className="mt-4 text-2xl font-semibold">
            {analysis.rPeaks.length}
          </p>
          <p className="mt-1 text-sm text-slate-500">
            Detected beats
          </p>
        </div>

        <div className="rounded-2xl border bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <ScanLine className="h-5 w-5 text-red-600" />
            <span className="text-xs text-slate-400">
              Abnormal
            </span>
          </div>
          <p className="mt-4 text-2xl font-semibold text-red-600">
            {analysis.prediction.abnormalBeatCount}
          </p>
          <p className="mt-1 text-sm text-slate-500">
            Influential beats
          </p>
        </div>

        <div className="rounded-2xl border bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <Waves className="h-5 w-5 text-emerald-600" />
            <span className="text-xs text-slate-400">
              Signal quality
            </span>
          </div>
          <p className="mt-4 text-2xl font-semibold text-emerald-600">
            {Math.round(analysis.signalQuality.score * 100)}%
          </p>
          <p className="mt-1 capitalize text-sm text-slate-500">
            {analysis.signalQuality.status}
          </p>
        </div>
      </section>

      <section className="space-y-5">
        <div className="space-y-5">
          <div className="rounded-3xl border bg-white p-4 shadow-sm">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold">
                  Explainable ECG viewer
                </h3>
                <p className="text-sm text-slate-500">
                  Select an explanation method and inspect the
                  highlighted regions.
                </p>
              </div>

              <div className="flex rounded-xl border bg-slate-50 p-1">
                {[
                  ["grad-cam", "Grad CAM"],
                  [
                    "integrated-gradients",
                    "Integrated Gradients",
                  ],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() =>
                      setActiveExplanation(
                        value as typeof activeExplanation,
                      )
                    }
                    className={`rounded-lg px-3 py-2 text-xs font-medium transition ${
                      activeExplanation === value
                        ? "bg-slate-950 text-white shadow-sm"
                        : "text-slate-600 hover:bg-white"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

                        <ECGLeadSelector
              leads={availableLeads}
              selectedLead={selectedLead}
              onSelectLead={setSelectedLead}
            />

            <ECGWaveformViewer
              points={selectedLeadPoints}
              rPeaks={analysis.rPeaks}
              attributionRegions={filteredAttributionRegions}
              title="30 minute ECG recording"
              leadName={selectedLead}
              samplingRateHz={
                analysis.metadata.samplingRateHz
              }
              visibleWindowSeconds={10}
              focusTimeSeconds={
                selectedBeat?.rPeakTimeSeconds ?? null
              }
            />
          </div>

          <div className="rounded-3xl border bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">
                  Abnormal event navigator
                </h3>
                <p className="text-sm text-slate-500">
                  Select an event to inspect it on the waveform.
                </p>
              </div>

              <span className="rounded-full bg-red-50 px-3 py-1 text-xs font-semibold text-red-600">
                {abnormalBeats.length} events
              </span>
            </div>

            <div className="mt-5 overflow-x-auto pb-2">
              <div className="flex min-w-max gap-3">
                {abnormalBeats.slice(0, 18).map((beat) => {
                  const selected =
                    beat.beatIndex === selectedBeatIndex;

                  return (
                    <button
                      key={beat.beatIndex}
                      type="button"
                      onClick={() =>
                        setSelectedBeatIndex(beat.beatIndex)
                      }
                      className={`w-48 rounded-2xl border p-4 text-left transition ${
                        selected
                          ? "border-blue-500 bg-blue-50 ring-2 ring-blue-100"
                          : "border-slate-200 bg-white hover:border-slate-300 hover:shadow-sm"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-400">
                          Beat {beat.beatIndex + 1}
                        </span>

                        <CircleDot className="h-4 w-4 text-red-500" />
                      </div>

                      <p className="mt-3 text-sm font-semibold text-slate-900">
                        {beat.prediction.className}
                      </p>

                      <p className="mt-2 text-xs text-slate-500">
                        {formatRecordingTime(
                          beat.rPeakTimeSeconds,
                        )}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
          <ECGEventTimeline
        durationSeconds={analysis.metadata.durationSeconds}
        beats={analysis.beats}
        regions={filteredAttributionRegions}
        selectedBeatIndex={selectedBeatIndex}
        onSelectBeat={setSelectedBeatIndex}
      />

        </div>

        <div className="space-y-5">
          <section className="rounded-3xl border bg-white p-5 shadow-sm">
            <h3 className="text-lg font-semibold">
              Selected beat
            </h3>

            {selectedBeat ? (
              <div className="mt-5 space-y-4">
                <div className="rounded-2xl bg-slate-950 p-5 text-white">
                  <p className="text-xs uppercase tracking-wide text-slate-400">
                    Beat {selectedBeat.beatIndex + 1}
                  </p>

                  <p className="mt-2 text-lg font-semibold">
                    {selectedBeat.prediction.className}
                  </p>

                  <p className="mt-3 text-3xl font-semibold text-emerald-400">
                    {Math.round(
                      selectedBeat.prediction.confidence *
                        100,
                    )}
                    %
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl border p-3">
                    <p className="text-xs text-slate-400">
                      Timestamp
                    </p>
                    <p className="mt-1 font-semibold">
                      {formatRecordingTime(
                        selectedBeat.rPeakTimeSeconds,
                      )}
                    </p>
                  </div>

                  <div className="rounded-xl border p-3">
                    <p className="text-xs text-slate-400">
                      Class
                    </p>
                    <p className="mt-1 font-semibold">
                      {selectedBeat.prediction.classCode}
                    </p>
                  </div>
                </div>

                <ECGBeatInspector beat={selectedBeat} />

                <ECGBeatMetrics
                  beat={selectedBeat}
                  previousBeat={previousSelectedBeat}
                />

                <div className="rounded-2xl border border-red-200 bg-red-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-red-600">
                    Model interpretation
                  </p>

                  <p className="mt-2 text-sm leading-6 text-red-900">
                    The model identified ventricular morphology
                    around the R peak. This region contributed
                    strongly to the predicted class.
                  </p>
                </div>
              </div>
            ) : (
              <p className="mt-4 text-sm text-slate-500">
                Select an abnormal event.
              </p>
            )}
          </section>

          <section className="rounded-3xl border bg-white p-5 shadow-sm">
            <h3 className="text-lg font-semibold">
              Recording details
            </h3>

            <div className="mt-4 grid gap-3 sm:grid-cols-2 text-sm">
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
                  className="rounded-xl border bg-slate-50 p-3"
                >
                  <span className="block text-xs text-slate-500">
                    {label}
                  </span>

                  <span className="mt-1 block font-medium text-slate-900">
                    {value}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </div>
      </section>

      <section className="grid items-start gap-5 xl:grid-cols-2">
        <div className="rounded-3xl border bg-white p-6 shadow-sm">
          <h3 className="text-lg font-semibold">
            Clinical evidence
          </h3>

          <div className="mt-5 space-y-3">
            {analysis.reasoning.evidence.map((item) => {
              const style = getEvidenceStyle(item.direction);
              const Icon = style.Icon;

              return (
                <div
                  key={item.id}
                  className={`rounded-2xl border p-4 ${style.className}`}
                >
                  <div className="flex items-start gap-3">
                    <Icon
                      className={`mt-0.5 h-5 w-5 shrink-0 ${style.iconClassName}`}
                    />

                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="font-semibold">
                          {item.featureName}
                        </p>

                        <span className="rounded-full bg-white/80 px-2 py-0.5 text-xs font-medium">
                          {style.label}
                        </span>
                      </div>

                      <p className="mt-2 text-sm leading-6">
                        {item.explanation}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="rounded-3xl border bg-white p-6 shadow-sm">
          <h3 className="text-lg font-semibold">
            Uncertainty and limitations
          </h3>

          <div className="mt-5 rounded-2xl bg-amber-50 p-5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-amber-900">
                Prediction uncertainty
              </span>

              <span className="text-2xl font-semibold text-amber-600">
                {Math.round(
                  analysis.uncertainty
                    .predictionUncertainty * 100,
                )}
                %
              </span>
            </div>

            <div className="mt-3 h-2 overflow-hidden rounded-full bg-amber-100">
              <div
                className="h-full rounded-full bg-amber-500"
                style={{
                  width: `${Math.round(
                    analysis.uncertainty
                      .predictionUncertainty * 100,
                  )}%`,
                }}
              />
            </div>
          </div>

          <div className="mt-5 space-y-3">
            {analysis.uncertainty.limitations.map(
              (limitation) => (
                <div
                  key={limitation}
                  className="flex items-start gap-3 rounded-2xl border p-4"
                >
                  <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-500" />

                  <p className="text-sm leading-6 text-slate-700">
                    {limitation}
                  </p>
                </div>
              ),
            )}
          </div>
        </div>
      </section>

      <ECGClinicalReport analysis={analysis} />
    </div>
  );
}


















