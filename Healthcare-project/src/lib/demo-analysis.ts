import { detectRPeaks } from "./rpeak-detector";

export async function analyseDemoECG(file: File) {
  const text = await file.text();

  const samples = text
    .split(/\r?\n/)
    .flatMap(line => line.split(","))
    .map(Number)
    .filter(v => !Number.isNaN(v));

  const waveform = samples.map((value, index) => ({
    sampleIndex: index,
    timeSeconds: index / 360,
    amplitude: value,
  }));

  return {
    waveform,
    rPeaks: detectRPeaks(samples),
    attributionRegions: [],
    prediction: "Normal",
    confidence: 0.98,
  };
}
