export function detectRPeaks(
  samples: number[],
  samplingRate = 360,
) {
  const peaks = [];

  for (let i = 1; i < samples.length - 1; i++) {
    if (
      samples[i] > samples[i - 1] &&
      samples[i] > samples[i + 1] &&
      samples[i] > 0.5
    ) {
      peaks.push({
        index: peaks.length,
        sampleIndex: i,
        timeSeconds: i / samplingRate,
        amplitude: samples[i],
      });

      i += Math.floor(samplingRate * 0.20);
    }
  }

  return peaks;
}
