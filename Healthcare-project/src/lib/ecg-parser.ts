export async function parseECG(file: File) {
  const text = await file.text();

  const samples = text
    .split(/\r?\n/)
    .flatMap(line => line.split(","))
    .map(v => Number(v.trim()))
    .filter(v => !Number.isNaN(v));

  return {
    samples,
    samplingRate: 360,
    duration: samples.length / 360,
  };
}
