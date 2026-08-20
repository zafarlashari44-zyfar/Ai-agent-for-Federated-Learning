"use client";

import { useState } from "react";
import { parseECG } from "@/lib/ecg-parser";

export function useECGParser() {
  const [waveform, setWaveform] = useState<number[]>([]);
  const [duration, setDuration] = useState(0);
  const [loading, setLoading] = useState(false);

  async function load(file: File) {
    setLoading(true);

    const ecg = await parseECG(file);

    setWaveform(ecg.samples);
    setDuration(ecg.duration);

    setLoading(false);

    return ecg;
  }

  return {
    waveform,
    duration,
    loading,
    load,
  };
}
