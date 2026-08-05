"use client";

import { useState } from "react";
import { analyseECG } from "@/lib/ecg-api";

export function useECGAnalysis() {
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function analyse(formData: FormData) {
    try {
      setLoading(true);
      setError(null);

      const result = await analyseECG(formData);

      setAnalysis(result);

      return result;
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Analysis failed.",
      );

      throw err;
    } finally {
      setLoading(false);
    }
  }

  return {
    analyse,
    analysis,
    loading,
    error,
  };
}
