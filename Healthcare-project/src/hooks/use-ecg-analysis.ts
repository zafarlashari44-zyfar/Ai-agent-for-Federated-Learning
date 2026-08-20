"use client";

import { useCallback, useRef, useState } from "react";

import {
  analyseECG,
  type ECGUploadPayload,
} from "@/services/ecg-api";

import type {
  ECGAnalysisResult,
} from "@/types/ecg-analysis";

export function useECGAnalysis() {
  const [analysis, setAnalysis] =
    useState<ECGAnalysisResult | null>(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] =
    useState<string | null>(null);

  const controllerRef =
    useRef<AbortController | null>(null);

  const analyse = useCallback(
    async (
      payload: ECGUploadPayload,
    ): Promise<ECGAnalysisResult> => {
      controllerRef.current?.abort();

      const controller = new AbortController();
      controllerRef.current = controller;

      setLoading(true);
      setError(null);

      try {
        const result = await analyseECG(
          payload,
          controller.signal,
        );

        setAnalysis(result);

        return result;
      } catch (caughtError) {
        if (
          caughtError instanceof Error &&
          caughtError.name === "AbortError"
        ) {
          throw caughtError;
        }

        const message =
          caughtError instanceof Error
            ? caughtError.message
            : "ECG analysis failed.";

        setError(message);

        throw caughtError;
      } finally {
        if (controllerRef.current === controller) {
          controllerRef.current = null;
        }

        setLoading(false);
      }
    },
    [],
  );

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;

    setAnalysis(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    analyse,
    analysis,
    loading,
    error,
    reset,
  };
}
