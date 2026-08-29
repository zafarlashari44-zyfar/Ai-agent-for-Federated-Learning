import { Suspense } from "react";
import { ECGAnalysisWorkspace } from "@/components/ecg/ecg-analysis-workspace";

export default function ReportsPage() {
  return (
    <Suspense
      fallback={
        <div className="p-6 text-sm text-slate-500">
          Loading ECG workstation...
        </div>
      }
    >
      <ECGAnalysisWorkspace />
    </Suspense>
  );
}
