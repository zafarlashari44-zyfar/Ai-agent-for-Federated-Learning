import {
  Activity,
  AlertTriangle,
  Brain,
  ShieldAlert,
} from "lucide-react";

import { createAdminClient } from "@/lib/supabase/admin";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const riskVariant = {
  Low: "success",
  Medium: "warning",
  High: "destructive",
} as const;

function getRiskVariant(risk: string | null) {
  if (risk === "High") return riskVariant.High;
  if (risk === "Medium") return riskVariant.Medium;
  return riskVariant.Low;
}

function formatConfidence(value: number | null | undefined) {
  if (value == null) return null;

  const percentage = value <= 1 ? value * 100 : value;

  return `${percentage.toFixed(1)}%`;
}

export default async function DiagnosesPage() {
  const supabase = createAdminClient();

  const { data, error } = await supabase
    .from("agent_outputs")
    .select("*")
    .order("created_at", { ascending: false });

  if (error) {
    throw new Error(`Unable to load AI diagnoses: ${error.message}`);
  }

  const diagnoses = (data ?? []).sort((a, b) => {
  const aGoverned = a.governance_route ? 1 : 0;
  const bGoverned = b.governance_route ? 1 : 0;

  if (aGoverned !== bGoverned) {
    return bGoverned - aGoverned;
  }

  return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
});

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold text-gray-900">
          AI ECG Diagnoses
        </h2>

        <p className="text-sm text-gray-500">
          {diagnoses.length} AI records with governance status
        </p>
      </div>

      <div className="space-y-3">
        {diagnoses.map((diag) => (
          <Card key={diag.id}>
            <CardContent className="p-4">
              <div className="flex items-start gap-4">
                <div className="rounded-full bg-purple-100 p-2">
                  <Brain className="h-5 w-5 text-purple-700" />
                </div>

                <div className="flex-1">
                  <div className="mb-2 flex flex-wrap items-center gap-2">
                    <h3 className="font-semibold text-gray-900">
                      {diag.prediction ?? "No prediction"}
                    </h3>

                    <Badge variant={getRiskVariant(diag.risk_flag)}>
                      {diag.risk_flag ?? "Low"}
                    </Badge>

                    {diag.confidence != null && (
                      <Badge variant="outline">
                        Confidence {formatConfidence(diag.confidence)}
                      </Badge>
                    )}

                    {diag.calibrated_confidence != null && (
                      <Badge variant="outline">
                        Calibrated{" "}
                        {formatConfidence(diag.calibrated_confidence)}
                      </Badge>
                    )}

                    {diag.human_review_required && (
                      <Badge variant="destructive">
                        Human Review Required
                      </Badge>
                    )}
                  </div>

                  <p className="text-sm text-gray-500">
                    Patient External ID {diag.patient_external_id}
                  </p>

                  <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
                    <div className="rounded-lg bg-gray-50 p-3">
                      <p className="text-xs font-semibold uppercase text-gray-400">
                        Urgency
                      </p>

                      <p className="text-sm text-gray-800">
                        {diag.urgency_level ?? "Not set"}
                      </p>
                    </div>

                    <div className="rounded-lg bg-gray-50 p-3">
                      <p className="text-xs font-semibold uppercase text-gray-400">
                        Uncertainty
                      </p>

                      <p className="text-sm text-gray-800">
                        {diag.uncertainty_level ?? "Not set"}
                      </p>
                    </div>

                    <div className="rounded-lg bg-gray-50 p-3">
                      <p className="text-xs font-semibold uppercase text-gray-400">
                        ECG Region
                      </p>

                      <p className="text-sm text-gray-800">
                        {diag.dominant_ecg_region ?? "Not set"}
                      </p>
                    </div>

                    <div className="rounded-lg bg-gray-50 p-3">
                      <p className="text-xs font-semibold uppercase text-gray-400">
                        Governance Route
                      </p>

                      <p className="text-sm text-gray-800">
                        {diag.governance_route ?? "Not set"}
                      </p>
                    </div>

                    <div className="rounded-lg bg-gray-50 p-3">
                      <p className="text-xs font-semibold uppercase text-gray-400">
                        Review Status
                      </p>

                      <p className="text-sm text-gray-800">
                        {diag.clinical_review_status ?? "Not set"}
                      </p>
                    </div>

                    <div className="rounded-lg bg-gray-50 p-3">
                      <p className="text-xs font-semibold uppercase text-gray-400">
                        Risk Score
                      </p>

                      <p className="text-sm text-gray-800">
                        {diag.risk_score != null
                          ? Number(diag.risk_score).toFixed(3)
                          : "Not set"}
                      </p>
                    </div>

                    {(diag.low_confidence_flag ||
                      diag.disagreement_flag ||
                      diag.blind_class_risk) && (
                      <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 md:col-span-2">
                        <div className="mb-2 flex items-center gap-2">
                          <ShieldAlert className="h-4 w-4 text-amber-600" />

                          <p className="text-xs font-semibold uppercase text-amber-700">
                            Governance Flags
                          </p>
                        </div>

                        <div className="flex flex-wrap gap-2">
                          {diag.low_confidence_flag && (
                            <Badge variant="outline">
                              Low confidence
                            </Badge>
                          )}

                          {diag.disagreement_flag && (
                            <Badge variant="outline">
                              Model agent disagreement
                            </Badge>
                          )}

                          {diag.blind_class_risk && (
                            <Badge variant="destructive">
                              Blind class risk
                            </Badge>
                          )}
                        </div>
                      </div>
                    )}

                    {diag.review_reason && (
                      <div className="rounded-lg border border-red-200 bg-red-50 p-3 md:col-span-2">
                        <div className="mb-1 flex items-center gap-2">
                          <AlertTriangle className="h-4 w-4 text-red-600" />

                          <p className="text-xs font-semibold uppercase text-red-700">
                            Review Reason
                          </p>
                        </div>

                        <p className="text-sm leading-6 text-gray-800">
                          {diag.review_reason}
                        </p>
                      </div>
                    )}

                    {diag.exclusion_statement && (
                      <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 md:col-span-2">
                        <p className="text-xs font-semibold uppercase text-amber-700">
                          Exclusion Statement
                        </p>

                        <p className="mt-1 text-sm leading-6 text-gray-800">
                          {diag.exclusion_statement}
                        </p>
                      </div>
                    )}

                    <div className="rounded-lg bg-gray-50 p-3 md:col-span-2">
                      <p className="text-xs font-semibold uppercase text-gray-400">
                        Suggested Action
                      </p>

                      <p className="text-sm text-gray-800">
                        {diag.suggested_next_action ?? "Not set"}
                      </p>
                    </div>

                    <div className="rounded-lg bg-gray-50 p-3 md:col-span-2">
                      <p className="text-xs font-semibold uppercase text-gray-400">
                        Doctor Technical Alert
                      </p>

                      <p className="text-sm text-gray-800">
                        {diag.doctor_technical_alert ?? "Not set"}
                      </p>
                    </div>

                    <div className="rounded-lg bg-blue-50 p-3 md:col-span-2">
                      <div className="mb-1 flex items-center gap-2">
                        <Activity className="h-4 w-4 text-blue-600" />

                        <p className="text-xs font-semibold uppercase text-blue-700">
                          Family Message
                        </p>
                      </div>

                      <p className="text-sm text-gray-800">
                        {diag.family_reassurance_message ?? "Not set"}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}

        {diagnoses.length === 0 && (
          <Card>
            <CardContent className="p-8 text-center text-gray-500">
              No AI ECG diagnoses found yet.
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}




