import { Brain, Activity } from "lucide-react";

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

export default async function DiagnosesPage() {
  const supabase = createAdminClient();

  const { data, error } = await supabase
    .from("agent_outputs")
    .select("*")
    .order("created_at", { ascending: false });

  if (error) {
    throw new Error(`Unable to load AI diagnoses: ${error.message}`);
  }

  const diagnoses = data ?? [];

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold text-gray-900">AI ECG Diagnoses</h2>
        <p className="text-sm text-gray-500">{diagnoses.length} AI records</p>
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

                    {diag.confidence !== null && (
                      <Badge variant="outline">
                        Confidence: {Number(diag.confidence).toFixed(2)}%
                      </Badge>
                    )}
                  </div>

                  <p className="text-sm text-gray-500">
                    Patient External ID: {diag.patient_external_id}
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
                        ECG Region
                      </p>
                      <p className="text-sm text-gray-800">
                        {diag.dominant_ecg_region ?? "Not set"}
                      </p>
                    </div>

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