"use client";

import { useState } from "react";
import { AlertTriangle, Pill, Printer, RefreshCw } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { formatDate } from "@/lib/utils";
import type { Medication } from "@/lib/patient-types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

export interface PatientPrescription {
  id: string;
  active: boolean;
  medications: Medication[];
  prescribedBy: string;
  createdAt: string;
  validUntil: string | null;
  instructions: string | null;
  validityPercent: number;
}

export function PrescriptionsClient({
  prescriptions,
  userId,
  assignedDoctorProfileId,
}: {
  prescriptions: PatientPrescription[];
  userId: string;
  assignedDoctorProfileId: string | null;
}) {
  const [requestingId, setRequestingId] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const active = prescriptions.filter((prescription) => prescription.active);
  const completed = prescriptions.filter((prescription) => !prescription.active);
  const expiring = active.filter((prescription) => prescription.validityPercent <= 25);

  async function requestRefill(prescription: PatientPrescription) {
    if (!assignedDoctorProfileId) {
      setMessage("No doctor is assigned to your patient profile yet.");
      return;
    }

    setRequestingId(prescription.id);
    setMessage("");
    const names = prescription.medications.map((item) => item.name).join(", ");
    const supabase = createClient();
    const { error } = await supabase.from("messages").insert({
      sender_id: userId,
      receiver_id: assignedDoctorProfileId,
      content: `Refill request for ${names || "prescription"} (prescription ${prescription.id}).`,
    });
    setRequestingId(null);
    setMessage(error ? error.message : "Refill request sent to your assigned doctor.");
  }

  return (
    <div className="space-y-5 max-w-3xl">
      <div>
        <h2 className="text-xl font-bold text-gray-900">My Prescriptions</h2>
        <p className="text-sm text-gray-500">{active.length} active prescriptions</p>
      </div>

      {message && (
        <p className="rounded-lg bg-blue-50 p-3 text-sm text-blue-700">{message}</p>
      )}

      {expiring.length > 0 && (
        <div className="flex items-center gap-3 bg-amber-50 border border-amber-200 rounded-xl p-4">
          <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0" />
          <div>
            <p className="text-sm font-medium text-amber-800">Prescription Expiring Soon</p>
            <p className="text-xs text-amber-600">
              {expiring
                .flatMap((prescription) =>
                  prescription.medications.map((medication) => medication.name),
                )
                .join(", ")}
            </p>
          </div>
        </div>
      )}

      <section>
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
          Active Prescriptions
        </h3>
        {active.length ? (
          <div className="space-y-4">
            {active.map((prescription) => (
              <Card key={prescription.id} className="border-green-100">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="text-xs text-gray-400">
                        Prescribed by {prescription.prescribedBy}
                      </p>
                      <p className="text-xs text-gray-400">
                        {prescription.validUntil
                          ? `Valid until ${formatDate(prescription.validUntil)}`
                          : "No expiration date"}
                      </p>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="gap-1.5 text-xs h-7"
                        onClick={() => window.print()}
                      >
                        <Printer className="h-3 w-3" /> Print
                      </Button>
                      <Button
                        size="sm"
                        className="gap-1.5 text-xs h-7 bg-green-600 hover:bg-green-700"
                        onClick={() => requestRefill(prescription)}
                        disabled={requestingId === prescription.id}
                      >
                        <RefreshCw className="h-3 w-3" />
                        {requestingId === prescription.id ? "Sending..." : "Request Refill"}
                      </Button>
                    </div>
                  </div>
                  {prescription.validUntil && (
                    <div className="mt-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-gray-500">Prescription validity remaining</span>
                        <span className="text-xs font-medium text-gray-700">
                          {prescription.validityPercent}%
                        </span>
                      </div>
                      <Progress value={prescription.validityPercent} className="h-2" />
                    </div>
                  )}
                </CardHeader>
                <CardContent className="pt-0 space-y-3">
                  {prescription.medications.length ? (
                    prescription.medications.map((medication) => (
                      <div
                        key={`${prescription.id}-${medication.name}`}
                        className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg"
                      >
                        <div className="p-2 bg-green-50 rounded-lg">
                          <Pill className="h-4 w-4 text-green-600" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-semibold text-gray-900 text-sm">
                            {medication.name}
                          </p>
                          {medication.dosage && (
                            <p className="text-xs text-gray-500 mb-2">
                              {medication.dosage}
                            </p>
                          )}
                          <div className="flex flex-wrap gap-1.5">
                            {medication.morning && <Badge variant="info">Morning</Badge>}
                            {medication.evening && <Badge variant="secondary">Evening</Badge>}
                            {medication.withFood && <Badge variant="warning">With food</Badge>}
                          </div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500">
                      Medication details have not been entered.
                    </p>
                  )}
                  {prescription.instructions && (
                    <p className="text-xs text-gray-500 italic border-l-2 border-green-200 pl-3">
                      {prescription.instructions}
                    </p>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="p-8 text-center text-sm text-gray-500">
              No active prescriptions.
            </CardContent>
          </Card>
        )}
      </section>

      <section>
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
          Completed Prescriptions
        </h3>
        {completed.length ? completed.map((prescription) => (
          <Card key={prescription.id} className="opacity-70 mb-3">
            <CardContent className="p-4 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <Pill className="h-4 w-4 text-gray-400" />
                  <span className="text-sm font-medium text-gray-600">
                    {prescription.medications.map((item) => item.name).join(", ") || "Prescription"}
                  </span>
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  {prescription.prescribedBy} · {formatDate(prescription.createdAt)}
                </p>
              </div>
              <Badge variant="secondary">Completed</Badge>
            </CardContent>
          </Card>
        )) : (
          <Card>
            <CardContent className="p-8 text-center text-sm text-gray-500">
              No completed prescriptions.
            </CardContent>
          </Card>
        )}
      </section>
    </div>
  );
}
