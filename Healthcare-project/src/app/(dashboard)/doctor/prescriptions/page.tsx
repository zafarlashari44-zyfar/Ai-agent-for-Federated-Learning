import { Pill, Printer } from "lucide-react";

import { createAdminClient } from "@/lib/supabase/admin";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { getInitials, formatDate } from "@/lib/utils";

const statusVariant = {
  active: "success",
  completed: "secondary",
  expired: "destructive",
} as const;

type PrescriptionStatus = keyof typeof statusVariant;

type Medication = {
  name?: string;
  dosage?: string;
  duration?: string;
};

type Profile = {
  full_name: string | null;
  email: string | null;
  role: string | null;
};

type Patient = {
  id: string;
  profiles: Profile | Profile[] | null;
};

type Prescription = {
  id: string;
  medications: unknown;
  instructions: string | null;
  is_active: boolean | null;
  valid_until: string | null;
  created_at: string | null;
  patients: Patient | Patient[] | null;
};

function firstItem<T>(value: T | T[] | null | undefined): T | null {
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function parseMedications(value: unknown): Medication[] {
  if (!Array.isArray(value)) return [];

  return value
    .filter((item): item is Medication => {
      return item !== null && typeof item === "object";
    })
    .map((item) => ({
      name: item.name,
      dosage: item.dosage,
      duration: item.duration,
    }));
}

function getPrescriptionStatus(
  isActive: boolean | null,
  validUntil: string | null
): PrescriptionStatus {
  if (validUntil) {
    const expiryDate = new Date(validUntil);
    const today = new Date();

    if (!Number.isNaN(expiryDate.getTime()) && expiryDate < today) {
      return "expired";
    }
  }

  return isActive ? "active" : "completed";
}

export default async function PrescriptionsPage() {
  const supabase = createAdminClient();

  const { data, error } = await supabase
    .from("prescriptions")
    .select(
      `
      id,
      medications,
      instructions,
      is_active,
      valid_until,
      created_at,
      patients (
        id,
        profiles (
          full_name,
          email,
          role
        )
      )
    `
    )
    .order("created_at", { ascending: false });

  if (error) {
    throw new Error(`Unable to load prescriptions: ${error.message}`);
  }

  const prescriptions = (data ?? []) as Prescription[];

  const realPrescriptions = prescriptions.filter((rx) => {
    const patient = firstItem(rx.patients);
    const profile = firstItem(patient?.profiles);

    return profile?.role === "patient";
  });

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Prescriptions</h2>
        <p className="text-sm text-gray-500">
          {realPrescriptions.length} real records
        </p>
      </div>

      {realPrescriptions.length === 0 ? (
        <Card>
          <CardContent className="p-6 text-sm text-gray-500">
            No real prescriptions found.
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {realPrescriptions.map((rx) => {
            const patient = firstItem(rx.patients);
            const profile = firstItem(patient?.profiles);

            const patientName = profile?.full_name ?? "Unknown Patient";
            const medications = parseMedications(rx.medications);
            const status = getPrescriptionStatus(rx.is_active, rx.valid_until);

            return (
              <Card key={rx.id} className="transition-shadow hover:shadow-sm">
                <CardContent className="p-4">
                  <div className="flex items-start gap-4">
                    <Avatar className="h-9 w-9 shrink-0">
                      <AvatarFallback className="bg-green-100 text-sm text-green-700">
                        {getInitials(patientName)}
                      </AvatarFallback>
                    </Avatar>

                    <div className="min-w-0 flex-1">
                      <div className="mb-1 flex items-center justify-between gap-3">
                        <div className="flex min-w-0 items-center gap-2">
                          <p className="truncate font-semibold text-gray-900">
                            {patientName}
                          </p>
                          <Badge variant={statusVariant[status]}>
                            {status}
                          </Badge>
                        </div>

                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 shrink-0 text-gray-400 hover:text-gray-700"
                          aria-label={`Print prescription for ${patientName}`}
                        >
                          <Printer className="h-3.5 w-3.5" />
                        </Button>
                      </div>

                      <p className="mb-3 text-xs text-gray-400">
                        Issued:{" "}
                        {rx.created_at ? formatDate(rx.created_at) : "Not set"}{" "}
                        · Valid until:{" "}
                        {rx.valid_until ? formatDate(rx.valid_until) : "Not set"}
                      </p>

                      <div className="flex flex-wrap gap-2">
                        {medications.length > 0 ? (
                          medications.map((med, index) => (
                            <div
                              key={`${rx.id}-${index}`}
                              className="flex items-center gap-1.5 rounded-lg border border-gray-100 bg-gray-50 px-2.5 py-1.5"
                            >
                              <Pill className="h-3 w-3 shrink-0 text-green-600" />
                              <div>
                                <p className="text-xs font-medium text-gray-800">
                                  {med.name || "Medication"}
                                </p>
                                <p className="text-xs text-gray-400">
                                  {med.dosage || "No dosage"}
                                  {med.duration ? ` · ${med.duration}` : ""}
                                </p>
                              </div>
                            </div>
                          ))
                        ) : (
                          <p className="text-xs text-gray-400">
                            No medications listed.
                          </p>
                        )}
                      </div>

                      {rx.instructions ? (
                        <p className="mt-2 text-xs italic text-gray-500">
                          Note: {rx.instructions}
                        </p>
                      ) : null}
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}