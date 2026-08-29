import { NextResponse } from "next/server";

import { requireApiRole } from "@/lib/api-auth";
import { createAdminClient } from "@/lib/supabase/admin";

export const dynamic = "force-dynamic";

type PatientProfile = {
  full_name: string | null;
  email: string | null;
};

type ECGRecord = {
  patient_id: string;
  demo_ecg_id: string | null;
  record_id: string;
  hea_filename: string | null;
  dat_filename: string | null;
  dataset: string;
  mapping_type: string;
  is_active: boolean;
};

function firstItem<T>(
  value: T | T[] | null | undefined,
): T | null {
  return Array.isArray(value)
    ? value[0] ?? null
    : value ?? null;
}

export async function GET() {
  const auth = await requireApiRole(["doctor", "admin"]);

  if (!auth.ok) {
    return NextResponse.json(
      { error: auth.error },
      { status: auth.status },
    );
  }

  const supabase = createAdminClient();

  const { data: patientData, error: patientError } =
    await supabase
      .from("patients")
      .select(`
        id,
        profiles (
          full_name,
          email
        )
      `)
      .order("created_at", { ascending: false });

  if (patientError) {
    return NextResponse.json(
      {
        error: "Unable to load patients.",
        details: patientError.message,
      },
      { status: 500 },
    );
  }

  // patient_ecg_records was added after the generated Supabase
  // Database types. Keep this query isolated until types are regenerated.
  const ecgSupabase = supabase as any;

  const { data: ecgData, error: ecgError } =
    await ecgSupabase
      .from("patient_ecg_records")
      .select(`
        patient_id,
        demo_ecg_id,
        record_id,
        hea_filename,
        dat_filename,
        dataset,
        mapping_type,
        is_active
      `)
      .eq("is_active", true)
      .eq("mapping_type", "demo_research");

  if (ecgError) {
    return NextResponse.json(
      {
        error: "Unable to load ECG mappings.",
        details: ecgError.message,
      },
      { status: 500 },
    );
  }

  const ecgByPatient = new Map<string, ECGRecord>();

  for (const record of (ecgData ?? []) as ECGRecord[]) {
    ecgByPatient.set(record.patient_id, record);
  }

  const patients = (patientData ?? []).map((patient) => {
    const profile = firstItem(
      patient.profiles as
        | PatientProfile
        | PatientProfile[]
        | null,
    );

    const ecg = ecgByPatient.get(patient.id) ?? null;

    return {
      id: patient.id,
      name: profile?.full_name ?? "Unknown Patient",
      email: profile?.email ?? null,

      ecg: ecg
        ? {
            demoEcgId: ecg.demo_ecg_id,
            recordId: ecg.record_id,
            heaFilename: ecg.hea_filename,
            datFilename: ecg.dat_filename,
            dataset: ecg.dataset,
            mappingType: ecg.mapping_type,
          }
        : null,
    };
  });

  return NextResponse.json({ patients });
}

