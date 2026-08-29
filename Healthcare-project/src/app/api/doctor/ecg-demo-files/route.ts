import fs from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { requireApiRole } from "@/lib/api-auth";
import { createAdminClient } from "@/lib/supabase/admin";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type ECGMapping = {
  patient_id: string;
  demo_ecg_id: string | null;
  record_id: string;
  hea_filename: string | null;
  dat_filename: string | null;
  mapping_type: string;
  is_active: boolean;
};

function isUuid(value: string) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
    value,
  );
}

export async function GET(request: Request) {
  const auth = await requireApiRole(["doctor", "admin"]);

  if (!auth.ok) {
    return NextResponse.json(
      { error: auth.error },
      { status: auth.status },
    );
  }

  const url = new URL(request.url);
  const patientId =
    url.searchParams.get("patient_id")?.trim() ?? "";

  if (!patientId || !isUuid(patientId)) {
    return NextResponse.json(
      { error: "A valid patient_id is required." },
      { status: 400 },
    );
  }

  const supabase = createAdminClient();

  // Temporary cast because patient_ecg_records was added
  // after the generated Supabase Database types.
  const ecgSupabase = supabase as any;

  const { data, error } = await ecgSupabase
    .from("patient_ecg_records")
    .select(`
      patient_id,
      demo_ecg_id,
      record_id,
      hea_filename,
      dat_filename,
      mapping_type,
      is_active
    `)
    .eq("patient_id", patientId)
    .eq("mapping_type", "demo_research")
    .eq("is_active", true)
    .maybeSingle();

  if (error) {
    return NextResponse.json(
      {
        error: "Unable to load the patient's ECG assignment.",
        details: error.message,
      },
      { status: 500 },
    );
  }

  const mapping = data as ECGMapping | null;

  if (!mapping) {
    return NextResponse.json(
      { error: "No active demo ECG is assigned to this patient." },
      { status: 404 },
    );
  }

  if (!mapping.hea_filename || !mapping.dat_filename) {
    return NextResponse.json(
      { error: "The ECG assignment does not contain both WFDB files." },
      { status: 500 },
    );
  }

  // Healthcare-project is directly inside repository root.
  const datasetDirectory = path.resolve(
    process.cwd(),
    "..",
    "datasets",
    "mitdb",
  );

  const heaPath = path.join(
    datasetDirectory,
    path.basename(mapping.hea_filename),
  );

  const datPath = path.join(
    datasetDirectory,
    path.basename(mapping.dat_filename),
  );

  try {
    const [heaBuffer, datBuffer] = await Promise.all([
      fs.readFile(heaPath),
      fs.readFile(datPath),
    ]);

    return NextResponse.json({
      demoEcgId: mapping.demo_ecg_id,
      recordId: mapping.record_id,

      files: [
        {
          name: mapping.hea_filename,
          type: "text/plain",
          base64: heaBuffer.toString("base64"),
        },
        {
          name: mapping.dat_filename,
          type: "application/octet-stream",
          base64: datBuffer.toString("base64"),
        },
      ],
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: `Assigned MIT-BIH record ${mapping.record_id} was not found in the local dataset.`,
        details:
          error instanceof Error
            ? error.message
            : String(error),
      },
      { status: 404 },
    );
  }
}
