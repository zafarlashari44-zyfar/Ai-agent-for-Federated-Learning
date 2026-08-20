import {
  RecordsClient,
  type PatientMedicalRecord,
} from "@/components/patient/records-client";
import {
  doctorMap,
  doctorName,
  getDoctorDirectory,
  parseMedications,
  requirePatientContext,
} from "@/lib/patient-data";
import type { Json } from "@/types/database";

function normalizeVitals(value: Json | null) {
  if (!value || Array.isArray(value) || typeof value !== "object") {
    return {};
  }

  return Object.fromEntries(
    Object.entries(value).flatMap(([key, item]) =>
      typeof item === "string" || typeof item === "number"
        ? [[key, item]]
        : [],
    ),
  );
}

export default async function PatientRecordsPage() {
  const { supabase, patient } = await requirePatientContext();
  const [
    { data: records },
    { data: prescriptions },
    doctors,
  ] = await Promise.all([
    supabase
      .from("medical_records")
      .select("*")
      .eq("patient_id", patient.id)
      .order("created_at", { ascending: false }),
    supabase
      .from("prescriptions")
      .select("medical_record_id, medications")
      .eq("patient_id", patient.id),
    getDoctorDirectory(supabase),
  ]);

  const doctorsById = doctorMap(doctors);
  const medicationByRecord = new Map<string, string[]>();
  prescriptions?.forEach((prescription) => {
    if (!prescription.medical_record_id) {
      return;
    }
    medicationByRecord.set(
      prescription.medical_record_id,
      parseMedications(prescription.medications).map(
        (medication) =>
          `${medication.name}${medication.dosage ? ` — ${medication.dosage}` : ""}`,
      ),
    );
  });

  const viewModels: PatientMedicalRecord[] = (records ?? []).map((record) => {
    const doctor = doctorsById.get(record.doctor_id);
    return {
      id: record.id,
      createdAt: record.created_at,
      diagnosis: record.diagnosis,
      doctor: doctorName(doctor),
      specialty: doctor?.specialization || "Medical care",
      symptoms: record.symptoms,
      treatmentPlan: record.treatment_plan,
      notes: record.notes,
      vitals: normalizeVitals(record.vitals),
      prescriptions: medicationByRecord.get(record.id) ?? [],
    };
  });

  return <RecordsClient records={viewModels} />;
}
