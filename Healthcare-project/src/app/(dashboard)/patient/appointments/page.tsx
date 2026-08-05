import { AppointmentsClient } from "@/components/patient/appointments-client";
import {
  getDoctorDirectory,
  requirePatientContext,
} from "@/lib/patient-data";

export default async function PatientAppointmentsPage() {
  const { supabase, patient } = await requirePatientContext();
  const [{ data: appointments }, doctors] = await Promise.all([
    supabase
      .from("appointments")
      .select("*")
      .eq("patient_id", patient.id)
      .order("scheduled_at", { ascending: false }),
    getDoctorDirectory(supabase),
  ]);

  return (
    <AppointmentsClient
      patientId={patient.id}
      appointments={appointments ?? []}
      doctors={doctors}
    />
  );
}
