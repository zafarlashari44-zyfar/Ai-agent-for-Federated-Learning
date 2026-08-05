import { HealthClient } from "@/components/patient/health-client";
import { requirePatientContext } from "@/lib/patient-data";

export default async function PatientHealthPage() {
  const { supabase, user, patient } = await requirePatientContext();
  const [{ data: readings }, { data: goals }] = await Promise.all([
    supabase
      .from("vital_readings")
      .select("*")
      .eq("patient_id", patient.id)
      .order("recorded_at", { ascending: false })
      .limit(24),
    supabase
      .from("health_goals")
      .select("*")
      .eq("patient_id", patient.id)
      .order("created_at", { ascending: false }),
  ]);

  return (
    <HealthClient
      patientId={patient.id}
      userId={user.id}
      readings={readings ?? []}
      goals={goals ?? []}
    />
  );
}
