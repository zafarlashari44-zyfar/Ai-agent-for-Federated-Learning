import {
  PrescriptionsClient,
  type PatientPrescription,
} from "@/components/patient/prescriptions-client";
import {
  doctorMap,
  doctorName,
  getDoctorDirectory,
  parseMedications,
  prescriptionValidityPercent,
  requirePatientContext,
} from "@/lib/patient-data";

export default async function PatientPrescriptionsPage() {
  const { supabase, user, patient } = await requirePatientContext();
  const [{ data: prescriptions }, doctors] = await Promise.all([
    supabase
      .from("prescriptions")
      .select("*")
      .eq("patient_id", patient.id)
      .order("created_at", { ascending: false }),
    getDoctorDirectory(supabase),
  ]);

  const doctorsById = doctorMap(doctors);
  const assignedDoctor = patient.assigned_doctor_id
    ? doctorsById.get(patient.assigned_doctor_id)
    : undefined;
  const viewModels: PatientPrescription[] = (prescriptions ?? []).map(
    (prescription) => ({
      id: prescription.id,
      active: prescription.is_active,
      medications: parseMedications(prescription.medications),
      prescribedBy: doctorName(doctorsById.get(prescription.doctor_id)),
      createdAt: prescription.created_at,
      validUntil: prescription.valid_until,
      instructions: prescription.instructions,
      validityPercent: prescriptionValidityPercent(
        prescription.created_at,
        prescription.valid_until,
      ),
    }),
  );

  return (
    <PrescriptionsClient
      prescriptions={viewModels}
      userId={user.id}
      assignedDoctorProfileId={assignedDoctor?.profile_id ?? null}
    />
  );
}
