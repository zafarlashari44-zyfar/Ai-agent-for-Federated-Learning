import { DocumentsClient } from "@/components/patient/documents-client";
import { requirePatientContext } from "@/lib/patient-data";

export default async function PatientDocumentsPage() {
  const { supabase, user, patient } = await requirePatientContext();
  const { data: documents } = await supabase
    .from("documents")
    .select("*")
    .eq("patient_id", patient.id)
    .order("created_at", { ascending: false });

  return (
    <DocumentsClient
      documents={documents ?? []}
      patientId={patient.id}
      userId={user.id}
    />
  );
}
