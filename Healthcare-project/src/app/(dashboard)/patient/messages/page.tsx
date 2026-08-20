import { MessagesClient } from "@/components/patient/messages-client";
import {
  getDoctorDirectory,
  requirePatientContext,
} from "@/lib/patient-data";

export default async function PatientMessagesPage() {
  const { supabase, user, patient } = await requirePatientContext();
  const [{ data: messages }, doctors] = await Promise.all([
    supabase
      .from("messages")
      .select("*")
      .or(`sender_id.eq.${user.id},receiver_id.eq.${user.id}`)
      .order("created_at"),
    getDoctorDirectory(supabase),
  ]);
  const assignedDoctor =
    doctors.find((doctor) => doctor.doctor_id === patient.assigned_doctor_id) ??
    null;
  const doctorMessages = assignedDoctor
    ? (messages ?? []).filter(
        (message) =>
          message.sender_id === assignedDoctor.profile_id ||
          message.receiver_id === assignedDoctor.profile_id,
      )
    : [];

  return (
    <MessagesClient
      userId={user.id}
      doctor={assignedDoctor}
      messages={doctorMessages}
    />
  );
}
