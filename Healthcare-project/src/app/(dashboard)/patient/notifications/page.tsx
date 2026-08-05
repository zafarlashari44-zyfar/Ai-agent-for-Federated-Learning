import { NotificationsClient } from "@/components/patient/notifications-client";
import { requirePatientContext } from "@/lib/patient-data";

export default async function PatientNotificationsPage() {
  const { supabase, user } = await requirePatientContext();
  const { data: notifications } = await supabase
    .from("notifications")
    .select("*")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  return <NotificationsClient notifications={notifications ?? []} />;
}
