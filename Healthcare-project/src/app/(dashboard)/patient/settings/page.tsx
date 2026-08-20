import { SettingsClient } from "@/components/patient/settings-client";
import { requirePatientContext } from "@/lib/patient-data";

export default async function PatientSettingsPage() {
  const { supabase, user, profile, patient } = await requirePatientContext();
  let { data: preferences } = await supabase
    .from("notification_preferences")
    .select("*")
    .eq("user_id", user.id)
    .maybeSingle();

  if (!preferences) {
    const { data } = await supabase
      .from("notification_preferences")
      .insert({ user_id: user.id })
      .select()
      .single();
    preferences = data;
  }

  if (!preferences) {
    throw new Error("Unable to load notification preferences.");
  }

  return (
    <SettingsClient
      profile={profile}
      patient={patient}
      preferences={preferences}
    />
  );
}
