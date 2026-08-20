import { TopNav } from "@/components/layout/TopNav";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { PatientSidebar } from "@/components/patient/patient-sidebar";
import { requirePatientContext } from "@/lib/patient-data";

export default async function PatientLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { supabase, user, profile } = await requirePatientContext();
  const [{ count: unreadMessages }, { count: unreadNotifications }] =
    await Promise.all([
      supabase
        .from("messages")
        .select("id", { count: "exact", head: true })
        .eq("receiver_id", user.id)
        .eq("is_read", false),
      supabase
        .from("notifications")
        .select("id", { count: "exact", head: true })
        .eq("user_id", user.id)
        .eq("is_read", false),
    ]);

  return (
    <DashboardShell
      sidebar={
        <PatientSidebar
          userName={profile.full_name}
          userAvatar={profile.avatar_url ?? undefined}
          unreadMessages={unreadMessages ?? 0}
          unreadNotifications={unreadNotifications ?? 0}
        />
      }
      topNav={
        <TopNav
          title="Patient Portal"
          role="patient"
          userName={profile.full_name}
          avatarUrl={profile.avatar_url ?? undefined}
          notificationCount={unreadNotifications ?? 0}
        />
      }
    >
      {children}
    </DashboardShell>
  );
}
