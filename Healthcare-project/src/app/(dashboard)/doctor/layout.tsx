"use client";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopNav } from "@/components/layout/TopNav";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { useProfileName } from "@/hooks/use-profile-name";
import {
  LayoutDashboard, Users, Calendar, FileText, Activity,
  Bell, Settings, Stethoscope, ClipboardList,
} from "lucide-react";

const navItems = [
  { label: "Overview", href: "/doctor", icon: LayoutDashboard },
  { label: "Patients", href: "/doctor/patients", icon: Users, badge: 0 },
  { label: "Appointments", href: "/doctor/appointments", icon: Calendar },
  { label: "Diagnoses", href: "/doctor/diagnoses", icon: Stethoscope },
  { label: "Prescriptions", href: "/doctor/prescriptions", icon: ClipboardList },
  { label: "Reports", href: "/doctor/reports", icon: FileText },
  { label: "Analytics", href: "/doctor/analytics", icon: Activity },
  { label: "Notifications", href: "/doctor/notifications", icon: Bell, badge: 3 },
  { label: "Settings", href: "/doctor/settings", icon: Settings },
];

export default function DoctorLayout({ children }: { children: React.ReactNode }) {
  const userName = useProfileName("Doctor");

  return (
    <DashboardShell
      sidebar={<Sidebar items={navItems} role="doctor" userName={userName} />}
      topNav={<TopNav title="Doctor Dashboard" role="doctor" userName={userName} notificationCount={3} />}
    >
      {children}
    </DashboardShell>
  );
}
