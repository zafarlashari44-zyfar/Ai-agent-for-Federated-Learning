"use client";
import { useCallback, useEffect, useState } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopNav } from "@/components/layout/TopNav";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { createClient } from "@/lib/supabase/client";
import {
  LayoutDashboard, Users, UserCheck, BarChart3, Shield, Calendar,
  CreditCard, Settings, ScrollText, Bell,
} from "lucide-react";

const baseNavItems = [
  { label: "Overview", href: "/admin", icon: LayoutDashboard },
  { label: "Doctors", href: "/admin/doctors", icon: UserCheck },
  { label: "Patients", href: "/admin/patients", icon: Users },
  { label: "Appointments", href: "/admin/appointments", icon: Calendar },
  { label: "Analytics", href: "/admin/analytics", icon: BarChart3 },
  { label: "Billing", href: "/admin/billing", icon: CreditCard },
  { label: "Roles & Access", href: "/admin/roles", icon: Shield },
  { label: "Activity Logs", href: "/admin/logs", icon: ScrollText },
  { label: "Notifications", href: "/admin/notifications", icon: Bell },
  { label: "Settings", href: "/admin/settings", icon: Settings },
];

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const [userName, setUserName] = useState("Administrator");
  const [unreadNotifications, setUnreadNotifications] = useState(0);
  const [pendingRequests, setPendingRequests] = useState(0);

  const loadAdminShellData = useCallback(async () => {
    const supabase = createClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) return;

    const [profileResult, notificationResult, requestResult] =
      await Promise.all([
        supabase
          .from("profiles")
          .select("full_name")
          .eq("id", user.id)
          .single(),
        supabase
          .from("notifications")
          .select("id", { count: "exact", head: true })
          .eq("user_id", user.id)
          .eq("is_read", false),
        supabase
          .from("role_requests")
          .select("id", { count: "exact", head: true })
          .eq("status", "pending"),
      ]);

    if (profileResult.data?.full_name) {
      setUserName(profileResult.data.full_name);
    }
    if (!notificationResult.error) {
      setUnreadNotifications(notificationResult.count ?? 0);
    }
    if (!requestResult.error) {
      setPendingRequests(requestResult.count ?? 0);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadAdminShellData();
    }, 0);
    const interval = window.setInterval(() => {
      void loadAdminShellData();
    }, 10000);
    const handleFocus = () => {
      void loadAdminShellData();
    };
    window.addEventListener("focus", handleFocus);

    return () => {
      window.clearTimeout(timer);
      window.clearInterval(interval);
      window.removeEventListener("focus", handleFocus);
    };
  }, [loadAdminShellData]);

  const navItems = baseNavItems.map((item) => {
    if (item.href === "/admin/roles") {
      return { ...item, badge: pendingRequests };
    }
    if (item.href === "/admin/notifications") {
      return { ...item, badge: unreadNotifications };
    }
    return item;
  });

  return (
    <DashboardShell
      sidebar={<Sidebar items={navItems} role="admin" userName={userName} />}
      topNav={
        <TopNav
          title="Admin Dashboard"
          role="admin"
          userName={userName}
          notificationCount={unreadNotifications}
        />
      }
    >
      {children}
    </DashboardShell>
  );
}
