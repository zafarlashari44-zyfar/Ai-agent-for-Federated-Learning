"use client";

import {
  Bell,
  Calendar,
  ClipboardList,
  FileText,
  FolderOpen,
  Heart,
  LayoutDashboard,
  MessageSquare,
  Settings,
} from "lucide-react";
import { Sidebar } from "@/components/layout/Sidebar";

export function PatientSidebar({
  userName,
  userAvatar,
  unreadMessages,
  unreadNotifications,
}: {
  userName: string;
  userAvatar?: string;
  unreadMessages: number;
  unreadNotifications: number;
}) {
  const items = [
    { label: "Overview", href: "/patient", icon: LayoutDashboard },
    { label: "Medical Records", href: "/patient/records", icon: FileText },
    { label: "Appointments", href: "/patient/appointments", icon: Calendar },
    {
      label: "Prescriptions",
      href: "/patient/prescriptions",
      icon: ClipboardList,
    },
    { label: "Documents", href: "/patient/documents", icon: FolderOpen },
    {
      label: "Messages",
      href: "/patient/messages",
      icon: MessageSquare,
      badge: unreadMessages,
    },
    {
      label: "Notifications",
      href: "/patient/notifications",
      icon: Bell,
      badge: unreadNotifications,
    },
    { label: "My Health", href: "/patient/health", icon: Heart },
    { label: "Settings", href: "/patient/settings", icon: Settings },
  ];

  return (
    <Sidebar
      items={items}
      role="patient"
      userName={userName}
      userAvatar={userAvatar}
    />
  );
}
