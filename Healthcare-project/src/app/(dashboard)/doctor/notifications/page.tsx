"use client";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bell, Calendar, FileText, MessageSquare, AlertCircle, CheckCircle, X } from "lucide-react";
import { formatDateTime } from "@/lib/utils";
import { cn } from "@/lib/utils";

const notifTypes = {
  appointment: { icon: Calendar, color: "text-blue-600", bg: "bg-blue-50" },
  message: { icon: MessageSquare, color: "text-green-600", bg: "bg-green-50" },
  report: { icon: FileText, color: "text-purple-600", bg: "bg-purple-50" },
  alert: { icon: AlertCircle, color: "text-red-500", bg: "bg-red-50" },
  system: { icon: Bell, color: "text-gray-500", bg: "bg-gray-100" },
};

const initialNotifications = [
  { id: "1", type: "appointment" as const, title: "Appointment Reminder", message: "Robert Chen has an appointment in 30 minutes (11:00 AM)", time: "2026-06-09T10:30:00Z", read: false },
  { id: "2", type: "message" as const, title: "New Message", message: "Patient Maria Garcia sent you a message regarding her medication side effects", time: "2026-06-09T09:15:00Z", read: false },
  { id: "3", type: "report" as const, title: "Lab Results Available", message: "Blood test results for James Wilson are now available for review", time: "2026-06-09T08:00:00Z", read: false },
  { id: "4", type: "alert" as const, title: "Critical Patient Alert", message: "Michael Brown's blood pressure reading is critically elevated (185/110)", time: "2026-06-09T07:30:00Z", read: false },
  { id: "5", type: "appointment" as const, title: "Appointment Confirmed", message: "Emily Davis confirmed her 1:00 PM appointment for today", time: "2026-06-08T22:00:00Z", read: true },
  { id: "6", type: "system" as const, title: "System Update", message: "MediCare Pro has been updated to version 2.4.0 with new prescription features", time: "2026-06-08T18:00:00Z", read: true },
  { id: "7", type: "report" as const, title: "Monthly Report Ready", message: "Your May 2026 consultation report has been generated and is ready for review", time: "2026-06-08T15:00:00Z", read: true },
  { id: "8", type: "message" as const, title: "New Message", message: "Sarah Lee is asking about refilling her prescription", time: "2026-06-08T11:20:00Z", read: true },
];

export default function DoctorNotificationsPage() {
  const [notifications, setNotifications] = useState(initialNotifications);

  const unreadCount = notifications.filter((n) => !n.read).length;

  function markRead(id: string) {
    setNotifications((prev) => prev.map((n) => n.id === id ? { ...n, read: true } : n));
  }

  function markAllRead() {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  }

  function dismiss(id: string) {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }

  return (
    <div className="space-y-5 max-w-2xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Notifications</h2>
          <p className="text-sm text-gray-500">{unreadCount} unread</p>
        </div>
        {unreadCount > 0 && (
          <Button variant="outline" size="sm" onClick={markAllRead} className="gap-1.5 text-xs">
            <CheckCircle className="h-3.5 w-3.5" /> Mark all read
          </Button>
        )}
      </div>

      {unreadCount > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-600 font-medium">Unread ({unreadCount})</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {notifications.filter((n) => !n.read).map((notif) => {
              const cfg = notifTypes[notif.type];
              const Icon = cfg.icon;
              return (
                <div key={notif.id} className="flex items-start gap-3 px-5 py-3.5 hover:bg-blue-50/30 transition-colors group border-b border-gray-50 last:border-0">
                  <div className={cn("p-2 rounded-lg shrink-0", cfg.bg)}>
                    <Icon className={cn("h-4 w-4", cfg.color)} />
                  </div>
                  <div className="flex-1 min-w-0" onClick={() => markRead(notif.id)}>
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-sm font-semibold text-gray-900">{notif.title}</p>
                      <div className="flex items-center gap-1 shrink-0">
                        <div className="h-2 w-2 rounded-full bg-blue-500" />
                        <button onClick={(e) => { e.stopPropagation(); dismiss(notif.id); }} className="opacity-0 group-hover:opacity-100 text-gray-300 hover:text-gray-500 transition-all">
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mt-0.5">{notif.message}</p>
                    <p className="text-xs text-gray-400 mt-1">{formatDateTime(notif.time)}</p>
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-gray-600 font-medium">Earlier</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {notifications.filter((n) => n.read).map((notif) => {
            const cfg = notifTypes[notif.type];
            const Icon = cfg.icon;
            return (
              <div key={notif.id} className="flex items-start gap-3 px-5 py-3.5 hover:bg-gray-50 transition-colors group border-b border-gray-50 last:border-0 opacity-70">
                <div className={cn("p-2 rounded-lg shrink-0", cfg.bg)}>
                  <Icon className={cn("h-4 w-4", cfg.color)} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-sm font-medium text-gray-700">{notif.title}</p>
                    <button onClick={() => dismiss(notif.id)} className="opacity-0 group-hover:opacity-100 text-gray-300 hover:text-gray-500 transition-all shrink-0">
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <p className="text-sm text-gray-500 mt-0.5">{notif.message}</p>
                  <p className="text-xs text-gray-400 mt-1">{formatDateTime(notif.time)}</p>
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>
    </div>
  );
}
