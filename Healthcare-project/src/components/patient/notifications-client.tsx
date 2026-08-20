"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  Bell,
  Calendar,
  CheckCircle,
  FileText,
  X,
} from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import type { Database } from "@/types/database";
import { cn, formatDateTime } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

type NotificationRow = Database["public"]["Tables"]["notifications"]["Row"];

const notificationTypes = {
  appointment: { icon: Calendar, color: "text-blue-600", bg: "bg-blue-50" },
  report: { icon: FileText, color: "text-purple-600", bg: "bg-purple-50" },
  alert: { icon: AlertCircle, color: "text-red-500", bg: "bg-red-50" },
  reminder: { icon: Bell, color: "text-amber-600", bg: "bg-amber-50" },
  info: { icon: Bell, color: "text-blue-600", bg: "bg-blue-50" },
};

export function NotificationsClient({
  notifications: initialNotifications,
}: {
  notifications: NotificationRow[];
}) {
  const router = useRouter();
  const [notifications, setNotifications] = useState(initialNotifications);
  const [error, setError] = useState("");
  const unread = notifications.filter((notification) => !notification.is_read).length;

  async function markRead(ids: string[]) {
    if (!ids.length) return;
    const now = new Date().toISOString();
    const supabase = createClient();
    const { error: updateError } = await supabase
      .from("notifications")
      .update({ is_read: true, read_at: now })
      .in("id", ids);
    if (updateError) {
      setError(updateError.message);
      return;
    }
    setNotifications((current) =>
      current.map((item) =>
        ids.includes(item.id) ? { ...item, is_read: true, read_at: now } : item,
      ),
    );
    router.refresh();
  }

  async function removeNotification(id: string) {
    const supabase = createClient();
    const { error: deleteError } = await supabase
      .from("notifications")
      .delete()
      .eq("id", id);
    if (deleteError) {
      setError(deleteError.message);
      return;
    }
    setNotifications((current) => current.filter((item) => item.id !== id));
    router.refresh();
  }

  return (
    <div className="space-y-5 max-w-2xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Notifications</h2>
          <p className="text-sm text-gray-500">{unread} unread</p>
        </div>
        {unread > 0 && (
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              markRead(
                notifications
                  .filter((notification) => !notification.is_read)
                  .map((notification) => notification.id),
              )
            }
          >
            <CheckCircle className="h-3.5 w-3.5 mr-1.5" /> Mark all read
          </Button>
        )}
      </div>

      {error && <p className="rounded-lg bg-red-50 p-3 text-sm text-red-600">{error}</p>}

      {notifications.length ? (
        <div className="space-y-3">
          {notifications.map((notification) => {
            const config =
              notificationTypes[
                notification.type as keyof typeof notificationTypes
              ] ?? notificationTypes.info;
            const Icon = config.icon;
            return (
              <Card
                key={notification.id}
                className={cn(
                  !notification.is_read && "border-green-100 bg-green-50/20",
                )}
              >
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <div className={cn("p-2 rounded-lg", config.bg)}>
                      <Icon className={cn("h-4 w-4", config.color)} />
                    </div>
                    <button
                      className="flex-1 min-w-0 text-left"
                      onClick={() => markRead([notification.id])}
                    >
                      <div className="flex items-center gap-2 mb-0.5">
                        <p className="text-sm font-semibold text-gray-900">
                          {notification.title}
                        </p>
                        {!notification.is_read && (
                          <div className="h-1.5 w-1.5 rounded-full bg-green-500" />
                        )}
                      </div>
                      <p className="text-sm text-gray-600">{notification.message}</p>
                      <p className="text-xs text-gray-400 mt-1">
                        {formatDateTime(notification.created_at)}
                      </p>
                    </button>
                    <button
                      onClick={() => removeNotification(notification.id)}
                      className="text-gray-300 hover:text-gray-500"
                      aria-label="Delete notification"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      ) : (
        <Card>
          <CardContent className="p-10 text-center text-sm text-gray-500">
            You have no notifications.
          </CardContent>
        </Card>
      )}
    </div>
  );
}
