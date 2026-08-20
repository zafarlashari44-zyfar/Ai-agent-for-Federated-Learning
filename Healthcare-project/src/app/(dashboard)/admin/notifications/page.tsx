"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  Bell,
  CheckCircle,
  CreditCard,
  FileText,
  Loader2,
  Settings,
  Users,
  X,
} from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import type { Database } from "@/types/database";
import { cn, formatDateTime } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

type NotificationRow = Database["public"]["Tables"]["notifications"]["Row"];

const notificationTypes = {
  user: { icon: Users, color: "text-blue-600", bg: "bg-blue-50" },
  billing: {
    icon: CreditCard,
    color: "text-amber-600",
    bg: "bg-amber-50",
  },
  alert: { icon: AlertCircle, color: "text-red-500", bg: "bg-red-50" },
  report: { icon: FileText, color: "text-purple-600", bg: "bg-purple-50" },
  system: { icon: Settings, color: "text-purple-600", bg: "bg-purple-50" },
  info: { icon: Bell, color: "text-blue-600", bg: "bg-blue-50" },
};

export default function AdminNotificationsPage() {
  const router = useRouter();
  const [notifications, setNotifications] = useState<NotificationRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const unread = notifications.filter(
    (notification) => !notification.is_read,
  ).length;

  const loadNotifications = useCallback(async (showLoader = true) => {
    if (showLoader) {
      setLoading(true);
    }
    setError("");

    const supabase = createClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      setError("Authentication required.");
      setLoading(false);
      return;
    }

    const { data, error: loadError } = await supabase
      .from("notifications")
      .select("*")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false });

    if (loadError) {
      setError(loadError.message);
    } else {
      setNotifications(data ?? []);
    }

    if (showLoader) {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadNotifications();
    }, 0);
    const interval = window.setInterval(() => {
      void loadNotifications(false);
    }, 10000);

    return () => {
      window.clearTimeout(timer);
      window.clearInterval(interval);
    };
  }, [loadNotifications]);

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
        ids.includes(item.id)
          ? { ...item, is_read: true, read_at: now }
          : item,
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

    setNotifications((current) =>
      current.filter((notification) => notification.id !== id),
    );
    router.refresh();
  }

  async function openNotification(notification: NotificationRow) {
    if (!notification.is_read) {
      await markRead([notification.id]);
    }
    if (notification.link) {
      router.push(notification.link);
    }
  }

  return (
    <div className="max-w-2xl space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">
            Admin Notifications
          </h2>
          <p className="text-sm text-gray-500">{unread} unread alerts</p>
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
            <CheckCircle className="mr-1.5 h-3.5 w-3.5" /> Mark all read
          </Button>
        )}
      </div>

      {error && (
        <p className="rounded-lg bg-red-50 p-3 text-sm text-red-600">
          {error}
        </p>
      )}

      {loading ? (
        <div className="flex items-center gap-2 py-8 text-sm text-gray-500">
          <Loader2 className="h-4 w-4 animate-spin" />
          Loading notifications...
        </div>
      ) : notifications.length ? (
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
                  !notification.is_read && "border-blue-100 bg-blue-50/20",
                )}
              >
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <div className={cn("shrink-0 rounded-lg p-2", config.bg)}>
                      <Icon className={cn("h-4 w-4", config.color)} />
                    </div>
                    <button
                      className="min-w-0 flex-1 text-left"
                      onClick={() => openNotification(notification)}
                    >
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-semibold text-gray-900">
                          {notification.title}
                        </p>
                        {!notification.is_read && (
                          <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />
                        )}
                      </div>
                      <p className="mt-0.5 text-sm text-gray-600">
                        {notification.message}
                      </p>
                      <p className="mt-2 text-xs text-gray-400">
                        {formatDateTime(notification.created_at)}
                      </p>
                    </button>
                    <button
                      onClick={() => removeNotification(notification.id)}
                      className="shrink-0 text-gray-300 transition-colors hover:text-gray-500"
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
