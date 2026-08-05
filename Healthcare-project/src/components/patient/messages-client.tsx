"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Send } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import type { DoctorDirectoryItem } from "@/lib/patient-types";
import type { Database } from "@/types/database";
import { cn, formatDateTime, getInitials } from "@/lib/utils";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

type MessageRow = Database["public"]["Tables"]["messages"]["Row"];

export function MessagesClient({
  userId,
  doctor,
  messages,
}: {
  userId: string;
  doctor: DoctorDirectoryItem | null;
  messages: MessageRow[];
}) {
  const router = useRouter();
  const [newMessage, setNewMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const unreadIds = messages
      .filter((message) => message.receiver_id === userId && !message.is_read)
      .map((message) => message.id);
    if (!unreadIds.length) return;

    const supabase = createClient();
    void supabase
      .from("messages")
      .update({ is_read: true, read_at: new Date().toISOString() })
      .in("id", unreadIds)
      .then(() => router.refresh());
  }, [messages, router, userId]);

  async function sendMessage() {
    if (!doctor || !newMessage.trim()) return;

    setSending(true);
    setError("");
    const supabase = createClient();
    const { error: sendError } = await supabase.from("messages").insert({
      sender_id: userId,
      receiver_id: doctor.profile_id,
      content: newMessage.trim(),
    });
    setSending(false);

    if (sendError) {
      setError(sendError.message);
      return;
    }

    setNewMessage("");
    router.refresh();
  }

  if (!doctor) {
    return (
      <Card>
        <CardContent className="p-10 text-center">
          <h2 className="font-semibold text-gray-900">Messages</h2>
          <p className="text-sm text-gray-500 mt-2">
            No doctor is assigned to your profile yet. An administrator must
            assign a doctor before messaging is available.
          </p>
        </CardContent>
      </Card>
    );
  }

  const doctorDisplayName = `Dr. ${doctor.full_name}`;

  return (
    <div className="flex h-[calc(100vh-8rem)] border border-gray-100 rounded-xl overflow-hidden bg-white">
      <div className="w-72 border-r border-gray-100 flex flex-col shrink-0">
        <div className="p-4 border-b border-gray-100">
          <h2 className="font-semibold text-gray-900">Messages</h2>
          <p className="text-xs text-gray-400 mt-1">Your assigned care provider</p>
        </div>
        <div className="p-4 bg-green-50 flex items-start gap-3">
          <Avatar className="h-9 w-9">
            <AvatarFallback className="bg-green-100 text-green-700 text-sm">
              {getInitials(doctorDisplayName)}
            </AvatarFallback>
          </Avatar>
          <div className="min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">
              {doctorDisplayName}
            </p>
            <p className="text-xs text-gray-500 truncate">{doctor.specialization}</p>
            <p className="text-xs text-gray-400 mt-2">
              {messages.at(-1)?.content || "No messages yet"}
            </p>
          </div>
        </div>
      </div>

      <div className="flex flex-col flex-1 min-w-0">
        <div className="flex items-center gap-3 px-5 py-3.5 border-b border-gray-100">
          <Avatar className="h-8 w-8">
            <AvatarFallback className="bg-green-100 text-green-700 text-sm">
              {getInitials(doctorDisplayName)}
            </AvatarFallback>
          </Avatar>
          <div>
            <p className="font-semibold text-gray-900 text-sm">{doctorDisplayName}</p>
            <p className="text-xs text-gray-400">{doctor.specialization}</p>
          </div>
          <Badge variant="success" className="ml-auto text-xs">Assigned doctor</Badge>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {messages.length ? messages.map((message) => {
            const sentByPatient = message.sender_id === userId;
            return (
              <div
                key={message.id}
                className={cn("flex gap-3", sentByPatient ? "justify-end" : "justify-start")}
              >
                {!sentByPatient && (
                  <Avatar className="h-7 w-7 mt-1">
                    <AvatarFallback className="bg-green-100 text-green-700 text-xs">
                      {getInitials(doctorDisplayName)}
                    </AvatarFallback>
                  </Avatar>
                )}
                <div
                  className={cn(
                    "max-w-xs lg:max-w-md rounded-2xl px-4 py-2.5 text-sm",
                    sentByPatient
                      ? "bg-green-600 text-white rounded-br-sm"
                      : "bg-gray-100 text-gray-900 rounded-bl-sm",
                  )}
                >
                  <p>{message.content}</p>
                  <p className={cn(
                    "text-[10px] mt-1",
                    sentByPatient ? "text-green-200" : "text-gray-400",
                  )}>
                    {formatDateTime(message.created_at)}
                  </p>
                </div>
              </div>
            );
          }) : (
            <p className="h-full flex items-center justify-center text-sm text-gray-500">
              Start a conversation with your assigned doctor.
            </p>
          )}
        </div>

        <div className="p-4 border-t border-gray-100">
          {error && <p className="text-sm text-red-600 mb-2">{error}</p>}
          <div className="flex items-center gap-3">
            <Input
              placeholder="Type a message"
              value={newMessage}
              onChange={(event) => setNewMessage(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void sendMessage();
              }}
              className="flex-1"
            />
            <Button
              onClick={sendMessage}
              disabled={sending || !newMessage.trim()}
              className="bg-green-600 hover:bg-green-700"
              aria-label="Send message"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
