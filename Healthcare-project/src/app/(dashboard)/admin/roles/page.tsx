"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Check,
  Loader2,
  Lock,
  Shield,
  Stethoscope,
  User,
  X,
} from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import type { Database } from "@/types/database";
import { getInitials } from "@/lib/utils";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

type UserRole = Database["public"]["Enums"]["user_role"];
type RoleRequestRow = Database["public"]["Tables"]["role_requests"]["Row"];
type RoleRequest = RoleRequestRow & {
  profiles: {
    email: string;
    full_name: string;
  } | null;
};

const roleDefinitions: Array<{
  id: UserRole;
  label: string;
  description: string;
  color: string;
  icon: typeof Shield;
  permissions: string[];
}> = [
  {
    id: "admin",
    label: "Administrator",
    description: "Full system, user, billing, and configuration access.",
    color: "bg-purple-600",
    icon: Shield,
    permissions: [
      "Manage users and roles",
      "Manage doctors and patients",
      "View system analytics",
      "Manage billing and settings",
    ],
  },
  {
    id: "doctor",
    label: "Doctor",
    description: "Clinical access for assigned patients and care workflows.",
    color: "bg-blue-600",
    icon: Stethoscope,
    permissions: [
      "View assigned patients",
      "Manage appointments",
      "Create records and prescriptions",
      "Message assigned patients",
    ],
  },
  {
    id: "patient",
    label: "Patient",
    description: "Personal access to health records and care services.",
    color: "bg-green-600",
    icon: User,
    permissions: [
      "View own health records",
      "Book and cancel appointments",
      "Manage personal health data",
      "Message assigned doctor",
    ],
  },
];

export default function AdminRolesPage() {
  const router = useRouter();
  const [selectedRole, setSelectedRole] = useState<UserRole>("admin");
  const [requests, setRequests] = useState<RoleRequest[]>([]);
  const [admins, setAdmins] = useState<
    Array<{ full_name: string; email: string }>
  >([]);
  const [counts, setCounts] = useState<Record<UserRole, number>>({
    admin: 0,
    doctor: 0,
    patient: 0,
  });
  const [loading, setLoading] = useState(true);
  const [reviewingId, setReviewingId] = useState("");
  const [error, setError] = useState("");

  const loadAccessData = useCallback(async (showLoader = true) => {
    if (showLoader) {
      setLoading(true);
    }
    setError("");
    const supabase = createClient();
    const [requestResult, profileResult, adminResult] = await Promise.all([
      supabase
        .from("role_requests")
        .select(
          "*, profiles!role_requests_user_id_fkey(full_name, email)",
        )
        .eq("status", "pending")
        .order("created_at", { ascending: false }),
      supabase.from("profiles").select("role"),
      supabase
        .from("profiles")
        .select("full_name, email")
        .eq("role", "admin")
        .order("full_name"),
    ]);

    const firstError =
      requestResult.error || profileResult.error || adminResult.error;
    if (firstError) {
      setError(firstError.message);
      if (showLoader) {
        setLoading(false);
      }
      return;
    }

    const nextCounts: Record<UserRole, number> = {
      admin: 0,
      doctor: 0,
      patient: 0,
    };
    profileResult.data?.forEach((profile) => {
      nextCounts[profile.role] += 1;
    });

    setRequests((requestResult.data ?? []) as RoleRequest[]);
    setAdmins(adminResult.data ?? []);
    setCounts(nextCounts);
    if (showLoader) {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadAccessData();
    }, 0);
    const interval = window.setInterval(() => {
      void loadAccessData(false);
    }, 10000);

    return () => {
      window.clearTimeout(timer);
      window.clearInterval(interval);
    };
  }, [loadAccessData]);

  async function reviewRequest(request: RoleRequest, approve: boolean) {
    let note: string | null = null;

    if (!approve) {
      note = window.prompt("Reason for rejection (optional):")?.trim() || null;
      if (note === null && !window.confirm("Reject without a note?")) {
        return;
      }
    }

    setReviewingId(request.id);
    setError("");
    const supabase = createClient();
    const { error: reviewError } = await supabase.rpc(
      "admin_review_role_request",
      {
        target_request_id: request.id,
        approve_request: approve,
        review_message: note ?? undefined,
      },
    );
    setReviewingId("");

    if (reviewError) {
      setError(reviewError.message);
      return;
    }

    await loadAccessData();
    router.refresh();
  }

  const activeDefinition =
    roleDefinitions.find((role) => role.id === selectedRole) ??
    roleDefinitions[0];

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold text-gray-900">
          Roles & Access Control
        </h2>
        <p className="text-sm text-gray-500">
          Review signup requests and approved access levels
        </p>
      </div>

      {error && (
        <p className="rounded-lg bg-red-50 p-3 text-sm text-red-600">
          {error}
        </p>
      )}

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Pending Access Requests</CardTitle>
          <CardDescription>
            Doctor and administrator accounts remain patients until approved.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center gap-2 py-6 text-sm text-gray-500">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading access requests...
            </div>
          ) : requests.length ? (
            <div className="divide-y divide-gray-100">
              {requests.map((request) => (
                <div
                  key={request.id}
                  className="flex flex-wrap items-center gap-3 py-4 first:pt-0 last:pb-0"
                >
                  <Avatar className="h-9 w-9">
                    <AvatarFallback className="bg-purple-100 text-xs text-purple-700">
                      {getInitials(request.profiles?.full_name || "User")}
                    </AvatarFallback>
                  </Avatar>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-gray-900">
                      {request.profiles?.full_name || "Unknown user"}
                    </p>
                    <p className="text-xs text-gray-400">
                      {request.profiles?.email}
                    </p>
                    {request.requested_role === "doctor" && (
                      <p className="mt-1 text-xs text-gray-500">
                        {request.specialization} - {request.license_number}
                      </p>
                    )}
                  </div>
                  <Badge
                    variant={
                      request.requested_role === "admin" ? "purple" : "info"
                    }
                  >
                    {request.requested_role}
                  </Badge>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => reviewRequest(request, false)}
                    disabled={reviewingId === request.id}
                  >
                    <X className="h-3.5 w-3.5" /> Reject
                  </Button>
                  <Button
                    size="sm"
                    className="bg-purple-600 hover:bg-purple-700"
                    onClick={() => reviewRequest(request, true)}
                    disabled={reviewingId === request.id}
                  >
                    {reviewingId === request.id ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Check className="h-3.5 w-3.5" />
                    )}
                    Approve
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <p className="py-4 text-sm text-gray-500">
              No pending doctor or administrator requests.
            </p>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <div className="space-y-3">
          {roleDefinitions.map((role) => {
            const Icon = role.icon;
            const selected = selectedRole === role.id;
            return (
              <button
                key={role.id}
                onClick={() => setSelectedRole(role.id)}
                className={`w-full rounded-xl border p-4 text-left transition-all ${
                  selected
                    ? "border-purple-300 bg-purple-50 shadow-sm"
                    : "border-gray-100 bg-white hover:border-gray-200"
                }`}
              >
                <div className="mb-2 flex items-center gap-3">
                  <div
                    className={`flex h-8 w-8 items-center justify-center rounded-lg ${role.color}`}
                  >
                    <Icon className="h-4 w-4 text-white" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-gray-900">
                      {role.label}
                    </p>
                    <p className="text-xs text-gray-400">
                      {counts[role.id]} users
                    </p>
                  </div>
                </div>
                <p className="text-xs text-gray-500">{role.description}</p>
              </button>
            );
          })}
        </div>

        <div className="space-y-4 lg:col-span-2">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">
                {activeDefinition.label} Permissions
              </CardTitle>
              <CardDescription>
                {activeDefinition.description}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {activeDefinition.permissions.map((permission) => (
                <div
                  key={permission}
                  className="flex items-center gap-2 border-b border-gray-50 py-2 last:border-0"
                >
                  <Lock className="h-3.5 w-3.5 text-gray-400" />
                  <span className="text-sm text-gray-700">{permission}</span>
                </div>
              ))}
            </CardContent>
          </Card>

          {selectedRole === "admin" && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">
                  Active Administrators
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                {admins.length ? (
                  admins.map((admin) => (
                    <div
                      key={admin.email}
                      className="flex items-center gap-3 border-b border-gray-50 px-5 py-3 last:border-0"
                    >
                      <Avatar className="h-8 w-8">
                        <AvatarFallback className="bg-purple-100 text-xs text-purple-700">
                          {getInitials(admin.full_name)}
                        </AvatarFallback>
                      </Avatar>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium text-gray-900">
                          {admin.full_name}
                        </p>
                        <p className="text-xs text-gray-400">{admin.email}</p>
                      </div>
                      <Badge variant="success">Active</Badge>
                    </div>
                  ))
                ) : (
                  <p className="px-5 py-4 text-sm text-gray-500">
                    No administrator accounts found.
                  </p>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
