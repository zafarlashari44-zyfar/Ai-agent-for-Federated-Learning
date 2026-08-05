import Link from "next/link";
import { Calendar, ClipboardList, FileText, Users } from "lucide-react";

import { createAdminClient } from "@/lib/supabase/admin";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Profile = {
  full_name: string | null;
  email: string | null;
  role: string | null;
};

type RecentPatient = {
  id: string;
  created_at: string | null;
  profiles: Profile | Profile[] | null;
};

function firstItem<T>(value: T | T[] | null | undefined): T | null {
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

export default async function DoctorDashboardPage() {
  const supabase = createAdminClient();

  const [
    patientsResult,
    appointmentsResult,
    prescriptionsResult,
    recentPatientsResult,
  ] = await Promise.all([
    supabase.from("patients").select("*", { count: "exact", head: true }),
    supabase.from("appointments").select("*", { count: "exact", head: true }),
    supabase.from("prescriptions").select("*", { count: "exact", head: true }),
    supabase
      .from("patients")
      .select(
        `
        id,
        created_at,
        profiles (
          full_name,
          email,
          role
        )
      `
      )
      .order("created_at", { ascending: false })
      .limit(5),
  ]);

  if (patientsResult.error) {
    throw new Error(`Unable to load patients count: ${patientsResult.error.message}`);
  }

  if (appointmentsResult.error) {
    throw new Error(
      `Unable to load appointments count: ${appointmentsResult.error.message}`
    );
  }

  if (prescriptionsResult.error) {
    throw new Error(
      `Unable to load prescriptions count: ${prescriptionsResult.error.message}`
    );
  }

  if (recentPatientsResult.error) {
    throw new Error(
      `Unable to load recent patients: ${recentPatientsResult.error.message}`
    );
  }

  const recentPatients = (recentPatientsResult.data ?? []) as RecentPatient[];

  const realRecentPatients = recentPatients.filter((patient) => {
    const profile = firstItem(patient.profiles);
    return profile?.role === "patient";
  });

  const stats = [
    {
      label: "Total Patients",
      value: patientsResult.count ?? 0,
      icon: Users,
      iconClassName: "text-blue-600",
    },
    {
      label: "Appointments",
      value: appointmentsResult.count ?? 0,
      icon: Calendar,
      iconClassName: "text-purple-600",
    },
    {
      label: "Prescriptions",
      value: prescriptionsResult.count ?? 0,
      icon: ClipboardList,
      iconClassName: "text-green-600",
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Doctor Dashboard</h2>
        <p className="text-sm text-gray-500">Live overview from Supabase.</p>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {stats.map((stat) => {
          const Icon = stat.icon;

          return (
            <Card key={stat.label}>
              <CardContent className="flex items-center gap-4 p-5">
                <Icon className={`h-8 w-8 ${stat.iconClassName}`} />
                <div>
                  <p className="text-sm text-gray-500">{stat.label}</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {stat.value}
                  </p>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card>
        <CardContent className="p-5">
          <div className="mb-4 flex items-center justify-between gap-3">
            <h3 className="font-semibold text-gray-900">Recent Patients</h3>
            <Link
              href="/doctor/patients"
              className="text-sm font-medium text-blue-600 hover:text-blue-700"
            >
              View all
            </Link>
          </div>

          {realRecentPatients.length === 0 ? (
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <FileText className="h-4 w-4" />
              No recent patients found.
            </div>
          ) : (
            <div className="space-y-3">
              {realRecentPatients.map((patient) => {
                const profile = firstItem(patient.profiles);
                const patientName = profile?.full_name ?? "Unknown Patient";

                return (
                  <Link
                    key={patient.id}
                    href={`/doctor/patients/${patient.id}`}
                    className="flex items-center justify-between gap-4 rounded-lg border border-gray-100 p-3 transition hover:bg-gray-50"
                  >
                    <div className="min-w-0">
                      <p className="truncate font-medium text-gray-900">
                        {patientName}
                      </p>
                      <p className="truncate text-xs text-gray-400">
                        {profile?.email ?? "No email"}
                      </p>
                    </div>

                    <Badge variant="success">patient</Badge>
                  </Link>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}