import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  Calendar,
  ClipboardList,
  FileText,
  Users,
} from "lucide-react";

import { createAdminClient } from "@/lib/supabase/admin";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Profile = {
  full_name?: string | null;
  email?: string | null;
  role?: string | null;
};

type Patient = {
  id: string;
  profiles: Profile | Profile[] | null;
};

type AiReport = Record<string, unknown>;

function firstItem<T>(value: T | T[] | null | undefined): T | null {
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function getText(value: unknown, fallback = "Not set") {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function getRisk(report: AiReport) {
  const value =
    report.risk_level ??
    report.risk ??
    report.severity ??
    report.priority ??
    "unknown";

  return String(value).toLowerCase();
}

function getPrediction(report: AiReport) {
  return getText(
    report.prediction ??
      report.predicted_class ??
      report.diagnosis ??
      report.output ??
      report.label,
    "AI Report"
  );
}

export default async function DoctorDashboardPage() {
  const supabase = createAdminClient();

  const [
    patientsResult,
    appointmentsResult,
    prescriptionsResult,
    reportsResult,
    recentPatientsResult,
  ] = await Promise.all([
    supabase.from("patients").select("*", { count: "exact", head: true }),
    supabase.from("appointments").select("*", { count: "exact", head: true }),
    supabase.from("prescriptions").select("*", { count: "exact", head: true }),
    supabase
      .from("agent_outputs")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(5),
    supabase
      .from("patients")
      .select(
        `
        id,
        profiles (
          full_name,
          email,
          role
        )
      `
      )
      .limit(5),
  ]);

  if (patientsResult.error) throw new Error(patientsResult.error.message);
  if (appointmentsResult.error) throw new Error(appointmentsResult.error.message);
  if (prescriptionsResult.error) throw new Error(prescriptionsResult.error.message);
  if (reportsResult.error) throw new Error(reportsResult.error.message);
  if (recentPatientsResult.error) throw new Error(recentPatientsResult.error.message);

  const recentPatients = ((recentPatientsResult.data ?? []) as Patient[]).filter(
    (patient) => firstItem(patient.profiles)?.role === "patient"
  );

  const reports = (reportsResult.data ?? []) as AiReport[];

  const criticalReports = reports.filter((report) => {
    const risk = getRisk(report);
    return risk.includes("critical") || risk.includes("high");
  });

  const stats = [
    {
      label: "Total Patients",
      value: patientsResult.count ?? 0,
      icon: Users,
      href: "/doctor/patients",
    },
    {
      label: "Appointments",
      value: appointmentsResult.count ?? 0,
      icon: Calendar,
      href: "/doctor/appointments",
    },
    {
      label: "Prescriptions",
      value: prescriptionsResult.count ?? 0,
      icon: ClipboardList,
      href: "/doctor/prescriptions",
    },
    {
      label: "AI Reports",
      value: reports.length,
      icon: Activity,
      href: "/doctor/reports",
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Doctor Dashboard</h2>
        <p className="text-sm text-gray-500">
          Live hospital overview from Supabase.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon;

          return (
            <Link key={stat.label} href={stat.href}>
              <Card className="transition hover:shadow-sm">
                <CardContent className="flex items-center gap-4 p-5">
                  <Icon className="h-8 w-8 text-blue-600" />
                  <div>
                    <p className="text-sm text-gray-500">{stat.label}</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {stat.value}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardContent className="p-5">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="font-semibold text-gray-900">Recent Patients</h3>
              <Link href="/doctor/patients" className="text-sm text-blue-600">
                View all
              </Link>
            </div>

            <div className="space-y-3">
              {recentPatients.length === 0 ? (
                <p className="text-sm text-gray-500">No recent patients found.</p>
              ) : (
                recentPatients.map((patient) => {
                  const profile = firstItem(patient.profiles);

                  return (
                    <Link
                      key={patient.id}
                      href={`/doctor/patients/${patient.id}`}
                      className="flex items-center justify-between rounded-lg border border-gray-100 p-3 hover:bg-gray-50"
                    >
                      <div>
                        <p className="font-medium text-gray-900">
                          {profile?.full_name ?? "Unknown Patient"}
                        </p>
                        <p className="text-xs text-gray-400">
                          {profile?.email ?? "No email"}
                        </p>
                      </div>

                      <Badge variant="success">patient</Badge>
                    </Link>
                  );
                })
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-5">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="font-semibold text-gray-900">Recent AI Reports</h3>
              <Link href="/doctor/reports" className="text-sm text-blue-600">
                View all
              </Link>
            </div>

            <div className="space-y-3">
              {reports.length === 0 ? (
                <p className="text-sm text-gray-500">No AI reports found.</p>
              ) : (
                reports.map((report, index) => {
                  const risk = getRisk(report);
                  const isCritical =
                    risk.includes("critical") || risk.includes("high");

                  return (
                    <div
                      key={String(report.id ?? index)}
                      className="flex items-center justify-between rounded-lg border border-gray-100 p-3"
                    >
                      <div>
                        <p className="font-medium text-gray-900">
                          {getPrediction(report)}
                        </p>
                        <p className="text-xs text-gray-400">
                          Risk level: {risk}
                        </p>
                      </div>

                      <Badge variant={isCritical ? "destructive" : "secondary"}>
                        {isCritical ? "critical" : "normal"}
                      </Badge>
                    </div>
                  );
                })
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {criticalReports.length > 0 && (
        <Card className="border-red-100 bg-red-50">
          <CardContent className="flex items-center gap-3 p-5">
            <AlertTriangle className="h-5 w-5 text-red-600" />
            <div>
              <p className="font-semibold text-red-700">
                {criticalReports.length} critical AI case found
              </p>
              <p className="text-sm text-red-600">
                Review high-risk ECG/AI reports immediately.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardContent className="flex items-center gap-3 p-5 text-sm text-gray-500">
          <FileText className="h-4 w-4" />
          Dashboard is connected to real Supabase tables. No demo data used.
        </CardContent>
      </Card>
    </div>
  );
}