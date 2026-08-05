import { Calendar, CheckCircle, Clock, XCircle } from "lucide-react";

import { createAdminClient } from "@/lib/supabase/admin";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { getInitials, formatDate } from "@/lib/utils";

const statusVariant = {
  scheduled: "secondary",
  confirmed: "info",
  completed: "success",
  cancelled: "destructive",
} as const;

type AppointmentStatus = keyof typeof statusVariant;

type Profile = {
  full_name: string | null;
  email: string | null;
  role: string | null;
};

type Patient = {
  id: string;
  profiles: Profile | Profile[] | null;
};

type Appointment = {
  id: string;
  scheduled_at: string | null;
  duration_minutes: number | null;
  status: string | null;
  type: string | null;
  reason: string | null;
  notes: string | null;
  patients: Patient | Patient[] | null;
};

function firstItem<T>(value: T | T[] | null | undefined): T | null {
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function getStatus(status: string | null): AppointmentStatus {
  if (status === "confirmed") return "confirmed";
  if (status === "completed") return "completed";
  if (status === "cancelled") return "cancelled";
  return "scheduled";
}

function getTime(value: string | null): string {
  if (!value) return "Not set";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return "Invalid time";

  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function isToday(value: string | null): boolean {
  if (!value) return false;

  const date = new Date(value);
  const today = new Date();

  return date.toDateString() === today.toDateString();
}

export default async function DoctorAppointmentsPage() {
  const supabase = createAdminClient();

  const { data, error } = await supabase
    .from("appointments")
    .select(
      `
      id,
      scheduled_at,
      duration_minutes,
      status,
      type,
      reason,
      notes,
      patients (
        id,
        profiles (
          full_name,
          email,
          role
        )
      )
    `
    )
    .order("scheduled_at", { ascending: true });

  if (error) {
    throw new Error(`Unable to load appointments: ${error.message}`);
  }

  const allAppointments = (data ?? []) as Appointment[];

  const appointments = allAppointments.filter((appointment) => {
    const patient = firstItem(appointment.patients);
    const profile = firstItem(patient?.profiles);

    return profile?.role === "patient";
  });

  const todayAppointments = appointments.filter((appointment) =>
    isToday(appointment.scheduled_at)
  );

  const confirmedAppointments = appointments.filter(
    (appointment) => getStatus(appointment.status) === "confirmed"
  );

  const completedAppointments = appointments.filter(
    (appointment) => getStatus(appointment.status) === "completed"
  );

  const cancelledAppointments = appointments.filter(
    (appointment) => getStatus(appointment.status) === "cancelled"
  );

  const stats = [
    {
      label: "Total",
      value: appointments.length,
      icon: Calendar,
    },
    {
      label: "Today",
      value: todayAppointments.length,
      icon: Clock,
    },
    {
      label: "Confirmed",
      value: confirmedAppointments.length,
      icon: CheckCircle,
    },
    {
      label: "Cancelled",
      value: cancelledAppointments.length,
      icon: XCircle,
    },
  ];

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Appointments</h2>
        <p className="text-sm text-gray-500">
          {appointments.length} real appointments
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon;

          return (
            <Card key={stat.label}>
              <CardContent className="flex items-center gap-4 p-5">
                <Icon className="h-7 w-7 text-blue-600" />
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

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Today</CardTitle>
          </CardHeader>

          <CardContent>
            <p className="text-3xl font-bold text-gray-900">
              {todayAppointments.length}
            </p>
            <p className="text-sm text-gray-500">appointments today</p>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Calendar className="h-4 w-4 text-blue-600" />
              Today&apos;s Schedule
            </CardTitle>
          </CardHeader>

          <CardContent className="p-0">
            {todayAppointments.length === 0 ? (
              <div className="py-10 text-center text-sm text-gray-400">
                No appointments for today.
              </div>
            ) : (
              <div className="divide-y divide-gray-50">
                {todayAppointments.map((appointment) => {
                  const patient = firstItem(appointment.patients);
                  const profile = firstItem(patient?.profiles);
                  const name = profile?.full_name ?? "Unknown Patient";
                  const status = getStatus(appointment.status);

                  return (
                    <div
                      key={appointment.id}
                      className="flex items-center gap-4 px-5 py-3"
                    >
                      <Clock className="h-4 w-4 shrink-0 text-gray-400" />

                      <span className="w-20 text-sm font-medium text-gray-900">
                        {getTime(appointment.scheduled_at)}
                      </span>

                      <Avatar className="h-8 w-8 shrink-0">
                        <AvatarFallback className="bg-blue-100 text-xs text-blue-700">
                          {getInitials(name)}
                        </AvatarFallback>
                      </Avatar>

                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium text-gray-900">
                          {name}
                        </p>
                        <p className="truncate text-xs text-gray-400">
                          {appointment.type ?? "General"} ·{" "}
                          {appointment.duration_minutes ?? 0} min
                        </p>
                      </div>

                      <Badge variant={statusVariant[status]}>{status}</Badge>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            All Appointments ({appointments.length})
          </CardTitle>
        </CardHeader>

        {appointments.length === 0 ? (
          <CardContent className="p-6 text-sm text-gray-500">
            No real appointments found.
          </CardContent>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-100 bg-gray-50/50">
                  <th className="px-5 py-3 text-left text-xs font-semibold uppercase text-gray-500">
                    Patient
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-gray-500">
                    Date & Time
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-gray-500">
                    Type
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-gray-500">
                    Reason
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-gray-500">
                    Status
                  </th>
                </tr>
              </thead>

              <tbody className="divide-y divide-gray-50">
                {appointments.map((appointment) => {
                  const patient = firstItem(appointment.patients);
                  const profile = firstItem(patient?.profiles);
                  const name = profile?.full_name ?? "Unknown Patient";
                  const status = getStatus(appointment.status);

                  return (
                    <tr key={appointment.id} className="hover:bg-gray-50">
                      <td className="px-5 py-3">
                        <div className="flex items-center gap-2.5">
                          <Avatar className="h-7 w-7">
                            <AvatarFallback className="bg-blue-100 text-xs text-blue-700">
                              {getInitials(name)}
                            </AvatarFallback>
                          </Avatar>

                          <div className="min-w-0">
                            <p className="truncate text-sm font-medium text-gray-900">
                              {name}
                            </p>
                            <p className="truncate text-xs text-gray-400">
                              {profile?.email ?? "No email"}
                            </p>
                          </div>
                        </div>
                      </td>

                      <td className="px-4 py-3 text-sm text-gray-600">
                        {appointment.scheduled_at
                          ? `${formatDate(appointment.scheduled_at)} · ${getTime(
                              appointment.scheduled_at
                            )}`
                          : "Not set"}
                      </td>

                      <td className="px-4 py-3 text-sm text-gray-600">
                        {appointment.type ?? "General"}
                      </td>

                      <td className="max-w-xs px-4 py-3 text-sm text-gray-600">
                        <p className="truncate">
                          {appointment.reason ?? "No reason provided"}
                        </p>
                      </td>

                      <td className="px-4 py-3">
                        <Badge variant={statusVariant[status]}>{status}</Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}