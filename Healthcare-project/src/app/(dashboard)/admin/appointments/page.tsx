"use client";
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Download } from "lucide-react";
import { getInitials, formatDate } from "@/lib/utils";
import { StatCard } from "@/components/ui/stat-card";
import { Calendar, CheckCircle, XCircle, Clock } from "lucide-react";

const appointments = Array.from({ length: 40 }, (_, i) => ({
  id: String(i + 1),
  patient: ["James Wilson", "Maria Garcia", "Robert Chen", "Emily Davis", "Michael Brown", "Sarah Lee"][i % 6],
  doctor: ["Dr. Sarah Johnson", "Dr. Michael Chen", "Dr. Emily Rodriguez", "Dr. James Williams"][i % 4],
  date: new Date(2026, 5, 1 + Math.floor(i / 5)).toISOString().split("T")[0],
  time: `${String(8 + (i % 9)).padStart(2, "0")}:${i % 2 === 0 ? "00" : "30"}`,
  type: ["Consultation", "Follow-up", "Check-up", "Lab Review"][i % 4],
  status: ["scheduled", "confirmed", "completed", "cancelled", "no_show"][i % 5] as "scheduled" | "confirmed" | "completed" | "cancelled" | "no_show",
}));

const statusVariant = {
  scheduled: "secondary", confirmed: "info", completed: "success",
  cancelled: "destructive", no_show: "warning",
} as const;

export default function AdminAppointmentsPage() {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  const filtered = appointments.filter((a) => {
    const matchSearch = a.patient.toLowerCase().includes(search.toLowerCase()) || a.doctor.toLowerCase().includes(search.toLowerCase());
    const matchStatus = statusFilter === "all" || a.status === statusFilter;
    return matchSearch && matchStatus;
  });

  const completed = appointments.filter((a) => a.status === "completed").length;
  const cancelled = appointments.filter((a) => a.status === "cancelled").length;
  const scheduled = appointments.filter((a) => ["scheduled", "confirmed"].includes(a.status)).length;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Appointment Monitoring</h2>
          <p className="text-sm text-gray-500">{filtered.length} appointments</p>
        </div>
        <Button variant="outline" size="sm" className="gap-2"><Download className="h-4 w-4" /> Export</Button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Total" value={String(appointments.length)} icon={Calendar} iconColor="text-purple-600" iconBg="bg-purple-50" changeType="neutral" />
        <StatCard title="Scheduled" value={String(scheduled)} icon={Clock} iconColor="text-blue-600" iconBg="bg-blue-50" changeType="neutral" />
        <StatCard title="Completed" value={String(completed)} icon={CheckCircle} iconColor="text-green-600" iconBg="bg-green-50" changeType="neutral" />
        <StatCard title="Cancelled" value={String(cancelled)} icon={XCircle} iconColor="text-red-500" iconBg="bg-red-50" changeType="neutral" />
      </div>

      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
          <Input placeholder="Search by patient or doctor…" className="pl-8 bg-white" value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-40"><SelectValue placeholder="Status" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="scheduled">Scheduled</SelectItem>
            <SelectItem value="confirmed">Confirmed</SelectItem>
            <SelectItem value="completed">Completed</SelectItem>
            <SelectItem value="cancelled">Cancelled</SelectItem>
            <SelectItem value="no_show">No Show</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50/50">
                <th className="text-left py-3 px-5 text-xs font-semibold text-gray-500 uppercase">Patient</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden md:table-cell">Doctor</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden lg:table-cell">Date & Time</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden sm:table-cell">Type</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Status</th>
                <th className="text-right py-3 px-5 text-xs font-semibold text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {filtered.map((appt) => (
                <tr key={appt.id} className="hover:bg-gray-50/50 transition-colors">
                  <td className="py-3 px-5">
                    <div className="flex items-center gap-2.5">
                      <Avatar className="h-7 w-7">
                        <AvatarFallback className="text-xs bg-blue-100 text-blue-700">{getInitials(appt.patient)}</AvatarFallback>
                      </Avatar>
                      <span className="text-sm font-medium text-gray-900">{appt.patient}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 hidden md:table-cell text-sm text-gray-600">{appt.doctor}</td>
                  <td className="py-3 px-4 hidden lg:table-cell">
                    <p className="text-sm text-gray-700">{formatDate(appt.date)}</p>
                    <p className="text-xs text-gray-400">{appt.time}</p>
                  </td>
                  <td className="py-3 px-4 hidden sm:table-cell text-sm text-gray-600">{appt.type}</td>
                  <td className="py-3 px-4">
                    <Badge variant={statusVariant[appt.status]}>{appt.status.replace("_", " ")}</Badge>
                  </td>
                  <td className="py-3 px-5 text-right">
                    <Button variant="ghost" size="sm" className="text-xs h-7">View</Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
