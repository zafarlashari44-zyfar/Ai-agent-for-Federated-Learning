"use client";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar, Clock, Plus, Search, ChevronLeft, ChevronRight } from "lucide-react";
import { getInitials, formatDate } from "@/lib/utils";

const appointments = Array.from({ length: 35 }, (_, i) => ({
  id: String(i + 1),
  patient: ["James Wilson", "Maria Garcia", "Robert Chen", "Emily Davis", "Michael Brown", "Sarah Lee", "David Kim", "Anna White"][i % 8],
  type: ["Consultation", "Follow-up", "Check-up", "Lab Review", "Prescription", "Emergency"][i % 6],
  date: new Date(2026, 5, 1 + Math.floor(i / 4)).toISOString().split("T")[0],
  time: `${String(8 + (i % 9)).padStart(2, "0")}:${i % 2 === 0 ? "00" : "30"}`,
  duration: [15, 30, 45, 60][i % 4],
  status: ["scheduled", "confirmed", "completed", "cancelled", "in_progress"][i % 5] as "scheduled" | "confirmed" | "completed" | "cancelled" | "in_progress",
  notes: i % 3 === 0 ? "Patient requested early morning slot" : "",
}));

const statusVariant = {
  scheduled: "secondary", confirmed: "info", completed: "success",
  cancelled: "destructive", in_progress: "warning",
} as const;

export default function DoctorAppointmentsPage() {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [viewDate, setViewDate] = useState(new Date(2026, 5, 9));

  const filtered = appointments.filter((a) => {
    const matchSearch = a.patient.toLowerCase().includes(search.toLowerCase()) || a.type.toLowerCase().includes(search.toLowerCase());
    const matchStatus = statusFilter === "all" || a.status === statusFilter;
    return matchSearch && matchStatus;
  });

  const dateStr = viewDate.toISOString().split("T")[0];
  const todayAppts = filtered.filter((a) => a.date === dateStr);

  const days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(viewDate);
    d.setDate(d.getDate() - d.getDay() + i);
    return d;
  });

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Appointments</h2>
          <p className="text-sm text-gray-500">{filtered.length} total appointments</p>
        </div>
        <Button size="sm" className="gap-2">
          <Plus className="h-4 w-4" /> New Appointment
        </Button>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4 flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
            <Input placeholder="Search appointments…" className="pl-8" value={search} onChange={(e) => setSearch(e.target.value)} />
          </div>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="scheduled">Scheduled</SelectItem>
              <SelectItem value="confirmed">Confirmed</SelectItem>
              <SelectItem value="in_progress">In Progress</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="cancelled">Cancelled</SelectItem>
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Calendar Week */}
        <Card className="lg:col-span-1">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Weekly View</CardTitle>
              <div className="flex gap-1">
                <Button variant="ghost" size="icon" className="h-7 w-7"
                  onClick={() => { const d = new Date(viewDate); d.setDate(d.getDate() - 7); setViewDate(d); }}>
                  <ChevronLeft className="h-3.5 w-3.5" />
                </Button>
                <Button variant="ghost" size="icon" className="h-7 w-7"
                  onClick={() => { const d = new Date(viewDate); d.setDate(d.getDate() + 7); setViewDate(d); }}>
                  <ChevronRight className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pb-4">
            <div className="grid grid-cols-7 gap-1 mb-3">
              {["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"].map((d) => (
                <div key={d} className="text-center text-xs font-medium text-gray-400 py-1">{d}</div>
              ))}
              {days.map((d) => {
                const ds = d.toISOString().split("T")[0];
                const count = appointments.filter((a) => a.date === ds).length;
                const isSelected = ds === dateStr;
                const isToday = ds === new Date(2026, 5, 9).toISOString().split("T")[0];
                return (
                  <button
                    key={ds}
                    onClick={() => setViewDate(d)}
                    className={`flex flex-col items-center py-2 rounded-lg text-xs font-medium transition-colors ${isSelected ? "bg-blue-600 text-white" : isToday ? "bg-blue-50 text-blue-700" : "text-gray-600 hover:bg-gray-50"}`}
                  >
                    <span>{d.getDate()}</span>
                    {count > 0 && (
                      <span className={`mt-0.5 h-1 w-1 rounded-full ${isSelected ? "bg-white" : "bg-blue-400"}`} />
                    )}
                  </button>
                );
              })}
            </div>
            <p className="text-xs text-gray-500 text-center">{formatDate(viewDate)}</p>
          </CardContent>
        </Card>

        {/* Day Schedule */}
        <Card className="lg:col-span-2">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Calendar className="h-4 w-4 text-blue-600" />
              {formatDate(viewDate)} — {todayAppts.length} appointments
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {todayAppts.length === 0 ? (
              <div className="text-center py-12 text-gray-400">
                <Calendar className="h-10 w-10 mx-auto mb-3 opacity-30" />
                <p className="text-sm">No appointments for this day</p>
              </div>
            ) : (
              <div className="divide-y divide-gray-50">
                {todayAppts.map((appt) => (
                  <div key={appt.id} className="flex items-center gap-4 px-5 py-3 hover:bg-gray-50 transition-colors">
                    <div className="flex items-center gap-1.5 w-16 shrink-0">
                      <Clock className="h-3.5 w-3.5 text-gray-400" />
                      <span className="text-sm font-medium text-gray-600">{appt.time}</span>
                    </div>
                    <div className="w-px h-10 bg-gray-100" />
                    <Avatar className="h-8 w-8 shrink-0">
                      <AvatarFallback className="text-xs bg-blue-100 text-blue-700">{getInitials(appt.patient)}</AvatarFallback>
                    </Avatar>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900">{appt.patient}</p>
                      <p className="text-xs text-gray-400">{appt.type} · {appt.duration}min</p>
                    </div>
                    <Badge variant={statusVariant[appt.status]}>{appt.status.replace("_", " ")}</Badge>
                    <Button variant="ghost" size="sm" className="text-xs shrink-0">Manage</Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* All appointments list */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">All Appointments ({filtered.length})</CardTitle>
        </CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50/50">
                <th className="text-left py-3 px-5 text-xs font-semibold text-gray-500 uppercase">Patient</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden md:table-cell">Date & Time</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden sm:table-cell">Type</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Status</th>
                <th className="text-right py-3 px-5 text-xs font-semibold text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {filtered.slice(0, 20).map((appt) => (
                <tr key={appt.id} className="hover:bg-gray-50 transition-colors">
                  <td className="py-3 px-5">
                    <div className="flex items-center gap-2.5">
                      <Avatar className="h-7 w-7">
                        <AvatarFallback className="text-xs bg-blue-100 text-blue-700">{getInitials(appt.patient)}</AvatarFallback>
                      </Avatar>
                      <span className="text-sm font-medium text-gray-900">{appt.patient}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 hidden md:table-cell text-sm text-gray-600">
                    {formatDate(appt.date)} · {appt.time}
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
