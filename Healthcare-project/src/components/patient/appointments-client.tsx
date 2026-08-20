"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Calendar, Clock, Plus, X } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { formatDate, formatTime, getInitials } from "@/lib/utils";
import type { DoctorDirectoryItem } from "@/lib/patient-types";
import type { Database } from "@/types/database";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

type Appointment = Database["public"]["Tables"]["appointments"]["Row"];

const statusVariant = {
  confirmed: "success",
  scheduled: "info",
  in_progress: "warning",
  completed: "secondary",
  cancelled: "destructive",
  no_show: "destructive",
} as const;

const appointmentTypes = ["Consultation", "Follow-up", "Check-up", "Lab Review"];
const appointmentTimes = [
  "09:00",
  "09:30",
  "10:00",
  "10:30",
  "11:00",
  "13:00",
  "14:00",
  "15:00",
  "16:00",
];

export function AppointmentsClient({
  patientId,
  appointments,
  doctors,
}: {
  patientId: string;
  appointments: Appointment[];
  doctors: DoctorDirectoryItem[];
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [doctorId, setDoctorId] = useState("");
  const [type, setType] = useState("");
  const [date, setDate] = useState("");
  const [time, setTime] = useState("");
  const [reason, setReason] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const doctorsById = new Map(doctors.map((doctor) => [doctor.doctor_id, doctor]));
  const upcoming = appointments.filter((appointment) =>
    ["confirmed", "scheduled", "in_progress"].includes(appointment.status),
  );
  const past = appointments.filter((appointment) =>
    ["completed", "cancelled", "no_show"].includes(appointment.status),
  );

  async function bookAppointment() {
    if (!doctorId || !type || !date || !time) {
      setMessage("Choose a doctor, appointment type, date, and time.");
      return;
    }

    setSaving(true);
    setMessage("");
    const supabase = createClient();
    const scheduledAt = new Date(`${date}T${time}:00`);
    const { error } = await supabase.from("appointments").insert({
      patient_id: patientId,
      doctor_id: doctorId,
      scheduled_at: scheduledAt.toISOString(),
      type,
      reason: reason.trim() || null,
      status: "scheduled",
    });

    setSaving(false);
    if (error) {
      setMessage(
        error.code === "23505"
          ? "That time is no longer available. Please choose another slot."
          : error.message,
      );
      return;
    }

    setOpen(false);
    setDoctorId("");
    setType("");
    setDate("");
    setTime("");
    setReason("");
    router.refresh();
  }

  async function cancelAppointment(id: string) {
    setMessage("");
    const supabase = createClient();
    const { error } = await supabase
      .from("appointments")
      .update({ status: "cancelled" })
      .eq("id", id);

    if (error) {
      setMessage(error.message);
      return;
    }

    router.refresh();
  }

  function doctorLabel(doctorIdValue: string) {
    const doctor = doctorsById.get(doctorIdValue);
    return doctor ? `Dr. ${doctor.full_name}` : "Healthcare provider";
  }

  return (
    <div className="space-y-5 max-w-3xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">My Appointments</h2>
          <p className="text-sm text-gray-500">{upcoming.length} upcoming</p>
        </div>
        <Dialog open={open} onOpenChange={setOpen}>
          <DialogTrigger asChild>
            <Button
              size="sm"
              className="gap-2 bg-green-600 hover:bg-green-700"
              disabled={!doctors.length}
            >
              <Plus className="h-4 w-4" /> Book Appointment
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Book New Appointment</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div className="space-y-1.5">
                <Label>Select Doctor</Label>
                <Select value={doctorId} onValueChange={setDoctorId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose a doctor" />
                  </SelectTrigger>
                  <SelectContent>
                    {doctors.map((doctor) => (
                      <SelectItem key={doctor.doctor_id} value={doctor.doctor_id}>
                        Dr. {doctor.full_name} — {doctor.specialization}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label>Appointment Type</Label>
                <Select value={type} onValueChange={setType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    {appointmentTypes.map((option) => (
                      <SelectItem key={option} value={option}>
                        {option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Preferred Date</Label>
                  <Input
                    type="date"
                    min={new Date().toISOString().split("T")[0]}
                    value={date}
                    onChange={(event) => setDate(event.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Preferred Time</Label>
                  <Select value={time} onValueChange={setTime}>
                    <SelectTrigger>
                      <SelectValue placeholder="Time" />
                    </SelectTrigger>
                    <SelectContent>
                      {appointmentTimes.map((option) => (
                        <SelectItem key={option} value={option}>
                          {formatTime(`2000-01-01T${option}:00`)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="space-y-1.5">
                <Label>Reason for Visit</Label>
                <Textarea
                  placeholder="Briefly describe your reason for the visit"
                  rows={3}
                  value={reason}
                  onChange={(event) => setReason(event.target.value)}
                />
              </div>
              {message && <p className="text-sm text-red-600">{message}</p>}
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setOpen(false)}>
                Cancel
              </Button>
              <Button
                className="bg-green-600 hover:bg-green-700"
                onClick={bookAppointment}
                disabled={saving}
              >
                {saving ? "Requesting..." : "Request Appointment"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {!doctors.length && (
        <Card>
          <CardContent className="p-4 text-sm text-amber-700 bg-amber-50">
            No active doctors are available yet. An administrator must create a
            doctor account before appointments can be booked.
          </CardContent>
        </Card>
      )}

      {message && !open && (
        <p className="rounded-lg bg-red-50 p-3 text-sm text-red-600">{message}</p>
      )}

      <section>
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
          Upcoming
        </h3>
        {upcoming.length ? (
          <div className="space-y-3">
            {upcoming.map((appointment) => {
              const doctor = doctorsById.get(appointment.doctor_id);
              const name = doctorLabel(appointment.doctor_id);
              return (
                <Card key={appointment.id} className="border-green-100">
                  <CardContent className="p-4">
                    <div className="flex items-start gap-4">
                      <Avatar className="h-10 w-10">
                        <AvatarFallback className="bg-green-100 text-green-700 text-sm">
                          {getInitials(name)}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-0.5">
                          <p className="font-semibold text-gray-900">{name}</p>
                          <Badge variant={statusVariant[appointment.status]}>
                            {appointment.status}
                          </Badge>
                        </div>
                        <p className="text-xs text-gray-400 mb-2">
                          {doctor?.specialization || "General care"} · {appointment.type} ·{" "}
                          {appointment.duration_minutes} min
                        </p>
                        <div className="flex items-center gap-4 text-sm text-gray-600">
                          <span className="flex items-center gap-1.5">
                            <Calendar className="h-3.5 w-3.5 text-gray-400" />
                            {formatDate(appointment.scheduled_at)}
                          </span>
                          <span className="flex items-center gap-1.5">
                            <Clock className="h-3.5 w-3.5 text-gray-400" />
                            {formatTime(appointment.scheduled_at)}
                          </span>
                        </div>
                        {appointment.reason && (
                          <p className="text-xs text-gray-500 mt-2">
                            Reason: {appointment.reason}
                          </p>
                        )}
                      </div>
                      {appointment.status !== "in_progress" && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-gray-400 hover:text-red-500"
                          onClick={() => cancelAppointment(appointment.id)}
                          aria-label="Cancel appointment"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        ) : (
          <Card>
            <CardContent className="p-8 text-center text-sm text-gray-500">
              No upcoming appointments.
            </CardContent>
          </Card>
        )}
      </section>

      <section>
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
          Past Appointments
        </h3>
        <Card>
          {past.length ? (
            <div className="divide-y divide-gray-50">
              {past.map((appointment) => (
                <div key={appointment.id} className="flex items-center gap-4 px-5 py-3">
                  <div className="w-24 shrink-0">
                    <p className="text-xs font-medium text-gray-600">
                      {formatDate(appointment.scheduled_at)}
                    </p>
                    <p className="text-xs text-gray-400">
                      {formatTime(appointment.scheduled_at)}
                    </p>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900">
                      {doctorLabel(appointment.doctor_id)}
                    </p>
                    <p className="text-xs text-gray-400">{appointment.type}</p>
                  </div>
                  <Badge variant={statusVariant[appointment.status]}>
                    {appointment.status}
                  </Badge>
                </div>
              ))}
            </div>
          ) : (
            <p className="p-8 text-center text-sm text-gray-500">
              No past appointments.
            </p>
          )}
        </Card>
      </section>
    </div>
  );
}
