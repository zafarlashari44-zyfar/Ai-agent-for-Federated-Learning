"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  Droplets,
  Heart,
  Plus,
  Target,
  Thermometer,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { createClient } from "@/lib/supabase/client";
import type { Database } from "@/types/database";
import { formatDate } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type VitalReading = Database["public"]["Tables"]["vital_readings"]["Row"];
type HealthGoal = Database["public"]["Tables"]["health_goals"]["Row"];

function numberOrNull(value: string) {
  return value.trim() ? Number(value) : null;
}

function goalProgress(goal: HealthGoal) {
  if (
    goal.current_value === null ||
    goal.target_value === null ||
    goal.target_value === 0
  ) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(
    (Number(goal.current_value) / Number(goal.target_value)) * 100,
  )));
}

export function HealthClient({
  patientId,
  userId,
  readings,
  goals,
}: {
  patientId: string;
  userId: string;
  readings: VitalReading[];
  goals: HealthGoal[];
}) {
  const router = useRouter();
  const [readingOpen, setReadingOpen] = useState(false);
  const [goalOpen, setGoalOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [reading, setReading] = useState({
    systolic: "",
    diastolic: "",
    heartRate: "",
    temperature: "",
    glucose: "",
    oxygen: "",
    weight: "",
  });
  const [goal, setGoal] = useState({
    title: "",
    currentValue: "",
    targetValue: "",
    unit: "",
    targetDate: "",
    status: "in_progress" as Database["public"]["Enums"]["health_goal_status"],
  });

  const latest = readings[0];
  const chronological = useMemo(() => [...readings].reverse(), [readings]);
  const bloodPressureData = chronological
    .filter((item) => item.blood_pressure_systolic && item.blood_pressure_diastolic)
    .map((item) => ({
      date: formatDate(item.recorded_at),
      systolic: item.blood_pressure_systolic,
      diastolic: item.blood_pressure_diastolic,
    }));
  const weightData = chronological
    .filter((item) => item.weight_kg !== null)
    .map((item) => ({
      date: formatDate(item.recorded_at),
      weight: Math.round(Number(item.weight_kg) * 2.20462 * 10) / 10,
    }));

  const latestVitals = latest
    ? [
        latest.blood_pressure_systolic && latest.blood_pressure_diastolic
          ? {
              label: "Blood Pressure",
              value: `${latest.blood_pressure_systolic}/${latest.blood_pressure_diastolic} mmHg`,
              icon: Heart,
              color: "text-red-500",
              bg: "bg-red-50",
            }
          : null,
        latest.heart_rate
          ? {
              label: "Heart Rate",
              value: `${latest.heart_rate} bpm`,
              icon: Activity,
              color: "text-blue-600",
              bg: "bg-blue-50",
            }
          : null,
        latest.temperature_celsius
          ? {
              label: "Temperature",
              value: `${Math.round((Number(latest.temperature_celsius) * 9 / 5 + 32) * 10) / 10} °F`,
              icon: Thermometer,
              color: "text-amber-600",
              bg: "bg-amber-50",
            }
          : null,
        latest.blood_glucose_mg_dl
          ? {
              label: "Blood Glucose",
              value: `${latest.blood_glucose_mg_dl} mg/dL`,
              icon: Droplets,
              color: "text-green-600",
              bg: "bg-green-50",
            }
          : null,
      ].filter((item): item is NonNullable<typeof item> => item !== null)
    : [];

  async function saveReading() {
    const values = Object.values(reading);
    if (!values.some((value) => value.trim())) {
      setMessage("Enter at least one vital measurement.");
      return;
    }

    setSaving(true);
    setMessage("");
    const supabase = createClient();
    const { error } = await supabase.from("vital_readings").insert({
      patient_id: patientId,
      recorded_by: userId,
      blood_pressure_systolic: numberOrNull(reading.systolic),
      blood_pressure_diastolic: numberOrNull(reading.diastolic),
      heart_rate: numberOrNull(reading.heartRate),
      temperature_celsius: numberOrNull(reading.temperature),
      blood_glucose_mg_dl: numberOrNull(reading.glucose),
      oxygen_saturation: numberOrNull(reading.oxygen),
      weight_kg: numberOrNull(reading.weight),
    });
    setSaving(false);

    if (error) {
      setMessage(error.message);
      return;
    }

    setReadingOpen(false);
    setReading({
      systolic: "",
      diastolic: "",
      heartRate: "",
      temperature: "",
      glucose: "",
      oxygen: "",
      weight: "",
    });
    router.refresh();
  }

  async function saveGoal() {
    if (!goal.title.trim()) {
      setMessage("Enter a title for the health goal.");
      return;
    }

    setSaving(true);
    setMessage("");
    const supabase = createClient();
    const { error } = await supabase.from("health_goals").insert({
      patient_id: patientId,
      created_by: userId,
      title: goal.title.trim(),
      current_value: numberOrNull(goal.currentValue),
      target_value: numberOrNull(goal.targetValue),
      unit: goal.unit.trim() || null,
      target_date: goal.targetDate || null,
      status: goal.status,
    });
    setSaving(false);

    if (error) {
      setMessage(error.message);
      return;
    }

    setGoalOpen(false);
    setGoal({
      title: "",
      currentValue: "",
      targetValue: "",
      unit: "",
      targetDate: "",
      status: "in_progress",
    });
    router.refresh();
  }

  return (
    <div className="space-y-5 max-w-4xl">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-gray-900">My Health</h2>
          <p className="text-sm text-gray-500">Live health metrics and goals</p>
        </div>
        <div className="flex gap-2">
          <Dialog open={readingOpen} onOpenChange={setReadingOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm">
                <Plus className="h-4 w-4" /> Add Reading
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader><DialogTitle>Add Vital Reading</DialogTitle></DialogHeader>
              <div className="grid grid-cols-2 gap-4">
                {[
                  ["systolic", "Systolic BP"],
                  ["diastolic", "Diastolic BP"],
                  ["heartRate", "Heart rate"],
                  ["temperature", "Temperature (°C)"],
                  ["glucose", "Blood glucose (mg/dL)"],
                  ["oxygen", "Oxygen saturation (%)"],
                  ["weight", "Weight (kg)"],
                ].map(([key, label]) => (
                  <div key={key} className="space-y-1.5">
                    <Label>{label}</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={reading[key as keyof typeof reading]}
                      onChange={(event) =>
                        setReading((current) => ({
                          ...current,
                          [key]: event.target.value,
                        }))
                      }
                    />
                  </div>
                ))}
              </div>
              {message && <p className="text-sm text-red-600">{message}</p>}
              <DialogFooter>
                <Button variant="outline" onClick={() => setReadingOpen(false)}>Cancel</Button>
                <Button onClick={saveReading} disabled={saving} className="bg-green-600 hover:bg-green-700">
                  {saving ? "Saving..." : "Save Reading"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <Dialog open={goalOpen} onOpenChange={setGoalOpen}>
            <DialogTrigger asChild>
              <Button size="sm" className="bg-green-600 hover:bg-green-700">
                <Plus className="h-4 w-4" /> Add Goal
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader><DialogTitle>Add Health Goal</DialogTitle></DialogHeader>
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <Label>Goal title</Label>
                  <Input
                    value={goal.title}
                    onChange={(event) => setGoal({ ...goal, title: event.target.value })}
                    placeholder="Walk 10,000 steps daily"
                  />
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div className="space-y-1.5">
                    <Label>Current</Label>
                    <Input type="number" value={goal.currentValue} onChange={(event) => setGoal({ ...goal, currentValue: event.target.value })} />
                  </div>
                  <div className="space-y-1.5">
                    <Label>Target</Label>
                    <Input type="number" value={goal.targetValue} onChange={(event) => setGoal({ ...goal, targetValue: event.target.value })} />
                  </div>
                  <div className="space-y-1.5">
                    <Label>Unit</Label>
                    <Input value={goal.unit} onChange={(event) => setGoal({ ...goal, unit: event.target.value })} />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <Label>Target date</Label>
                    <Input type="date" value={goal.targetDate} onChange={(event) => setGoal({ ...goal, targetDate: event.target.value })} />
                  </div>
                  <div className="space-y-1.5">
                    <Label>Status</Label>
                    <Select value={goal.status} onValueChange={(value: typeof goal.status) => setGoal({ ...goal, status: value })}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="in_progress">In progress</SelectItem>
                        <SelectItem value="on_track">On track</SelectItem>
                        <SelectItem value="paused">Paused</SelectItem>
                        <SelectItem value="completed">Completed</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
              {message && <p className="text-sm text-red-600">{message}</p>}
              <DialogFooter>
                <Button variant="outline" onClick={() => setGoalOpen(false)}>Cancel</Button>
                <Button onClick={saveGoal} disabled={saving} className="bg-green-600 hover:bg-green-700">
                  {saving ? "Saving..." : "Save Goal"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {latestVitals.length ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {latestVitals.map(({ label, value, icon: Icon, color, bg }) => (
            <Card key={label}>
              <CardContent className="p-4">
                <div className={`p-2 rounded-lg w-fit mb-3 ${bg}`}>
                  <Icon className={`h-4 w-4 ${color}`} />
                </div>
                <p className="text-sm font-bold text-gray-900">{value}</p>
                <p className="text-xs text-gray-500 mt-0.5">{label}</p>
                <Badge variant="success" className="text-[10px] mt-2">
                  {formatDate(latest.recorded_at)}
                </Badge>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="p-8 text-center text-sm text-gray-500">
            No vital readings yet. Add your first reading to start tracking.
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Heart className="h-4 w-4 text-red-500" /> Blood Pressure Trend
            </CardTitle>
            <CardDescription>{bloodPressureData.length} recorded readings</CardDescription>
          </CardHeader>
          <CardContent>
            {bloodPressureData.length ? (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={bloodPressureData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="systolic" stroke="#ef4444" strokeWidth={2} />
                  <Line type="monotone" dataKey="diastolic" stroke="#f97316" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="h-[200px] flex items-center justify-center text-sm text-gray-500">
                No blood pressure history.
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Weight Trend</CardTitle>
            <CardDescription>{weightData.length} recorded readings</CardDescription>
          </CardHeader>
          <CardContent>
            {weightData.length ? (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={weightData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="weight" stroke="#10b981" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="h-[200px] flex items-center justify-center text-sm text-gray-500">
                No weight history.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="h-4 w-4 text-green-600" /> Health Goals
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          {goals.length ? goals.map((item) => {
            const progress = goalProgress(item);
            return (
              <div key={item.id}>
                <div className="flex items-center justify-between mb-1.5">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{item.title}</p>
                    <p className="text-xs text-gray-400">
                      {item.current_value ?? "—"} / {item.target_value ?? "—"} {item.unit || ""}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-700">{progress}%</span>
                    <Badge variant={item.status === "on_track" || item.status === "completed" ? "success" : "warning"}>
                      {item.status.replaceAll("_", " ")}
                    </Badge>
                  </div>
                </div>
                <Progress value={progress} className="h-2" />
              </div>
            );
          }) : (
            <p className="py-6 text-center text-sm text-gray-500">
              No health goals yet.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
