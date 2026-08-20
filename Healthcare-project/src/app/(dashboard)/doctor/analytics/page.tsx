"use client";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, Radar } from "recharts";

const weeklyData = [
  { day: "Mon", patients: 12, completed: 11 },
  { day: "Tue", patients: 15, completed: 14 },
  { day: "Wed", patients: 9, completed: 8 },
  { day: "Thu", patients: 14, completed: 13 },
  { day: "Fri", patients: 11, completed: 11 },
  { day: "Sat", patients: 6, completed: 5 },
  { day: "Sun", patients: 0, completed: 0 },
];

const responseTimeData = [
  { month: "Jan", avg: 22 }, { month: "Feb", avg: 20 }, { month: "Mar", avg: 18 },
  { month: "Apr", avg: 19 }, { month: "May", avg: 17 }, { month: "Jun", avg: 15 },
];

const radarData = [
  { subject: "Diagnosis Accuracy", value: 94 },
  { subject: "Patient Satisfaction", value: 88 },
  { subject: "Appointment Punctuality", value: 92 },
  { subject: "Prescription Adherence", value: 79 },
  { subject: "Follow-up Rate", value: 85 },
  { subject: "Documentation", value: 96 },
];

const topConditions = [
  { name: "Hypertension", count: 87, trend: "+5%" },
  { name: "Type 2 Diabetes", count: 54, trend: "+2%" },
  { name: "Asthma", count: 38, trend: "-1%" },
  { name: "Arthritis", count: 31, trend: "+8%" },
  { name: "Anemia", count: 24, trend: "+3%" },
];

export default function DoctorAnalyticsPage() {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Analytics</h2>
        <p className="text-sm text-gray-500">Your clinical performance insights</p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: "Patient Retention", value: "91%", change: "+3%", positive: true },
          { label: "Avg Consultation Time", value: "24 min", change: "-2 min", positive: true },
          { label: "No-Show Rate", value: "4.2%", change: "-0.8%", positive: true },
          { label: "Prescription Accuracy", value: "99.1%", change: "+0.3%", positive: true },
        ].map(({ label, value, change, positive }) => (
          <Card key={label}>
            <CardContent className="p-4">
              <p className="text-xs text-gray-500 mb-1">{label}</p>
              <p className="text-2xl font-bold text-gray-900">{value}</p>
              <Badge variant={positive ? "success" : "destructive"} className="text-xs mt-1">{change}</Badge>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Weekly Consultations */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Weekly Consultations</CardTitle>
            <CardDescription>Current week patient load</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={weeklyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="day" tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <Bar dataKey="patients" name="Scheduled" fill="#bfdbfe" radius={[3, 3, 0, 0]} />
                <Bar dataKey="completed" name="Completed" fill="#3b82f6" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Response Time */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Avg. Response Time</CardTitle>
            <CardDescription>Minutes to respond to patient messages</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={responseTimeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <Line type="monotone" dataKey="avg" stroke="#10b981" strokeWidth={2.5} dot={{ fill: "#10b981", r: 4 }} name="Minutes" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Performance Radar */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Performance Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={220}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: "#9ca3af" }} />
                <Radar name="Performance" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.12} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Top Conditions */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Most Treated Conditions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {topConditions.map((cond, i) => (
                <div key={cond.name} className="flex items-center gap-3">
                  <span className="text-xs font-semibold text-gray-400 w-4">{i + 1}</span>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <p className="text-sm text-gray-700">{cond.name}</p>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-gray-900">{cond.count}</span>
                        <Badge variant={cond.trend.startsWith("+") ? "success" : "destructive"} className="text-[10px]">{cond.trend}</Badge>
                      </div>
                    </div>
                    <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 rounded-full"
                        style={{ width: `${(cond.count / topConditions[0].count) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
