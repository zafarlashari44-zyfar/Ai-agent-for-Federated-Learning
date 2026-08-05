"use client";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell,
} from "recharts";

const revenueData = [
  { month: "Jan", revenue: 31200, expenses: 22000, profit: 9200 },
  { month: "Feb", revenue: 33800, expenses: 23500, profit: 10300 },
  { month: "Mar", revenue: 38500, expenses: 25000, profit: 13500 },
  { month: "Apr", revenue: 35200, expenses: 24000, profit: 11200 },
  { month: "May", revenue: 40100, expenses: 26000, profit: 14100 },
  { month: "Jun", revenue: 42800, expenses: 27500, profit: 15300 },
];

const patientGrowth = [
  { month: "Jan", new: 34, churned: 8, total: 1190 },
  { month: "Feb", new: 42, churned: 6, total: 1226 },
  { month: "Mar", new: 51, churned: 9, total: 1268 },
  { month: "Apr", new: 38, churned: 7, total: 1299 },
  { month: "May", new: 56, churned: 11, total: 1344 },
  { month: "Jun", new: 47, churned: 8, total: 1384 },
];

const departmentRevenue = [
  { dept: "Cardiology", revenue: 12400, patients: 192 },
  { dept: "Internal Med", revenue: 10800, patients: 248 },
  { dept: "Pediatrics", revenue: 8200, patients: 167 },
  { dept: "Neurology", revenue: 7600, patients: 134 },
  { dept: "Dermatology", revenue: 5100, patients: 98 },
  { dept: "Orthopedics", revenue: 9800, patients: 145 },
];

const appointmentTypes = [
  { name: "Consultation", value: 38 },
  { name: "Follow-up", value: 29 },
  { name: "Check-up", value: 18 },
  { name: "Emergency", value: 9 },
  { name: "Other", value: 6 },
];

const COLORS = ["#9333ea", "#3b82f6", "#10b981", "#f59e0b", "#6b7280"];

const kpis = [
  { label: "Avg Revenue per Patient", value: "$30.9", change: "+8%", pos: true },
  { label: "Patient Lifetime Value", value: "$1,240", change: "+12%", pos: true },
  { label: "Appointment Utilization", value: "87%", change: "+3%", pos: true },
  { label: "No-Show Rate", value: "4.8%", change: "-1.2%", pos: true },
  { label: "Patient Satisfaction", value: "4.7/5", change: "+0.2", pos: true },
  { label: "Staff-to-Patient Ratio", value: "1:53", change: "-2", pos: true },
];

export default function AdminAnalyticsPage() {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold text-gray-900">System Analytics</h2>
        <p className="text-sm text-gray-500">Platform-wide performance and business intelligence</p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {kpis.map(({ label, value, change, pos }) => (
          <Card key={label}>
            <CardContent className="p-4">
              <p className="text-xs text-gray-500 mb-1">{label}</p>
              <p className="text-lg font-bold text-gray-900">{value}</p>
              <Badge variant={pos ? "success" : "destructive"} className="text-[10px] mt-1">{change}</Badge>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Revenue vs Expenses */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Revenue vs Expenses</CardTitle>
            <CardDescription>Last 6 months financial overview</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={230}>
              <AreaChart data={revenueData}>
                <defs>
                  <linearGradient id="revGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#9333ea" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#9333ea" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="expGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.1} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`} />
                <Tooltip formatter={(v) => [`$${Number(v).toLocaleString()}`, ""]} contentStyle={{ borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <Area type="monotone" dataKey="revenue" stroke="#9333ea" fill="url(#revGrad)" strokeWidth={2} name="Revenue" />
                <Area type="monotone" dataKey="expenses" stroke="#ef4444" fill="url(#expGrad)" strokeWidth={2} name="Expenses" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Patient Growth */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Patient Growth</CardTitle>
            <CardDescription>New vs churned patients</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={patientGrowth}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <Bar dataKey="new" name="New" fill="#10b981" radius={[3, 3, 0, 0]} />
                <Bar dataKey="churned" name="Churned" fill="#fca5a5" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Department Revenue */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Revenue by Department</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {departmentRevenue.sort((a, b) => b.revenue - a.revenue).map((dept) => (
              <div key={dept.dept}>
                <div className="flex items-center justify-between mb-1">
                  <p className="text-sm text-gray-700">{dept.dept}</p>
                  <div className="flex items-center gap-3 text-sm">
                    <span className="text-gray-400 text-xs">{dept.patients} pts</span>
                    <span className="font-semibold text-gray-900">${dept.revenue.toLocaleString()}</span>
                  </div>
                </div>
                <Progress value={(dept.revenue / 12400) * 100} className="h-1.5" />
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Appointment Types */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Appointment Type Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-center mb-4">
              <PieChart width={170} height={170}>
                <Pie data={appointmentTypes} cx={82} cy={82} innerRadius={50} outerRadius={78} paddingAngle={3} dataKey="value">
                  {appointmentTypes.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                </Pie>
              </PieChart>
            </div>
            <div className="space-y-2">
              {appointmentTypes.map((item, i) => (
                <div key={item.name} className="flex items-center gap-2.5">
                  <div className="h-2.5 w-2.5 rounded-full shrink-0" style={{ background: COLORS[i] }} />
                  <span className="text-xs text-gray-600 flex-1">{item.name}</span>
                  <Progress value={item.value} className="w-20 h-1.5" />
                  <span className="text-xs font-medium text-gray-700 w-8 text-right">{item.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
