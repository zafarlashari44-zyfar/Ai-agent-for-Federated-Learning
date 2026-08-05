"use client";
import { StatCard } from "@/components/ui/stat-card";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Progress } from "@/components/ui/progress";
import {
  Users, UserCheck, Calendar, CreditCard, TrendingUp, Activity,
  ArrowRight, AlertTriangle, CheckCircle,
} from "lucide-react";
import Link from "next/link";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getInitials, formatDate } from "@/lib/utils";

const stats = [
  { title: "Total Doctors", value: "24", change: "+2 this month", changeType: "up" as const, icon: UserCheck, iconColor: "text-purple-600", iconBg: "bg-purple-50" },
  { title: "Total Patients", value: "1,284", change: "+47 this month", changeType: "up" as const, icon: Users, iconColor: "text-blue-600", iconBg: "bg-blue-50" },
  { title: "Today's Appointments", value: "86", change: "12 pending", changeType: "neutral" as const, icon: Calendar, iconColor: "text-green-600", iconBg: "bg-green-50" },
  { title: "Monthly Revenue", value: "$42,800", change: "+12.4% vs last month", changeType: "up" as const, icon: CreditCard, iconColor: "text-amber-600", iconBg: "bg-amber-50" },
];

const revenueData = [
  { month: "Jan", revenue: 31200, target: 35000 },
  { month: "Feb", revenue: 33800, target: 35000 },
  { month: "Mar", revenue: 38500, target: 38000 },
  { month: "Apr", revenue: 35200, target: 38000 },
  { month: "May", revenue: 40100, target: 40000 },
  { month: "Jun", revenue: 42800, target: 42000 },
];

const recentActivities = [
  { user: "Dr. Sarah Johnson", action: "Added new patient James Wilson", time: "2 min ago", type: "patient" },
  { user: "System", action: "Monthly billing report generated", time: "15 min ago", type: "system" },
  { user: "Admin", action: "Dr. Michael Lee account activated", time: "1 hour ago", type: "doctor" },
  { user: "Dr. Chen", action: "Updated prescription for Maria Garcia", time: "2 hours ago", type: "prescription" },
  { user: "System", action: "Backup completed successfully", time: "3 hours ago", type: "system" },
];

const pendingTasks = [
  { task: "Review 3 doctor license renewals", priority: "high", due: "2026-06-12" },
  { task: "Process 7 pending insurance claims", priority: "medium", due: "2026-06-15" },
  { task: "Update appointment availability for new wing", priority: "low", due: "2026-06-20" },
  { task: "Audit access logs for compliance", priority: "high", due: "2026-06-11" },
];

const topDoctors = [
  { name: "Dr. Sarah Johnson", specialty: "Internal Medicine", patients: 248, rating: 4.9 },
  { name: "Dr. Michael Chen", specialty: "Cardiology", patients: 192, rating: 4.8 },
  { name: "Dr. Emily Rodriguez", specialty: "Pediatrics", patients: 167, rating: 4.9 },
  { name: "Dr. James Williams", specialty: "Neurology", patients: 134, rating: 4.7 },
];

export default function AdminOverviewPage() {
  return (
    <div className="space-y-6 max-w-full">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Admin Overview</h2>
          <p className="text-sm text-gray-500">{formatDate(new Date())} — System is operating normally</p>
        </div>
        <div className="flex items-center gap-2 bg-green-50 text-green-700 rounded-lg px-3 py-1.5 text-sm">
          <CheckCircle className="h-4 w-4" />
          All systems operational
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {stats.map((stat) => <StatCard key={stat.title} {...stat} />)}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Revenue Chart */}
        <Card className="lg:col-span-2">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">Revenue Overview</CardTitle>
                <CardDescription>Monthly revenue vs target</CardDescription>
              </div>
              <Link href="/admin/billing">
                <Button variant="ghost" size="sm" className="gap-1 text-xs text-purple-600">Details <ArrowRight className="h-3 w-3" /></Button>
              </Link>
            </div>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={revenueData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`} />
                <Tooltip formatter={(v) => [`$${Number(v).toLocaleString()}`, ""]} contentStyle={{ borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <Bar dataKey="target" name="Target" fill="#e9d5ff" radius={[3, 3, 0, 0]} />
                <Bar dataKey="revenue" name="Revenue" fill="#9333ea" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* System Health */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Activity className="h-4 w-4 text-purple-600" /> System Health
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {[
              { label: "Server Uptime", value: 99.8, status: "Excellent" },
              { label: "Database Performance", value: 94, status: "Good" },
              { label: "Storage Used", value: 62, status: "Normal" },
              { label: "API Response Time", value: 88, status: "Good" },
            ].map(({ label, value, status }) => (
              <div key={label}>
                <div className="flex items-center justify-between mb-1">
                  <p className="text-xs text-gray-600">{label}</p>
                  <span className="text-xs font-medium text-gray-700">{value}%</span>
                </div>
                <Progress value={value} className="h-1.5" />
                <p className="text-[10px] text-gray-400 mt-0.5">{status}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Top Doctors */}
        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between pb-3">
            <CardTitle className="text-base">Top Performing Doctors</CardTitle>
            <Link href="/admin/doctors">
              <Button variant="ghost" size="sm" className="gap-1 text-xs text-purple-600">View all <ArrowRight className="h-3 w-3" /></Button>
            </Link>
          </CardHeader>
          <CardContent className="p-0">
            {topDoctors.map((doc, i) => (
              <div key={doc.name} className="flex items-center gap-4 px-5 py-3 hover:bg-gray-50 transition-colors border-b border-gray-50 last:border-0">
                <span className="text-sm font-bold text-gray-300 w-5">{i + 1}</span>
                <Avatar className="h-9 w-9">
                  <AvatarFallback className="bg-purple-100 text-purple-700 text-sm">{getInitials(doc.name)}</AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900">{doc.name}</p>
                  <p className="text-xs text-gray-400">{doc.specialty}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium text-gray-700">{doc.patients} patients</p>
                  <p className="text-xs text-amber-500">★ {doc.rating}</p>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Pending Tasks */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-500" /> Pending Tasks
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {pendingTasks.map((task, i) => (
              <div key={i} className="px-5 py-3 border-b border-gray-50 last:border-0">
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm text-gray-700">{task.task}</p>
                  <Badge
                    variant={task.priority === "high" ? "destructive" : task.priority === "medium" ? "warning" : "secondary"}
                    className="text-[10px] shrink-0"
                  >
                    {task.priority}
                  </Badge>
                </div>
                <p className="text-xs text-gray-400 mt-0.5">Due: {formatDate(task.due)}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-3">
          <CardTitle className="text-base">Recent Activity</CardTitle>
          <Link href="/admin/logs">
            <Button variant="ghost" size="sm" className="gap-1 text-xs text-purple-600">View logs <ArrowRight className="h-3 w-3" /></Button>
          </Link>
        </CardHeader>
        <CardContent className="p-0">
          <div className="divide-y divide-gray-50">
            {recentActivities.map((activity, i) => (
              <div key={i} className="flex items-center gap-3 px-5 py-3">
                <Avatar className="h-7 w-7 shrink-0">
                  <AvatarFallback className="text-xs bg-purple-100 text-purple-700">{getInitials(activity.user)}</AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <span className="text-sm font-medium text-gray-900">{activity.user} </span>
                  <span className="text-sm text-gray-500">{activity.action}</span>
                </div>
                <span className="text-xs text-gray-400 shrink-0">{activity.time}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
