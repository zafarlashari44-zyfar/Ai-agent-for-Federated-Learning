"use client";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { StatCard } from "@/components/ui/stat-card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Download, CreditCard, TrendingUp, AlertCircle, CheckCircle } from "lucide-react";
import { formatDate } from "@/lib/utils";

const invoices = Array.from({ length: 30 }, (_, i) => ({
  id: `INV-2026-${String(1000 + i).padStart(4, "0")}`,
  patient: ["James Wilson", "Maria Garcia", "Robert Chen", "Emily Davis", "Michael Brown"][i % 5],
  doctor: ["Dr. Sarah Johnson", "Dr. Michael Chen", "Dr. Emily Rodriguez"][i % 3],
  amount: 80 + (i * 37) % 320,
  status: ["paid", "paid", "pending", "paid", "overdue", "paid"][i % 6] as "paid" | "pending" | "overdue" | "cancelled",
  date: new Date(2026, 5, Math.max(1, 9 - Math.floor(i / 4))).toISOString().split("T")[0],
  dueDate: new Date(2026, 5, Math.min(30, 19 - Math.floor(i / 4))).toISOString().split("T")[0],
  service: ["Consultation", "Lab Tests", "Follow-up", "Prescription", "Check-up"][i % 5],
}));

const statusVariant = { paid: "success", pending: "warning", overdue: "destructive", cancelled: "secondary" } as const;

export default function AdminBillingPage() {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  const filtered = invoices.filter((inv) => {
    const matchSearch = inv.patient.toLowerCase().includes(search.toLowerCase()) || inv.id.toLowerCase().includes(search.toLowerCase());
    const matchStatus = statusFilter === "all" || inv.status === statusFilter;
    return matchSearch && matchStatus;
  });

  const totalRevenue = invoices.filter((i) => i.status === "paid").reduce((sum, i) => sum + i.amount, 0);
  const pendingAmount = invoices.filter((i) => i.status === "pending").reduce((sum, i) => sum + i.amount, 0);
  const overdueAmount = invoices.filter((i) => i.status === "overdue").reduce((sum, i) => sum + i.amount, 0);

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Billing & Payments</h2>
          <p className="text-sm text-gray-500">{filtered.length} invoices</p>
        </div>
        <Button variant="outline" size="sm" className="gap-2"><Download className="h-4 w-4" /> Export CSV</Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Total Collected" value={`$${totalRevenue.toLocaleString()}`} icon={CheckCircle} iconColor="text-green-600" iconBg="bg-green-50" change="This month" changeType="neutral" />
        <StatCard title="Pending" value={`$${pendingAmount.toLocaleString()}`} icon={CreditCard} iconColor="text-amber-600" iconBg="bg-amber-50" change={`${invoices.filter((i) => i.status === "pending").length} invoices`} changeType="neutral" />
        <StatCard title="Overdue" value={`$${overdueAmount.toLocaleString()}`} icon={AlertCircle} iconColor="text-red-500" iconBg="bg-red-50" change="Needs attention" changeType="down" />
        <StatCard title="Total Invoices" value={String(invoices.length)} icon={TrendingUp} iconColor="text-purple-600" iconBg="bg-purple-50" change="All time" changeType="neutral" />
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
          <Input placeholder="Search invoices…" className="pl-8 bg-white" value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-36"><SelectValue placeholder="Status" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="paid">Paid</SelectItem>
            <SelectItem value="pending">Pending</SelectItem>
            <SelectItem value="overdue">Overdue</SelectItem>
            <SelectItem value="cancelled">Cancelled</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Table */}
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50/50">
                <th className="text-left py-3 px-5 text-xs font-semibold text-gray-500 uppercase">Invoice</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden sm:table-cell">Patient</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden lg:table-cell">Service</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden md:table-cell">Date</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden xl:table-cell">Due Date</th>
                <th className="text-right py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Amount</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Status</th>
                <th className="text-right py-3 px-5 text-xs font-semibold text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {filtered.map((inv) => (
                <tr key={inv.id} className="hover:bg-gray-50/50 transition-colors">
                  <td className="py-3 px-5">
                    <p className="text-sm font-mono font-medium text-gray-900">{inv.id}</p>
                    <p className="text-xs text-gray-400">{inv.doctor}</p>
                  </td>
                  <td className="py-3 px-4 hidden sm:table-cell text-sm text-gray-700">{inv.patient}</td>
                  <td className="py-3 px-4 hidden lg:table-cell text-sm text-gray-600">{inv.service}</td>
                  <td className="py-3 px-4 hidden md:table-cell text-sm text-gray-500">{formatDate(inv.date)}</td>
                  <td className="py-3 px-4 hidden xl:table-cell text-sm text-gray-500">{formatDate(inv.dueDate)}</td>
                  <td className="py-3 px-4 text-right">
                    <span className="text-sm font-semibold text-gray-900">${inv.amount}</span>
                  </td>
                  <td className="py-3 px-4">
                    <Badge variant={statusVariant[inv.status]}>{inv.status}</Badge>
                  </td>
                  <td className="py-3 px-5 text-right">
                    <Button variant="ghost" size="sm" className="text-xs h-7">
                      {inv.status === "pending" || inv.status === "overdue" ? "Collect" : "View"}
                    </Button>
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
