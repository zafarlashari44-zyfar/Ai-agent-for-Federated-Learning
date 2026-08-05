"use client";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, UserPlus, Filter, Eye, Phone, Mail, MoreHorizontal } from "lucide-react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import Link from "next/link";
import { getInitials, formatDate } from "@/lib/utils";

const allPatients = Array.from({ length: 50 }, (_, i) => ({
  id: String(i + 1),
  name: ["James Wilson", "Maria Garcia", "Robert Chen", "Emily Davis", "Michael Brown", "Sarah Lee", "David Kim", "Anna White", "Thomas Moore", "Jessica Taylor"][i % 10],
  age: 25 + (i * 7) % 45,
  gender: i % 3 === 0 ? "Female" : i % 3 === 1 ? "Male" : "Male",
  phone: `+1 (555) ${String(100 + i).padStart(3, "0")}-${String(1000 + i * 7).slice(0, 4)}`,
  email: `patient${i + 1}@example.com`,
  condition: ["Hypertension", "Type 2 Diabetes", "Asthma", "Arthritis", "Anemia", "Migraine", "Thyroid Disorder", "GERD", "Depression", "Anxiety"][i % 10],
  bloodType: ["A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-"][i % 8],
  lastVisit: new Date(2026, 5, Math.max(1, 9 - (i % 30))).toISOString().split("T")[0],
  nextAppointment: i % 4 === 0 ? new Date(2026, 5, 10 + (i % 20)).toISOString().split("T")[0] : null,
  status: ["stable", "stable", "review", "stable", "critical", "stable", "stable", "review", "stable", "stable"][i % 10] as "stable" | "review" | "critical",
  insuranceProvider: ["Blue Cross", "Aetna", "United Health", "Cigna", "Humana"][i % 5],
}));

const statusVariant = { stable: "success", review: "warning", critical: "destructive" } as const;

const PAGE_SIZE = 15;

export default function DoctorPatientsPage() {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [page, setPage] = useState(1);
  const [viewMode, setViewMode] = useState<"table" | "grid">("table");

  const filtered = allPatients.filter((p) => {
    const matchSearch = p.name.toLowerCase().includes(search.toLowerCase()) ||
      p.condition.toLowerCase().includes(search.toLowerCase()) ||
      p.email.toLowerCase().includes(search.toLowerCase());
    const matchStatus = statusFilter === "all" || p.status === statusFilter;
    return matchSearch && matchStatus;
  });

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Patients</h2>
          <p className="text-sm text-gray-500">{filtered.length} records</p>
        </div>
        <Link href="/doctor/patients/new">
          <Button size="sm" className="gap-2">
            <UserPlus className="h-4 w-4" /> Add Patient
          </Button>
        </Link>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4 flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
            <Input
              placeholder="Search patients by name, condition, or email…"
              className="pl-8"
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(1); }}
            />
          </div>
          <Select value={statusFilter} onValueChange={(v) => { setStatusFilter(v); setPage(1); }}>
            <SelectTrigger className="w-36">
              <Filter className="h-3.5 w-3.5 mr-1 text-gray-400" />
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="stable">Stable</SelectItem>
              <SelectItem value="review">Needs Review</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
            </SelectContent>
          </Select>
          <div className="flex gap-1">
            <Button variant={viewMode === "table" ? "default" : "outline"} size="sm" onClick={() => setViewMode("table")}>Table</Button>
            <Button variant={viewMode === "grid" ? "default" : "outline"} size="sm" onClick={() => setViewMode("grid")}>Grid</Button>
          </div>
        </CardContent>
      </Card>

      {/* Table View */}
      {viewMode === "table" ? (
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-100 bg-gray-50/50">
                  <th className="text-left py-3 px-5 text-xs font-semibold text-gray-500 uppercase tracking-wide">Patient</th>
                  <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase tracking-wide hidden md:table-cell">Age / Gender</th>
                  <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase tracking-wide hidden lg:table-cell">Condition</th>
                  <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase tracking-wide hidden xl:table-cell">Last Visit</th>
                  <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase tracking-wide">Status</th>
                  <th className="text-right py-3 px-5 text-xs font-semibold text-gray-500 uppercase tracking-wide">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {paginated.map((patient) => (
                  <tr key={patient.id} className="hover:bg-gray-50/50 transition-colors group">
                    <td className="py-3 px-5">
                      <div className="flex items-center gap-3">
                        <Avatar className="h-8 w-8 shrink-0">
                          <AvatarFallback className="text-xs bg-blue-100 text-blue-700">{getInitials(patient.name)}</AvatarFallback>
                        </Avatar>
                        <div>
                          <p className="text-sm font-medium text-gray-900">{patient.name}</p>
                          <p className="text-xs text-gray-400">{patient.email}</p>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-4 hidden md:table-cell">
                      <p className="text-sm text-gray-700">{patient.age}y</p>
                      <p className="text-xs text-gray-400">{patient.gender}</p>
                    </td>
                    <td className="py-3 px-4 hidden lg:table-cell">
                      <p className="text-sm text-gray-700">{patient.condition}</p>
                      <p className="text-xs text-gray-400">Blood: {patient.bloodType}</p>
                    </td>
                    <td className="py-3 px-4 hidden xl:table-cell text-sm text-gray-500">
                      {formatDate(patient.lastVisit)}
                    </td>
                    <td className="py-3 px-4">
                      <Badge variant={statusVariant[patient.status]}>{patient.status}</Badge>
                    </td>
                    <td className="py-3 px-5 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <Link href={`/doctor/patients/${patient.id}`}>
                          <Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 group-hover:opacity-100">
                            <Eye className="h-3.5 w-3.5" />
                          </Button>
                        </Link>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 group-hover:opacity-100">
                              <MoreHorizontal className="h-3.5 w-3.5" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem><Eye className="h-4 w-4 mr-2" />View Record</DropdownMenuItem>
                            <DropdownMenuItem><Phone className="h-4 w-4 mr-2" />Call Patient</DropdownMenuItem>
                            <DropdownMenuItem><Mail className="h-4 w-4 mr-2" />Send Message</DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="flex items-center justify-between px-5 py-3 border-t border-gray-100">
              <p className="text-xs text-gray-400">
                Showing {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, filtered.length)} of {filtered.length}
              </p>
              <div className="flex gap-1">
                <Button variant="outline" size="sm" onClick={() => setPage(page - 1)} disabled={page === 1}>Prev</Button>
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  const pageNum = page <= 3 ? i + 1 : page - 2 + i;
                  if (pageNum < 1 || pageNum > totalPages) return null;
                  return (
                    <Button key={pageNum} variant={pageNum === page ? "default" : "outline"} size="sm"
                      onClick={() => setPage(pageNum)} className="w-8">
                      {pageNum}
                    </Button>
                  );
                })}
                <Button variant="outline" size="sm" onClick={() => setPage(page + 1)} disabled={page === totalPages}>Next</Button>
              </div>
            </div>
          )}
        </Card>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {paginated.map((patient) => (
            <Link key={patient.id} href={`/doctor/patients/${patient.id}`}>
              <Card className="hover:shadow-md transition-all cursor-pointer">
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <Avatar className="h-10 w-10">
                      <AvatarFallback className="bg-blue-100 text-blue-700 text-sm">{getInitials(patient.name)}</AvatarFallback>
                    </Avatar>
                    <Badge variant={statusVariant[patient.status]}>{patient.status}</Badge>
                  </div>
                  <h3 className="font-semibold text-gray-900 text-sm mb-0.5">{patient.name}</h3>
                  <p className="text-xs text-gray-500 mb-3">{patient.age}y · {patient.gender} · {patient.bloodType}</p>
                  <div className="space-y-1">
                    <p className="text-xs text-gray-600 font-medium truncate">{patient.condition}</p>
                    <p className="text-xs text-gray-400">Last visit: {formatDate(patient.lastVisit)}</p>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
