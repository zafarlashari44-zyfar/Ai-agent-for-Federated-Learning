"use client";
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Download, MoreHorizontal, UserPlus } from "lucide-react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { getInitials, formatDate } from "@/lib/utils";

const patients = Array.from({ length: 60 }, (_, i) => ({
  id: String(i + 1),
  name: ["James Wilson", "Maria Garcia", "Robert Chen", "Emily Davis", "Michael Brown", "Sarah Lee", "David Kim", "Anna White", "Thomas Moore", "Jessica Taylor"][i % 10],
  age: 22 + (i * 5) % 55,
  gender: i % 2 === 0 ? "Male" : "Female",
  doctor: ["Dr. Sarah Johnson", "Dr. Michael Chen", "Dr. Emily Rodriguez", "Dr. James Williams"][i % 4],
  condition: ["Hypertension", "Diabetes", "Asthma", "Arthritis", "Anemia", "Migraine"][i % 6],
  status: ["active", "active", "inactive", "active"][i % 4] as "active" | "inactive",
  registered: new Date(2024, i % 12, 1 + (i % 28)).toISOString().split("T")[0],
  lastVisit: new Date(2026, 5, Math.max(1, 9 - (i % 30))).toISOString().split("T")[0],
  insurance: ["Blue Cross", "Aetna", "United Health", "None", "Cigna"][i % 5],
}));

const PAGE_SIZE = 20;

export default function AdminPatientsPage() {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [page, setPage] = useState(1);

  const filtered = patients.filter((p) => {
    const matchSearch = p.name.toLowerCase().includes(search.toLowerCase()) || p.doctor.toLowerCase().includes(search.toLowerCase());
    const matchStatus = statusFilter === "all" || p.status === statusFilter;
    return matchSearch && matchStatus;
  });

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Patient Management</h2>
          <p className="text-sm text-gray-500">{filtered.length} patients</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="gap-2"><Download className="h-4 w-4" /> Export</Button>
          <Button size="sm" className="gap-2 bg-purple-600 hover:bg-purple-700"><UserPlus className="h-4 w-4" /> Add Patient</Button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
          <Input placeholder="Search patients or doctors…" className="pl-8 bg-white" value={search} onChange={(e) => { setSearch(e.target.value); setPage(1); }} />
        </div>
        <Select value={statusFilter} onValueChange={(v) => { setStatusFilter(v); setPage(1); }}>
          <SelectTrigger className="w-36"><SelectValue placeholder="Status" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="inactive">Inactive</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Table */}
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50/50">
                <th className="text-left py-3 px-5 text-xs font-semibold text-gray-500 uppercase">Patient</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden md:table-cell">Age</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden lg:table-cell">Assigned Doctor</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden xl:table-cell">Condition</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden xl:table-cell">Last Visit</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Status</th>
                <th className="text-right py-3 px-5 text-xs font-semibold text-gray-500 uppercase">Actions</th>
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
                        <p className="text-xs text-gray-400">{patient.gender} · {patient.insurance}</p>
                      </div>
                    </div>
                  </td>
                  <td className="py-3 px-4 hidden md:table-cell text-sm text-gray-600">{patient.age}y</td>
                  <td className="py-3 px-4 hidden lg:table-cell text-sm text-gray-600">{patient.doctor}</td>
                  <td className="py-3 px-4 hidden xl:table-cell text-sm text-gray-600">{patient.condition}</td>
                  <td className="py-3 px-4 hidden xl:table-cell text-sm text-gray-500">{formatDate(patient.lastVisit)}</td>
                  <td className="py-3 px-4">
                    <Badge variant={patient.status === "active" ? "success" : "secondary"}>{patient.status}</Badge>
                  </td>
                  <td className="py-3 px-5 text-right">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 group-hover:opacity-100">
                          <MoreHorizontal className="h-3.5 w-3.5" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem>View Profile</DropdownMenuItem>
                        <DropdownMenuItem>Edit Patient</DropdownMenuItem>
                        <DropdownMenuItem>Assign Doctor</DropdownMenuItem>
                        <DropdownMenuItem className="text-red-600">Deactivate</DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-5 py-3 border-t border-gray-100">
            <p className="text-xs text-gray-400">Showing {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, filtered.length)} of {filtered.length}</p>
            <div className="flex gap-1">
              <Button variant="outline" size="sm" onClick={() => setPage(page - 1)} disabled={page === 1}>Prev</Button>
              <Button variant="outline" size="sm" onClick={() => setPage(page + 1)} disabled={page === totalPages}>Next</Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
