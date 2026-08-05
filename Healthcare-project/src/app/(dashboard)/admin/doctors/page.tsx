"use client";
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Search, UserPlus, MoreHorizontal, Star, Users } from "lucide-react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { getInitials } from "@/lib/utils";

const doctors = Array.from({ length: 24 }, (_, i) => ({
  id: String(i + 1),
  name: ["Dr. Sarah Johnson", "Dr. Michael Chen", "Dr. Emily Rodriguez", "Dr. James Williams", "Dr. Lisa Park", "Dr. Robert Kim"][i % 6],
  specialization: ["Internal Medicine", "Cardiology", "Pediatrics", "Neurology", "Dermatology", "Orthopedics"][i % 6],
  department: ["Medicine", "Cardiology", "Pediatrics", "Neurology", "Dermatology", "Surgery"][i % 6],
  patients: 80 + (i * 17) % 180,
  appointments: 5 + (i * 3) % 15,
  rating: (4.5 + ((i * 0.1) % 0.5)).toFixed(1),
  experience: 3 + (i * 2) % 20,
  fee: 100 + (i * 25) % 250,
  status: i % 7 === 0 ? "inactive" : "active" as "active" | "inactive",
  license: `MD-2024-${String(10000 + i * 7).slice(0, 5)}`,
  joinDate: `202${2 + (i % 3)}-0${1 + (i % 9)}-15`,
}));

export default function AdminDoctorsPage() {
  const [search, setSearch] = useState("");
  const [specFilter, setSpecFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [open, setOpen] = useState(false);

  const filtered = doctors.filter((d) => {
    const matchSearch = d.name.toLowerCase().includes(search.toLowerCase()) || d.specialization.toLowerCase().includes(search.toLowerCase());
    const matchSpec = specFilter === "all" || d.specialization === specFilter;
    const matchStatus = statusFilter === "all" || d.status === statusFilter;
    return matchSearch && matchSpec && matchStatus;
  });

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Doctor Management</h2>
          <p className="text-sm text-gray-500">{filtered.length} of {doctors.length} doctors</p>
        </div>
        <Dialog open={open} onOpenChange={setOpen}>
          <DialogTrigger asChild>
            <Button size="sm" className="gap-2 bg-purple-600 hover:bg-purple-700"><UserPlus className="h-4 w-4" /> Add Doctor</Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>Add New Doctor</DialogTitle>
            </DialogHeader>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <Label>First Name</Label>
                <Input placeholder="John" />
              </div>
              <div className="space-y-1.5">
                <Label>Last Name</Label>
                <Input placeholder="Smith" />
              </div>
              <div className="space-y-1.5 col-span-2">
                <Label>Email</Label>
                <Input type="email" placeholder="dr.smith@medicare.com" />
              </div>
              <div className="space-y-1.5">
                <Label>Specialization</Label>
                <Select>
                  <SelectTrigger><SelectValue placeholder="Select specialty" /></SelectTrigger>
                  <SelectContent>
                    {["Internal Medicine", "Cardiology", "Pediatrics", "Neurology", "Dermatology", "Orthopedics"].map((s) => (
                      <SelectItem key={s} value={s}>{s}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label>License Number</Label>
                <Input placeholder="MD-2024-XXXXX" />
              </div>
              <div className="space-y-1.5">
                <Label>Experience (Years)</Label>
                <Input type="number" placeholder="5" />
              </div>
              <div className="space-y-1.5">
                <Label>Consultation Fee ($)</Label>
                <Input type="number" placeholder="150" />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
              <Button className="bg-purple-600 hover:bg-purple-700" onClick={() => setOpen(false)}>Add Doctor</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
          <Input placeholder="Search doctors…" className="pl-8 bg-white" value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <Select value={specFilter} onValueChange={setSpecFilter}>
          <SelectTrigger className="w-44">
            <SelectValue placeholder="Specialization" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Specialties</SelectItem>
            {["Internal Medicine", "Cardiology", "Pediatrics", "Neurology", "Dermatology", "Orthopedics"].map((s) => (
              <SelectItem key={s} value={s}>{s}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-32">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="inactive">Inactive</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {filtered.map((doctor) => (
          <Card key={doctor.id} className="hover:shadow-md transition-all group">
            <CardContent className="p-5">
              <div className="flex items-start justify-between mb-3">
                <Avatar className="h-11 w-11">
                  <AvatarFallback className="bg-purple-100 text-purple-700 font-semibold">{getInitials(doctor.name)}</AvatarFallback>
                </Avatar>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 group-hover:opacity-100">
                      <MoreHorizontal className="h-3.5 w-3.5" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem>View Profile</DropdownMenuItem>
                    <DropdownMenuItem>Edit Details</DropdownMenuItem>
                    <DropdownMenuItem className="text-red-600">{doctor.status === "active" ? "Deactivate" : "Activate"}</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
              <h3 className="font-semibold text-gray-900 text-sm leading-tight">{doctor.name}</h3>
              <p className="text-xs text-gray-500 mb-3">{doctor.specialization}</p>
              <div className="flex items-center justify-between mb-3">
                <Badge variant={doctor.status === "active" ? "success" : "secondary"}>{doctor.status}</Badge>
                <div className="flex items-center gap-1 text-amber-500 text-xs font-medium">
                  <Star className="h-3 w-3 fill-amber-400" />
                  {doctor.rating}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 text-center">
                <div className="bg-gray-50 rounded-lg py-2">
                  <div className="flex items-center justify-center gap-1 text-gray-500 mb-0.5">
                    <Users className="h-3 w-3" />
                    <span className="text-xs">Patients</span>
                  </div>
                  <p className="text-sm font-bold text-gray-900">{doctor.patients}</p>
                </div>
                <div className="bg-gray-50 rounded-lg py-2">
                  <p className="text-xs text-gray-500 mb-0.5">Fee</p>
                  <p className="text-sm font-bold text-gray-900">${doctor.fee}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
