"use client";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ClipboardList, Plus, Search, Pill, Printer } from "lucide-react";
import { getInitials, formatDate } from "@/lib/utils";

const prescriptions = Array.from({ length: 25 }, (_, i) => ({
  id: String(i + 1),
  patient: ["James Wilson", "Maria Garcia", "Robert Chen", "Emily Davis", "Michael Brown", "Sarah Lee"][i % 6],
  date: new Date(2026, 5, Math.max(1, 9 - Math.floor(i / 3))).toISOString().split("T")[0],
  status: (["active", "active", "completed", "active", "expired"][i % 5]) as "active" | "completed" | "expired",
  medications: [
    [{ name: "Lisinopril 10mg", dosage: "1 daily", duration: "30 days" }],
    [{ name: "Metformin 500mg", dosage: "2 daily", duration: "30 days" }, { name: "Atorvastatin 20mg", dosage: "1 nightly", duration: "30 days" }],
    [{ name: "Salbutamol 100mcg", dosage: "As needed", duration: "90 days" }],
    [{ name: "Ibuprofen 400mg", dosage: "3 daily with food", duration: "14 days" }],
    [{ name: "Ferrous Sulfate 325mg", dosage: "1 daily", duration: "60 days" }],
  ][i % 5],
  instructions: ["Take with water. Avoid alcohol.", "Monitor blood sugar.", "Shake well before use.", "Take with food.", "Avoid dairy products."][i % 5],
  validUntil: new Date(2026, 5 + Math.floor(i / 3), 9).toISOString().split("T")[0],
}));

const statusVariant = { active: "success", completed: "secondary", expired: "destructive" } as const;

export default function PrescriptionsPage() {
  const [search, setSearch] = useState("");
  const [newRx, setNewRx] = useState({ patient: "", medications: [{ name: "", dosage: "", duration: "" }], instructions: "" });
  const [openDialog, setOpenDialog] = useState(false);

  const filtered = prescriptions.filter(
    (rx) => rx.patient.toLowerCase().includes(search.toLowerCase()) ||
      rx.medications.some((m) => m.name.toLowerCase().includes(search.toLowerCase()))
  );

  function addMedication() {
    setNewRx((prev) => ({ ...prev, medications: [...prev.medications, { name: "", dosage: "", duration: "" }] }));
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Prescriptions</h2>
          <p className="text-sm text-gray-500">{filtered.length} records</p>
        </div>
        <Dialog open={openDialog} onOpenChange={setOpenDialog}>
          <DialogTrigger asChild>
            <Button size="sm" className="gap-2"><Plus className="h-4 w-4" /> New Prescription</Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>New Prescription</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div className="space-y-1.5">
                <Label>Patient</Label>
                <Select onValueChange={(v) => setNewRx((p) => ({ ...p, patient: v }))}>
                  <SelectTrigger><SelectValue placeholder="Select patient" /></SelectTrigger>
                  <SelectContent>
                    {["James Wilson", "Maria Garcia", "Robert Chen", "Emily Davis", "Michael Brown"].map((n) => (
                      <SelectItem key={n} value={n}>{n}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {newRx.medications.map((med, idx) => (
                <div key={idx} className="space-y-2 p-3 bg-gray-50 rounded-lg">
                  <p className="text-xs font-semibold text-gray-500 uppercase">Medication {idx + 1}</p>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="col-span-2 space-y-1">
                      <Label className="text-xs">Medication Name & Strength</Label>
                      <Input placeholder="e.g., Lisinopril 10mg" className="text-sm" />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">Dosage</Label>
                      <Input placeholder="e.g., 1 daily" className="text-sm" />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">Duration</Label>
                      <Input placeholder="e.g., 30 days" className="text-sm" />
                    </div>
                  </div>
                </div>
              ))}
              <Button variant="outline" size="sm" onClick={addMedication} className="gap-1.5 text-xs">
                <Plus className="h-3 w-3" /> Add Medication
              </Button>
              <div className="space-y-1.5">
                <Label>Special Instructions</Label>
                <Textarea placeholder="Take with food, avoid alcohol…" rows={2} />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setOpenDialog(false)}>Cancel</Button>
              <Button onClick={() => setOpenDialog(false)}>Issue Prescription</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
        <Input placeholder="Search prescriptions…" className="pl-8 bg-white" value={search} onChange={(e) => setSearch(e.target.value)} />
      </div>

      <div className="space-y-3">
        {filtered.map((rx) => (
          <Card key={rx.id} className="hover:shadow-sm transition-shadow">
            <CardContent className="p-4">
              <div className="flex items-start gap-4">
                <Avatar className="h-9 w-9 shrink-0">
                  <AvatarFallback className="bg-green-100 text-green-700 text-sm">{getInitials(rx.patient)}</AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <p className="font-semibold text-gray-900">{rx.patient}</p>
                      <Badge variant={statusVariant[rx.status]}>{rx.status}</Badge>
                    </div>
                    <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-400 hover:text-gray-700">
                      <Printer className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                  <p className="text-xs text-gray-400 mb-3">Issued: {formatDate(rx.date)} · Valid until: {formatDate(rx.validUntil)}</p>
                  <div className="flex flex-wrap gap-2">
                    {rx.medications.map((med, idx) => (
                      <div key={idx} className="flex items-center gap-1.5 bg-gray-50 border border-gray-100 rounded-lg px-2.5 py-1.5">
                        <Pill className="h-3 w-3 text-green-600 shrink-0" />
                        <div>
                          <p className="text-xs font-medium text-gray-800">{med.name}</p>
                          <p className="text-xs text-gray-400">{med.dosage} · {med.duration}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                  {rx.instructions && (
                    <p className="text-xs text-gray-500 mt-2 italic">Note: {rx.instructions}</p>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
