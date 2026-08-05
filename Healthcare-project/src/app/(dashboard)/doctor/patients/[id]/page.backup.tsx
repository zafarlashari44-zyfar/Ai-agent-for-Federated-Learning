"use client";
import { use } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ArrowLeft, Phone, Mail, Calendar, FileText, ClipboardList, Stethoscope, Activity } from "lucide-react";
import Link from "next/link";
import { getInitials } from "@/lib/utils";

const patient = {
  id: "1", name: "James Wilson", age: 54, gender: "Male", bloodType: "A+",
  phone: "+1 (555) 234-5678", email: "james.wilson@email.com",
  address: "123 Oak Street, Boston, MA 02101",
  condition: "Hypertension", status: "stable" as const,
  insurance: "Blue Cross BlueShield", insuranceNum: "BCB-123456",
  emergencyContact: "Jane Wilson", emergencyPhone: "+1 (555) 234-5679",
  allergies: ["Penicillin", "Sulfa drugs"],
  assignedSince: "2024-03-15",
};

const medicalHistory = [
  { date: "2026-06-08", diagnosis: "Hypertension management", doctor: "Dr. Johnson", notes: "BP well controlled. Continue current medication. Lifestyle advice given." },
  { date: "2026-05-12", diagnosis: "Annual check-up", doctor: "Dr. Johnson", notes: "All vitals normal. Lab results pending." },
  { date: "2026-04-02", diagnosis: "Hypertension follow-up", doctor: "Dr. Johnson", notes: "BP slightly elevated. Adjusted medication dosage." },
  { date: "2026-02-15", diagnosis: "Flu treatment", doctor: "Dr. Wilson", notes: "Prescribed antiviral medication. Rest recommended." },
];

const prescriptions = [
  { id: "rx1", medication: "Lisinopril 10mg", dosage: "1 tablet daily", status: "active", date: "2026-06-08" },
  { id: "rx2", medication: "Amlodipine 5mg", dosage: "1 tablet daily", status: "active", date: "2026-06-08" },
  { id: "rx3", medication: "Aspirin 81mg", dosage: "1 tablet daily", status: "active", date: "2026-04-02" },
  { id: "rx4", medication: "Tamiflu 75mg", dosage: "1 tablet twice daily for 5 days", status: "completed", date: "2026-02-15" },
];

const vitals = [
  { label: "Blood Pressure", value: "128/82", unit: "mmHg", status: "normal" },
  { label: "Heart Rate", value: "74", unit: "bpm", status: "normal" },
  { label: "Temperature", value: "98.6", unit: "°F", status: "normal" },
  { label: "SpO₂", value: "98", unit: "%", status: "normal" },
  { label: "Weight", value: "182", unit: "lbs", status: "normal" },
  { label: "BMI", value: "26.4", unit: "kg/m²", status: "overweight" },
];

export default function PatientDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);

  return (
    <div className="space-y-5 max-w-5xl">
      <div className="flex items-center gap-4">
        <Link href="/doctor/patients">
          <Button variant="ghost" size="sm" className="gap-1.5">
            <ArrowLeft className="h-3.5 w-3.5" /> Patients
          </Button>
        </Link>
        <div className="flex items-center gap-2 ml-auto">
          <Button variant="outline" size="sm" className="gap-1.5">
            <Calendar className="h-3.5 w-3.5" /> Schedule
          </Button>
          <Link href={`/doctor/diagnoses/new?patient=${id}`}>
            <Button size="sm" className="gap-1.5">
              <Stethoscope className="h-3.5 w-3.5" /> New Diagnosis
            </Button>
          </Link>
        </div>
      </div>

      {/* Patient Header */}
      <Card>
        <CardContent className="p-5">
          <div className="flex flex-col sm:flex-row gap-4 items-start">
            <Avatar className="h-16 w-16 shrink-0">
              <AvatarFallback className="bg-blue-100 text-blue-700 text-xl">{getInitials(patient.name)}</AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2 mb-1">
                <h2 className="text-xl font-bold text-gray-900">{patient.name}</h2>
                <Badge variant={patient.status === "stable" ? "success" : patient.status === "review" ? "warning" : "destructive"}>
                  {patient.status}
                </Badge>
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-gray-500 mb-3">
                <span>{patient.age}y · {patient.gender} · {patient.bloodType}</span>
                <span className="flex items-center gap-1"><Phone className="h-3.5 w-3.5" />{patient.phone}</span>
                <span className="flex items-center gap-1"><Mail className="h-3.5 w-3.5" />{patient.email}</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {patient.allergies.map((a) => (
                  <Badge key={a} variant="destructive" className="text-xs">Allergy: {a}</Badge>
                ))}
                <Badge variant="secondary">Insurance: {patient.insurance}</Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Vitals */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2"><Activity className="h-4 w-4 text-blue-600" /> Latest Vitals</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            {vitals.map((v) => (
              <div key={v.label} className="text-center p-3 rounded-lg bg-gray-50">
                <p className="text-xs text-gray-500 mb-1">{v.label}</p>
                <p className="text-lg font-bold text-gray-900">{v.value}</p>
                <p className="text-xs text-gray-400">{v.unit}</p>
                <Badge variant={v.status === "normal" ? "success" : "warning"} className="text-[10px] mt-1">{v.status}</Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Tabs */}
      <Tabs defaultValue="history">
        <TabsList className="bg-white border border-gray-100">
          <TabsTrigger value="history">Medical History</TabsTrigger>
          <TabsTrigger value="prescriptions">Prescriptions</TabsTrigger>
          <TabsTrigger value="info">Patient Info</TabsTrigger>
        </TabsList>

        <TabsContent value="history" className="mt-4">
          <Card>
            <CardContent className="p-0">
              <div className="divide-y divide-gray-50">
                {medicalHistory.map((record, i) => (
                  <div key={i} className="p-5">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-semibold text-gray-900">{record.diagnosis}</h4>
                        <p className="text-xs text-gray-400">{record.doctor} · {record.date}</p>
                      </div>
                      <Button variant="ghost" size="sm" className="gap-1 text-xs">
                        <FileText className="h-3.5 w-3.5" /> View
                      </Button>
                    </div>
                    <p className="text-sm text-gray-600">{record.notes}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="prescriptions" className="mt-4">
          <Card>
            <CardContent className="p-0">
              <div className="divide-y divide-gray-50">
                {prescriptions.map((rx) => (
                  <div key={rx.id} className="flex items-center justify-between p-5">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-green-50 rounded-lg">
                        <ClipboardList className="h-4 w-4 text-green-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900">{rx.medication}</p>
                        <p className="text-xs text-gray-400">{rx.dosage} · {rx.date}</p>
                      </div>
                    </div>
                    <Badge variant={rx.status === "active" ? "success" : "secondary"}>{rx.status}</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="info" className="mt-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[
              { label: "Address", value: patient.address },
              { label: "Insurance Provider", value: patient.insurance },
              { label: "Insurance Number", value: patient.insuranceNum },
              { label: "Emergency Contact", value: patient.emergencyContact },
              { label: "Emergency Phone", value: patient.emergencyPhone },
              { label: "Patient Since", value: patient.assignedSince },
            ].map(({ label, value }) => (
              <Card key={label}>
                <CardContent className="p-4">
                  <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-1">{label}</p>
                  <p className="text-sm text-gray-900">{value}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
