"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  FileText,
  Pill,
  Search,
} from "lucide-react";
import { formatDate } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

export interface PatientMedicalRecord {
  id: string;
  createdAt: string;
  diagnosis: string;
  doctor: string;
  specialty: string;
  symptoms: string[];
  treatmentPlan: string | null;
  notes: string | null;
  vitals: Record<string, string | number>;
  prescriptions: string[];
}

function vitalLabel(key: string) {
  const labels: Record<string, string> = {
    bp: "Blood Pressure",
    blood_pressure: "Blood Pressure",
    hr: "Heart Rate",
    heart_rate: "Heart Rate",
    temp: "Temperature",
    temperature: "Temperature",
    weight: "Weight",
    blood_glucose: "Blood Glucose",
    oxygen_saturation: "Oxygen Saturation",
  };
  return labels[key] || key.replaceAll("_", " ");
}

export function RecordsClient({
  records,
}: {
  records: PatientMedicalRecord[];
}) {
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const query = search.trim().toLowerCase();
  const filtered = records.filter((record) =>
    [record.diagnosis, record.doctor, record.specialty]
      .some((value) => value.toLowerCase().includes(query)),
  );

  return (
    <div className="space-y-5 max-w-3xl">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Medical Records</h2>
        <p className="text-sm text-gray-500">
          {records.length} records in your history
        </p>
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
        <Input
          placeholder="Search records"
          className="pl-8 bg-white"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </div>

      {filtered.length ? (
        <div className="space-y-3">
          {filtered.map((record) => {
            const isOpen = expanded === record.id;
            return (
              <Card key={record.id} className="overflow-hidden">
                <button
                  className="w-full text-left"
                  onClick={() => setExpanded(isOpen ? null : record.id)}
                >
                  <CardContent className="p-4 flex items-center gap-4 hover:bg-gray-50">
                    <div className="p-2.5 bg-blue-50 rounded-lg">
                      <FileText className="h-4 w-4 text-blue-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex flex-wrap items-center gap-2 mb-1">
                        <p className="text-sm font-semibold text-gray-900">
                          {record.diagnosis}
                        </p>
                        <Badge variant="info" className="text-xs">
                          {record.specialty}
                        </Badge>
                      </div>
                      <p className="text-xs text-gray-400">
                        {record.doctor} · {formatDate(record.createdAt)}
                      </p>
                    </div>
                    {isOpen ? (
                      <ChevronUp className="h-4 w-4 text-gray-400" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-gray-400" />
                    )}
                  </CardContent>
                </button>

                {isOpen && (
                  <div className="border-t border-gray-100 bg-gray-50/50 p-5 space-y-4">
                    {(record.notes || record.treatmentPlan) && (
                      <div>
                        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                          Clinical Notes
                        </p>
                        {record.notes && (
                          <p className="text-sm text-gray-700">{record.notes}</p>
                        )}
                        {record.treatmentPlan && (
                          <p className="text-sm text-gray-700 mt-2">
                            <strong>Treatment plan:</strong> {record.treatmentPlan}
                          </p>
                        )}
                      </div>
                    )}

                    {record.symptoms.length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                          Symptoms Reported
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {record.symptoms.map((symptom) => (
                            <span
                              key={symptom}
                              className="text-xs bg-white border border-gray-200 rounded px-2 py-0.5 text-gray-600"
                            >
                              {symptom}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {Object.keys(record.vitals).length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                          Vitals at Visit
                        </p>
                        <div className="flex flex-wrap gap-3">
                          {Object.entries(record.vitals).map(([key, value]) => (
                            <div
                              key={key}
                              className="bg-white border border-gray-100 rounded-lg px-3 py-2 text-center"
                            >
                              <p className="text-xs text-gray-400 capitalize">
                                {vitalLabel(key)}
                              </p>
                              <p className="text-sm font-semibold text-gray-900">
                                {value}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {record.prescriptions.length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                          Prescriptions
                        </p>
                        <div className="space-y-1">
                          {record.prescriptions.map((prescription) => (
                            <div
                              key={prescription}
                              className="flex items-center gap-2 text-sm text-gray-700"
                            >
                              <Pill className="h-3.5 w-3.5 text-green-600" />
                              {prescription}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      ) : (
        <Card>
          <CardContent className="p-10 text-center text-sm text-gray-500">
            {records.length
              ? "No records match your search."
              : "No medical records have been added yet."}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
