"use client";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Stethoscope, Plus, Search, Brain, Loader2, ChevronRight } from "lucide-react";
import { getInitials, formatDate } from "@/lib/utils";

const diagnoses = Array.from({ length: 20 }, (_, i) => ({
  id: String(i + 1),
  patient: ["James Wilson", "Maria Garcia", "Robert Chen", "Emily Davis", "Michael Brown"][i % 5],
  diagnosis: ["Hypertension Stage 1", "Type 2 Diabetes Mellitus", "Bronchial Asthma", "Rheumatoid Arthritis", "Iron Deficiency Anemia"][i % 5],
  symptoms: [
    ["Headache", "Dizziness", "Elevated BP"],
    ["Polyuria", "Polydipsia", "Fatigue"],
    ["Wheezing", "Shortness of breath", "Cough"],
    ["Joint pain", "Swelling", "Morning stiffness"],
    ["Fatigue", "Pallor", "Palpitations"],
  ][i % 5],
  icd: ["I10", "E11", "J45.20", "M05", "D50.9"][i % 5],
  severity: ["mild", "moderate", "severe"][i % 3] as "mild" | "moderate" | "severe",
  date: new Date(2026, 5, Math.max(1, 9 - i)).toISOString().split("T")[0],
  treatmentPlan: "Follow standard protocol for the diagnosis.",
}));

const severityVariant = { mild: "success", moderate: "warning", severe: "destructive" } as const;

export default function DiagnosesPage() {
  const [search, setSearch] = useState("");
  const [aiInput, setAiInput] = useState("");
  const [aiResult, setAiResult] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [openAI, setOpenAI] = useState(false);

  const filtered = diagnoses.filter(
    (d) => d.patient.toLowerCase().includes(search.toLowerCase()) ||
      d.diagnosis.toLowerCase().includes(search.toLowerCase())
  );

  async function runAIDiagnosis() {
    if (!aiInput.trim()) return;
    setAiLoading(true);
    try {
      const res = await fetch("/api/ai/diagnose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms: aiInput.split(",").map((s) => s.trim()) }),
      });
      const data = await res.json();
      setAiResult(data.result || "No result returned from AI.");
    } catch {
      setAiResult("Unable to reach AI service. Please ensure Ollama is running locally.");
    }
    setAiLoading(false);
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Diagnoses</h2>
          <p className="text-sm text-gray-500">{filtered.length} records</p>
        </div>
        <div className="flex gap-2">
          <Dialog open={openAI} onOpenChange={setOpenAI}>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                <Brain className="h-4 w-4 text-purple-600" /> AI Support
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-lg">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-purple-600" /> AI Diagnosis Support
                </DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <p className="text-sm text-gray-500">Enter patient symptoms separated by commas. The AI will suggest possible diagnoses and recommended investigations.</p>
                <div className="space-y-1.5">
                  <Label>Symptoms</Label>
                  <Textarea
                    placeholder="e.g., chest pain, shortness of breath, elevated BP, fatigue"
                    value={aiInput}
                    onChange={(e) => setAiInput(e.target.value)}
                    rows={3}
                  />
                </div>
                <Button onClick={runAIDiagnosis} disabled={aiLoading || !aiInput.trim()} className="w-full gap-2">
                  {aiLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Brain className="h-4 w-4" />}
                  {aiLoading ? "Analyzing…" : "Get AI Suggestions"}
                </Button>
                {aiResult && (
                  <div className="bg-purple-50 border border-purple-100 rounded-lg p-4">
                    <p className="text-xs font-semibold text-purple-700 mb-2 uppercase tracking-wide">AI Analysis</p>
                    <p className="text-sm text-gray-800 whitespace-pre-wrap">{aiResult}</p>
                    <p className="text-xs text-gray-400 mt-2 italic">This is AI-assisted support only. Clinical judgment should always take precedence.</p>
                  </div>
                )}
              </div>
            </DialogContent>
          </Dialog>
          <Button size="sm" className="gap-2">
            <Plus className="h-4 w-4" /> New Diagnosis
          </Button>
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
        <Input placeholder="Search diagnoses…" className="pl-8 bg-white" value={search} onChange={(e) => setSearch(e.target.value)} />
      </div>

      {/* List */}
      <div className="space-y-3">
        {filtered.map((diag) => (
          <Card key={diag.id} className="hover:shadow-md transition-all cursor-pointer group">
            <CardContent className="p-4">
              <div className="flex items-start gap-4">
                <Avatar className="h-10 w-10 shrink-0">
                  <AvatarFallback className="bg-blue-100 text-blue-700 text-sm">{getInitials(diag.patient)}</AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2 mb-1">
                    <h3 className="font-semibold text-gray-900">{diag.diagnosis}</h3>
                    <Badge variant="outline" className="text-xs font-mono">{diag.icd}</Badge>
                    <Badge variant={severityVariant[diag.severity]}>{diag.severity}</Badge>
                  </div>
                  <p className="text-sm text-gray-500 mb-2">Patient: {diag.patient} · {formatDate(diag.date)}</p>
                  <div className="flex flex-wrap gap-1.5">
                    {diag.symptoms.map((s) => (
                      <span key={s} className="text-xs bg-gray-100 text-gray-600 rounded px-2 py-0.5">{s}</span>
                    ))}
                  </div>
                </div>
                <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-blue-600 transition-colors shrink-0 mt-1" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
