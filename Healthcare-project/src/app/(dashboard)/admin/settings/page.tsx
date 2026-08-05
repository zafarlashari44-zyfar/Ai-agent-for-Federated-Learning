"use client";
import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Save, Globe, Bell, Shield, Database, Cpu, RefreshCw } from "lucide-react";

type OllamaHealth = {
  status: "ready" | "model_missing" | "offline";
  model: string;
  availableModels: string[];
  message: string;
};

type N8nHealth = {
  status: "ready" | "offline";
  url: string;
  webhookPath: string;
  webhookSecretConfigured: boolean;
  message: string;
};

async function fetchOllamaHealth(): Promise<OllamaHealth> {
  try {
    const response = await fetch("/api/ai/health", { cache: "no-store" });
    return (await response.json()) as OllamaHealth;
  } catch {
    return {
      status: "offline",
      model: "unknown",
      availableModels: [],
      message: "Unable to check the local Ollama service.",
    };
  }
}

async function fetchN8nHealth(): Promise<N8nHealth> {
  try {
    const response = await fetch("/api/automation/health", {
      cache: "no-store",
    });
    return (await response.json()) as N8nHealth;
  } catch {
    return {
      status: "offline",
      url: "http://localhost:5678",
      webhookPath: "medicare-events",
      webhookSecretConfigured: false,
      message: "Unable to check the local n8n service.",
    };
  }
}

export default function AdminSettingsPage() {
  const [saved, setSaved] = useState(false);
  const [ollamaHealth, setOllamaHealth] = useState<OllamaHealth | null>(null);
  const [checkingOllama, setCheckingOllama] = useState(true);
  const [n8nHealth, setN8nHealth] = useState<N8nHealth | null>(null);
  const [checkingN8n, setCheckingN8n] = useState(true);

  const checkOllama = useCallback(async () => {
    setCheckingOllama(true);
    setOllamaHealth(await fetchOllamaHealth());
    setCheckingOllama(false);
  }, []);

  const checkN8n = useCallback(async () => {
    setCheckingN8n(true);
    setN8nHealth(await fetchN8nHealth());
    setCheckingN8n(false);
  }, []);

  useEffect(() => {
    let active = true;

    void fetchOllamaHealth().then((health) => {
      if (active) {
        setOllamaHealth(health);
        setCheckingOllama(false);
      }
    });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;

    void fetchN8nHealth().then((health) => {
      if (active) {
        setN8nHealth(health);
        setCheckingN8n(false);
      }
    });

    return () => {
      active = false;
    };
  }, []);

  function handleSave() {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  return (
    <div className="space-y-5 max-w-3xl">
      <div>
        <h2 className="text-xl font-bold text-gray-900">System Settings</h2>
        <p className="text-sm text-gray-500">Configure platform-wide settings and preferences</p>
      </div>

      <Tabs defaultValue="general">
        <TabsList className="bg-white border border-gray-100 flex-wrap h-auto gap-1 p-1">
          <TabsTrigger value="general" className="gap-1.5"><Globe className="h-3.5 w-3.5" /> General</TabsTrigger>
          <TabsTrigger value="notifications" className="gap-1.5"><Bell className="h-3.5 w-3.5" /> Notifications</TabsTrigger>
          <TabsTrigger value="security" className="gap-1.5"><Shield className="h-3.5 w-3.5" /> Security</TabsTrigger>
          <TabsTrigger value="integrations" className="gap-1.5"><Cpu className="h-3.5 w-3.5" /> Integrations</TabsTrigger>
          <TabsTrigger value="database" className="gap-1.5"><Database className="h-3.5 w-3.5" /> Database</TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="mt-4 space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Clinic Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5 col-span-2">
                  <Label>Clinic / Hospital Name</Label>
                  <Input defaultValue="MediCare General Hospital" />
                </div>
                <div className="space-y-1.5">
                  <Label>Primary Email</Label>
                  <Input type="email" defaultValue="info@medicare.com" />
                </div>
                <div className="space-y-1.5">
                  <Label>Phone Number</Label>
                  <Input defaultValue="+1 (555) 000-0000" />
                </div>
                <div className="space-y-1.5 col-span-2">
                  <Label>Address</Label>
                  <Input defaultValue="100 Healthcare Blvd, Boston, MA 02101" />
                </div>
                <div className="space-y-1.5">
                  <Label>Timezone</Label>
                  <Select defaultValue="America/New_York">
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="America/New_York">Eastern Time (ET)</SelectItem>
                      <SelectItem value="America/Chicago">Central Time (CT)</SelectItem>
                      <SelectItem value="America/Denver">Mountain Time (MT)</SelectItem>
                      <SelectItem value="America/Los_Angeles">Pacific Time (PT)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Currency</Label>
                  <Select defaultValue="USD">
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="USD">USD — US Dollar</SelectItem>
                      <SelectItem value="EUR">EUR — Euro</SelectItem>
                      <SelectItem value="GBP">GBP — British Pound</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Appointment Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Default Duration (minutes)</Label>
                  <Input type="number" defaultValue="30" />
                </div>
                <div className="space-y-1.5">
                  <Label>Max Daily Appointments per Doctor</Label>
                  <Input type="number" defaultValue="20" />
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">Allow online booking</p>
                  <p className="text-xs text-gray-400">Patients can self-schedule appointments</p>
                </div>
                <Switch defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">Require confirmation</p>
                  <p className="text-xs text-gray-400">Doctor must confirm before appointment is set</p>
                </div>
                <Switch defaultChecked />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="mt-4 space-y-4">
          <Card>
            <CardContent className="p-5 space-y-5">
              {[
                { label: "Email Notifications", desc: "Send email alerts for critical events", checked: true },
                { label: "SMS Reminders", desc: "Send SMS reminders to patients", checked: true },
                { label: "System Alerts", desc: "Internal platform alerts and warnings", checked: true },
                { label: "Daily Summary", desc: "Send daily summary report to admin", checked: false },
                { label: "Billing Alerts", desc: "Notify on overdue or failed payments", checked: true },
                { label: "Security Notifications", desc: "Alert on suspicious login attempts", checked: true },
              ].map(({ label, desc, checked }) => (
                <div key={label} className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{label}</p>
                    <p className="text-xs text-gray-400">{desc}</p>
                  </div>
                  <Switch defaultChecked={checked} />
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="mt-4 space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Security Policies</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              {[
                { label: "Two-Factor Authentication", desc: "Require 2FA for all admin accounts", checked: true },
                { label: "Session Timeout", desc: "Auto-logout after 30 minutes of inactivity", checked: true },
                { label: "IP Allowlist", desc: "Only allow logins from approved IP ranges", checked: false },
                { label: "Audit Logging", desc: "Log all user actions and access", checked: true },
              ].map(({ label, desc, checked }) => (
                <div key={label} className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{label}</p>
                    <p className="text-xs text-gray-400">{desc}</p>
                  </div>
                  <Switch defaultChecked={checked} />
                </div>
              ))}
              <div className="space-y-1.5">
                <Label>Minimum Password Length</Label>
                <Input type="number" defaultValue="8" className="w-24" />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="integrations" className="mt-4 space-y-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="font-semibold text-gray-900">Ollama AI</p>
                    <Badge
                      variant={
                        checkingOllama
                          ? "secondary"
                          : ollamaHealth?.status === "ready"
                            ? "success"
                            : ollamaHealth?.status === "model_missing"
                              ? "warning"
                              : "destructive"
                      }
                    >
                      {checkingOllama
                        ? "checking"
                        : ollamaHealth?.status.replace("_", " ") ?? "offline"}
                    </Badge>
                  </div>
                  <p className="text-xs text-gray-500 mb-3">
                    {ollamaHealth?.message ??
                      "Checking the local AI service and configured model..."}
                  </p>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-1.5">
                      <Label className="text-xs">OLLAMA_URL</Label>
                      <Input
                        value="http://localhost:11434"
                        readOnly
                        className="text-sm font-mono"
                      />
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-xs">Configured model</Label>
                      <Input
                        value={ollamaHealth?.model ?? "llama3.2"}
                        readOnly
                        className="text-sm font-mono"
                      />
                    </div>
                  </div>
                  {ollamaHealth && ollamaHealth.availableModels.length > 0 && (
                    <p className="mt-3 text-xs text-gray-500">
                      Installed models: {ollamaHealth.availableModels.join(", ")}
                    </p>
                  )}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={checkOllama}
                  disabled={checkingOllama}
                  className="gap-1.5"
                >
                  <RefreshCw
                    className={`h-3.5 w-3.5 ${checkingOllama ? "animate-spin" : ""}`}
                  />
                  Recheck
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="font-semibold text-gray-900">
                      n8n Automation
                    </p>
                    <Badge
                      variant={
                        checkingN8n
                          ? "secondary"
                          : n8nHealth?.status === "ready" &&
                              n8nHealth.webhookSecretConfigured
                            ? "success"
                            : n8nHealth?.status === "ready"
                              ? "warning"
                              : "destructive"
                      }
                    >
                      {checkingN8n
                        ? "checking"
                        : n8nHealth?.status === "ready" &&
                            !n8nHealth.webhookSecretConfigured
                          ? "configuration needed"
                          : n8nHealth?.status ?? "offline"}
                    </Badge>
                  </div>
                  <p className="text-xs text-gray-500 mb-3">
                    {n8nHealth?.message ??
                      "Checking the local workflow automation service..."}
                  </p>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-1.5">
                      <Label className="text-xs">N8N_URL</Label>
                      <Input
                        value={n8nHealth?.url ?? "http://localhost:5678"}
                        readOnly
                        className="text-sm font-mono"
                      />
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-xs">Production webhook</Label>
                      <Input
                        value={`/webhook/${n8nHealth?.webhookPath ?? "medicare-events"}`}
                        readOnly
                        className="text-sm font-mono"
                      />
                    </div>
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={checkN8n}
                  disabled={checkingN8n}
                  className="gap-1.5"
                >
                  <RefreshCw
                    className={`h-3.5 w-3.5 ${checkingN8n ? "animate-spin" : ""}`}
                  />
                  Recheck
                </Button>
              </div>
            </CardContent>
          </Card>

          {[
            { name: "Supabase", desc: "Database, auth, and file storage", status: "connected", url: "NEXT_PUBLIC_SUPABASE_URL", defaultUrl: "https://xxx.supabase.co" },
          ].map((integration) => (
            <Card key={integration.name}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <p className="font-semibold text-gray-900">{integration.name}</p>
                      <Badge variant={integration.status === "connected" ? "success" : "secondary"}>
                        {integration.status}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-500 mb-3">{integration.desc}</p>
                    <div className="space-y-1.5">
                      <Label className="text-xs">{integration.url}</Label>
                      <Input defaultValue={integration.defaultUrl} className="text-sm font-mono" />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="database" className="mt-4 space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Database Status</CardTitle>
              <CardDescription>Supabase PostgreSQL</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                {[
                  { label: "Status", value: "Connected", ok: true },
                  { label: "Region", value: "us-east-1", ok: true },
                  { label: "Records", value: "48,392", ok: true },
                  { label: "Storage Used", value: "2.4 GB / 8 GB", ok: true },
                ].map(({ label, value, ok }) => (
                  <div key={label} className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-xs text-gray-500 mb-1">{label}</p>
                    <p className="text-sm font-semibold text-gray-900">{value}</p>
                    {ok && <Badge variant="success" className="text-[10px] mt-1">OK</Badge>}
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm">Run Backup Now</Button>
                <Button variant="outline" size="sm">View Logs</Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="flex justify-end">
        <Button onClick={handleSave} className="gap-2 bg-purple-600 hover:bg-purple-700 min-w-28">
          {saved ? "✓ Saved!" : <><Save className="h-4 w-4" /> Save Settings</>}
        </Button>
      </div>
    </div>
  );
}
