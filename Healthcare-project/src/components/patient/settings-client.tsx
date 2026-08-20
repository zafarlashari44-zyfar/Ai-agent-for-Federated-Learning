"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Bell, Heart, Save, Shield, User } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import type { Database } from "@/types/database";
import { getInitials } from "@/lib/utils";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

type Profile = Database["public"]["Tables"]["profiles"]["Row"];
type Patient = Database["public"]["Tables"]["patients"]["Row"];
type Preferences =
  Database["public"]["Tables"]["notification_preferences"]["Row"];

const preferenceRows = [
  ["appointment_reminders", "Appointment Reminders", "Notices before appointments"],
  ["prescription_reminders", "Prescription Reminders", "Medication and refill reminders"],
  ["lab_results", "Lab Results Ready", "Alerts when results are available"],
  ["new_messages", "Doctor Messages", "Alerts for new care-team messages"],
  ["health_tips", "Health Tips", "Health and wellness tips"],
  ["billing_notifications", "Billing Notifications", "Invoice and payment reminders"],
] as const;

export function SettingsClient({
  profile,
  patient,
  preferences,
}: {
  profile: Profile;
  patient: Patient;
  preferences: Preferences;
}) {
  const router = useRouter();
  const [profileForm, setProfileForm] = useState({
    fullName: profile.full_name,
    phone: profile.phone ?? "",
    dateOfBirth: profile.date_of_birth ?? "",
    gender: profile.gender ?? "",
    address: profile.address ?? "",
  });
  const [healthForm, setHealthForm] = useState({
    bloodType: patient.blood_type ?? "",
    insuranceProvider: patient.insurance_provider ?? "",
    insuranceNumber: patient.insurance_number ?? "",
    allergies: patient.allergies.join(", "),
    emergencyName: patient.emergency_contact_name ?? "",
    emergencyPhone: patient.emergency_contact_phone ?? "",
  });
  const [preferenceForm, setPreferenceForm] = useState(preferences);
  const [password, setPassword] = useState({
    current: "",
    next: "",
    confirm: "",
  });
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function saveSettings() {
    setSaving(true);
    setMessage("");
    setError("");
    const supabase = createClient();
    const [profileResult, patientResult, preferenceResult] = await Promise.all([
      supabase
        .from("profiles")
        .update({
          full_name: profileForm.fullName.trim(),
          phone: profileForm.phone.trim() || null,
          date_of_birth: profileForm.dateOfBirth || null,
          gender:
            (profileForm.gender as Database["public"]["Enums"]["gender"]) || null,
          address: profileForm.address.trim() || null,
        })
        .eq("id", profile.id),
      supabase
        .from("patients")
        .update({
          blood_type: healthForm.bloodType || null,
          insurance_provider: healthForm.insuranceProvider.trim() || null,
          insurance_number: healthForm.insuranceNumber.trim() || null,
          allergies: healthForm.allergies
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean),
          emergency_contact_name: healthForm.emergencyName.trim() || null,
          emergency_contact_phone: healthForm.emergencyPhone.trim() || null,
        })
        .eq("id", patient.id),
      supabase
        .from("notification_preferences")
        .update({
          appointment_reminders: preferenceForm.appointment_reminders,
          prescription_reminders: preferenceForm.prescription_reminders,
          lab_results: preferenceForm.lab_results,
          new_messages: preferenceForm.new_messages,
          health_tips: preferenceForm.health_tips,
          billing_notifications: preferenceForm.billing_notifications,
          email_enabled: preferenceForm.email_enabled,
          sms_enabled: preferenceForm.sms_enabled,
        })
        .eq("user_id", profile.id),
    ]);
    setSaving(false);

    const firstError =
      profileResult.error || patientResult.error || preferenceResult.error;
    if (firstError) {
      setError(firstError.message);
      return;
    }

    setMessage("Settings saved.");
    router.refresh();
  }

  async function changePassword() {
    setMessage("");
    setError("");
    if (password.next.length < 8) {
      setError("The new password must be at least 8 characters.");
      return;
    }
    if (password.next !== password.confirm) {
      setError("The new passwords do not match.");
      return;
    }

    setSaving(true);
    const supabase = createClient();
    const { error: loginError } = await supabase.auth.signInWithPassword({
      email: profile.email,
      password: password.current,
    });
    if (loginError) {
      setSaving(false);
      setError("The current password is incorrect.");
      return;
    }
    const { error: passwordError } = await supabase.auth.updateUser({
      password: password.next,
    });
    setSaving(false);
    if (passwordError) {
      setError(passwordError.message);
      return;
    }

    setPassword({ current: "", next: "", confirm: "" });
    setMessage("Password updated.");
  }

  return (
    <div className="space-y-5 max-w-2xl">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Settings</h2>
        <p className="text-sm text-gray-500">Manage your live profile and preferences</p>
      </div>

      {(message || error) && (
        <p className={`rounded-lg p-3 text-sm ${error ? "bg-red-50 text-red-600" : "bg-green-50 text-green-700"}`}>
          {error || message}
        </p>
      )}

      <Tabs defaultValue="profile">
        <TabsList className="bg-white border border-gray-100">
          <TabsTrigger value="profile" className="gap-1.5"><User className="h-3.5 w-3.5" /> Profile</TabsTrigger>
          <TabsTrigger value="health" className="gap-1.5"><Heart className="h-3.5 w-3.5" /> Health Info</TabsTrigger>
          <TabsTrigger value="notifications" className="gap-1.5"><Bell className="h-3.5 w-3.5" /> Notifications</TabsTrigger>
          <TabsTrigger value="security" className="gap-1.5"><Shield className="h-3.5 w-3.5" /> Security</TabsTrigger>
        </TabsList>

        <TabsContent value="profile" className="mt-4">
          <Card>
            <CardContent className="p-5 space-y-4">
              <div className="flex items-center gap-4">
                <Avatar className="h-16 w-16">
                  <AvatarImage src={profile.avatar_url ?? undefined} />
                  <AvatarFallback className="bg-green-100 text-green-700 text-xl">
                    {getInitials(profileForm.fullName)}
                  </AvatarFallback>
                </Avatar>
                <p className="text-xs text-gray-400">
                  Profile images can be managed after an avatar storage bucket is configured.
                </p>
              </div>
              <div className="space-y-1.5">
                <Label>Full Name</Label>
                <Input value={profileForm.fullName} onChange={(event) => setProfileForm({ ...profileForm, fullName: event.target.value })} />
              </div>
              <div className="space-y-1.5">
                <Label>Email</Label>
                <Input type="email" value={profile.email} disabled />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Phone</Label>
                  <Input value={profileForm.phone} onChange={(event) => setProfileForm({ ...profileForm, phone: event.target.value })} />
                </div>
                <div className="space-y-1.5">
                  <Label>Date of Birth</Label>
                  <Input type="date" value={profileForm.dateOfBirth} onChange={(event) => setProfileForm({ ...profileForm, dateOfBirth: event.target.value })} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Gender</Label>
                  <Select value={profileForm.gender || undefined} onValueChange={(value) => setProfileForm({ ...profileForm, gender: value })}>
                    <SelectTrigger><SelectValue placeholder="Select gender" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="male">Male</SelectItem>
                      <SelectItem value="female">Female</SelectItem>
                      <SelectItem value="other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Address</Label>
                  <Input value={profileForm.address} onChange={(event) => setProfileForm({ ...profileForm, address: event.target.value })} />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="health" className="mt-4">
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-base">Medical Information</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Blood Type</Label>
                  <Select value={healthForm.bloodType || undefined} onValueChange={(value) => setHealthForm({ ...healthForm, bloodType: value })}>
                    <SelectTrigger><SelectValue placeholder="Select blood type" /></SelectTrigger>
                    <SelectContent>
                      {["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"].map((bloodType) => (
                        <SelectItem key={bloodType} value={bloodType}>{bloodType}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Insurance Provider</Label>
                  <Input value={healthForm.insuranceProvider} onChange={(event) => setHealthForm({ ...healthForm, insuranceProvider: event.target.value })} />
                </div>
              </div>
              <div className="space-y-1.5">
                <Label>Insurance Number</Label>
                <Input value={healthForm.insuranceNumber} onChange={(event) => setHealthForm({ ...healthForm, insuranceNumber: event.target.value })} />
              </div>
              <div className="space-y-1.5">
                <Label>Known Allergies</Label>
                <Input value={healthForm.allergies} onChange={(event) => setHealthForm({ ...healthForm, allergies: event.target.value })} placeholder="Separate with commas" />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Emergency Contact Name</Label>
                  <Input value={healthForm.emergencyName} onChange={(event) => setHealthForm({ ...healthForm, emergencyName: event.target.value })} />
                </div>
                <div className="space-y-1.5">
                  <Label>Emergency Contact Phone</Label>
                  <Input value={healthForm.emergencyPhone} onChange={(event) => setHealthForm({ ...healthForm, emergencyPhone: event.target.value })} />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="mt-4">
          <Card>
            <CardContent className="p-5 space-y-5">
              {preferenceRows.map(([key, label, description]) => (
                <div key={key} className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{label}</p>
                    <p className="text-xs text-gray-400">{description}</p>
                  </div>
                  <Switch
                    checked={preferenceForm[key]}
                    onCheckedChange={(checked) =>
                      setPreferenceForm({ ...preferenceForm, [key]: checked })
                    }
                  />
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="mt-4">
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-base">Change Password</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1.5">
                <Label>Current Password</Label>
                <Input type="password" value={password.current} onChange={(event) => setPassword({ ...password, current: event.target.value })} />
              </div>
              <div className="space-y-1.5">
                <Label>New Password</Label>
                <Input type="password" value={password.next} onChange={(event) => setPassword({ ...password, next: event.target.value })} />
              </div>
              <div className="space-y-1.5">
                <Label>Confirm New Password</Label>
                <Input type="password" value={password.confirm} onChange={(event) => setPassword({ ...password, confirm: event.target.value })} />
              </div>
              <Button onClick={changePassword} disabled={saving}>
                Update Password
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="flex justify-end">
        <Button onClick={saveSettings} disabled={saving} className="gap-2 bg-green-600 hover:bg-green-700 min-w-32">
          <Save className="h-4 w-4" />
          {saving ? "Saving..." : "Save Changes"}
        </Button>
      </div>
    </div>
  );
}
