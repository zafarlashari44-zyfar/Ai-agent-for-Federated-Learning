"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  CheckCircle,
  Eye,
  EyeOff,
  ShieldCheck,
  Stethoscope,
  UserPlus,
} from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type RequestedRole = "patient" | "doctor" | "admin";

export default function RegisterPage() {
  const router = useRouter();
  const [form, setForm] = useState({
    fullName: "",
    email: "",
    role: "patient" as RequestedRole,
    specialization: "",
    licenseNumber: "",
    password: "",
    confirmPassword: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  function update(key: keyof typeof form, value: string) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function handleRegister(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    if (form.password !== form.confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    if (form.password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }
    if (
      form.role === "doctor" &&
      (!form.specialization.trim() || !form.licenseNumber.trim())
    ) {
      setError("Doctors must enter a specialization and license number");
      return;
    }

    setLoading(true);
    try {
      if (form.role === "patient") {
        const response = await fetch("/api/auth/register/patient", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            fullName: form.fullName,
            email: form.email,
            password: form.password,
          }),
        });
        const result = (await response.json()) as { error?: string };

        if (!response.ok) {
          setError(result.error || "Unable to create your account.");
          return;
        }
      } else {
        const supabase = createClient();
        const { data, error: authError } = await supabase.auth.signUp({
          email: form.email.trim(),
          password: form.password,
          options: {
            data: {
              full_name: form.fullName.trim(),
              requested_role: form.role,
              doctor_specialization:
                form.role === "doctor" ? form.specialization.trim() : null,
              doctor_license_number:
                form.role === "doctor" ? form.licenseNumber.trim() : null,
            },
          },
        });

        if (authError) {
          setError(authError.message);
          return;
        }

        if (!data.user) {
          setError("Unable to create your account.");
          return;
        }
      }

      setSuccess(true);
      setTimeout(() => router.push("/login"), 4500);
    } catch {
      setError("Unable to create your account right now.");
    } finally {
      setLoading(false);
    }
  }

  if (success) {
    return (
      <div className="w-full max-w-md text-center">
        <Card className="border-gray-100 p-8 shadow-xl">
          <CheckCircle className="mx-auto mb-4 h-12 w-12 text-green-500" />
          <h2 className="mb-2 text-xl font-bold text-gray-900">
            Account Created!
          </h2>
          <p className="text-sm text-gray-500">
            {form.role === "patient"
              ? "Your patient account is ready. No email verification is required; you can sign in now."
              : `Verify your email, then wait for an administrator to approve your ${form.role} access before signing in.`}
          </p>
          <p className="mt-2 text-xs text-gray-400">
            Redirecting you to login...
          </p>
        </Card>
      </div>
    );
  }

  const privilegedRole = form.role === "doctor" || form.role === "admin";

  return (
    <div className="w-full max-w-md">
      <Card className="border-gray-100 shadow-xl">
        <CardHeader className="space-y-1 pb-4 text-center">
          <CardTitle className="text-2xl font-bold text-gray-900">
            Create an account
          </CardTitle>
          <CardDescription>Join MediCare Pro today</CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          {error && (
            <div className="flex items-center gap-2 rounded-lg bg-red-50 p-3 text-sm text-red-600">
              <AlertCircle className="h-4 w-4 shrink-0" />
              {error}
            </div>
          )}

          <form onSubmit={handleRegister} className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="full-name">Full Name</Label>
              <Input
                id="full-name"
                placeholder="Dr. John Smith"
                value={form.fullName}
                onChange={(event) => update("fullName", event.target.value)}
                required
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="register-email">Email address</Label>
              <Input
                id="register-email"
                type="email"
                placeholder="you@example.com"
                value={form.email}
                onChange={(event) => update("email", event.target.value)}
                required
              />
            </div>

            <div className="space-y-1.5">
              <Label>I am registering as</Label>
              <Select
                value={form.role}
                onValueChange={(value: RequestedRole) => update("role", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="patient">Patient</SelectItem>
                  <SelectItem value="doctor">Doctor</SelectItem>
                  <SelectItem value="admin">Administrator</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {form.role === "doctor" && (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <Label htmlFor="specialization">Specialization</Label>
                  <Input
                    id="specialization"
                    placeholder="Cardiology"
                    value={form.specialization}
                    onChange={(event) =>
                      update("specialization", event.target.value)
                    }
                    required
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="license-number">License Number</Label>
                  <Input
                    id="license-number"
                    placeholder="MED-12345"
                    value={form.licenseNumber}
                    onChange={(event) =>
                      update("licenseNumber", event.target.value)
                    }
                    required
                  />
                </div>
              </div>
            )}

            <div className="flex items-start gap-2 rounded-lg bg-blue-50 px-3 py-2.5 text-xs text-gray-600">
              {form.role === "doctor" ? (
                <Stethoscope className="h-4 w-4 shrink-0 text-blue-600" />
              ) : form.role === "admin" ? (
                <ShieldCheck className="h-4 w-4 shrink-0 text-blue-600" />
              ) : (
                <UserPlus className="h-4 w-4 shrink-0 text-blue-600" />
              )}
              <p>
                {privilegedRole
                  ? `${form.role === "doctor" ? "Doctor" : "Administrator"} access requires email verification and administrator approval.`
                  : "Patient accounts do not require email verification."}
              </p>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="register-password">Password</Label>
              <div className="relative">
                <Input
                  id="register-password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Min. 8 characters"
                  value={form.password}
                  onChange={(event) => update("password", event.target.value)}
                  required
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((current) => !current)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="confirm-password">Confirm Password</Label>
              <Input
                id="confirm-password"
                type="password"
                placeholder="Repeat your password"
                value={form.confirmPassword}
                onChange={(event) =>
                  update("confirmPassword", event.target.value)
                }
                required
              />
            </div>

            <Button
              type="submit"
              className="w-full gap-2"
              disabled={loading}
            >
              {loading ? (
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              ) : (
                <UserPlus className="h-4 w-4" />
              )}
              {loading ? "Creating account..." : "Create Account"}
            </Button>
          </form>
        </CardContent>

        <CardFooter className="justify-center pt-0">
          <p className="text-sm text-gray-500">
            Already have an account?{" "}
            <Link
              href="/login"
              className="font-medium text-blue-600 hover:underline"
            >
              Sign in
            </Link>
          </p>
        </CardFooter>
      </Card>
    </div>
  );
}
