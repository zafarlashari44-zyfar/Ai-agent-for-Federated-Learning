import Link from "next/link";
import { notFound } from "next/navigation";
import {
  ArrowLeft,
  Calendar,
  ClipboardList,
  Mail,
  Phone,
  Shield,
  User,
} from "lucide-react";

import { createAdminClient } from "@/lib/supabase/admin";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { getInitials, formatDate } from "@/lib/utils";

type PageProps = {
  params: Promise<{
    id: string;
  }>;
};

type Profile = {
  full_name: string | null;
  email: string | null;
  phone: string | null;
  gender: string | null;
  date_of_birth: string | null;
  role: string | null;
};

type Patient = {
  id: string;
  blood_type: string | null;
  emergency_contact_name: string | null;
  emergency_contact_phone: string | null;
  insurance_provider: string | null;
  created_at: string | null;
  profiles: Profile | Profile[] | null;
};

function firstItem<T>(value: T | T[] | null | undefined): T | null {
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

export default async function DoctorPatientDetailsPage({ params }: PageProps) {
  const { id } = await params;
  const supabase = createAdminClient();

  const { data, error } = await supabase
    .from("patients")
    .select(
      `
      id,
      blood_type,
      emergency_contact_name,
      emergency_contact_phone,
      insurance_provider,
      created_at,
      profiles (
        full_name,
        email,
        phone,
        gender,
        date_of_birth,
        role
      )
    `
    )
    .eq("id", id)
    .single();

  if (error || !data) {
    notFound();
  }

  const patient = data as Patient;
  const profile = firstItem(patient.profiles);

  if (profile?.role !== "patient") {
    notFound();
  }

  const patientName = profile?.full_name ?? "Unknown Patient";

  const infoItems = [
    {
      label: "Email",
      value: profile?.email ?? "Not set",
      icon: Mail,
    },
    {
      label: "Phone",
      value: profile?.phone ?? "Not set",
      icon: Phone,
    },
    {
      label: "Gender",
      value: profile?.gender ?? "Not set",
      icon: User,
    },
    {
      label: "Date of Birth",
      value: profile?.date_of_birth
        ? formatDate(profile.date_of_birth)
        : "Not set",
      icon: Calendar,
    },
    {
      label: "Blood Type",
      value: patient.blood_type ?? "Not set",
      icon: ClipboardList,
    },
    {
      label: "Insurance",
      value: patient.insurance_provider ?? "Not set",
      icon: Shield,
    },
  ];

  return (
    <div className="space-y-6">
      <Link
        href="/doctor/patients"
        className="inline-flex items-center gap-2 text-sm font-medium text-gray-500 hover:text-gray-900"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to patients
      </Link>

      <Card>
        <CardContent className="flex flex-col gap-4 p-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-4">
            <Avatar className="h-14 w-14">
              <AvatarFallback className="bg-blue-100 text-lg text-blue-700">
                {getInitials(patientName)}
              </AvatarFallback>
            </Avatar>

            <div>
              <h2 className="text-xl font-bold text-gray-900">
                {patientName}
              </h2>
              <p className="text-sm text-gray-500">
                Patient ID: {patient.id}
              </p>
            </div>
          </div>

          <Badge variant="success">patient</Badge>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Patient Information</CardTitle>
          </CardHeader>

          <CardContent className="space-y-4">
            {infoItems.map((item) => {
              const Icon = item.icon;

              return (
                <div
                  key={item.label}
                  className="flex items-start gap-3 rounded-lg border border-gray-100 p-3"
                >
                  <Icon className="mt-0.5 h-4 w-4 text-blue-600" />
                  <div>
                    <p className="text-xs font-medium uppercase text-gray-400">
                      {item.label}
                    </p>
                    <p className="text-sm text-gray-900">{item.value}</p>
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Emergency Contact</CardTitle>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="rounded-lg border border-gray-100 p-4">
              <p className="text-xs font-medium uppercase text-gray-400">
                Name
              </p>
              <p className="text-sm text-gray-900">
                {patient.emergency_contact_name ?? "Not set"}
              </p>
            </div>

            <div className="rounded-lg border border-gray-100 p-4">
              <p className="text-xs font-medium uppercase text-gray-400">
                Phone
              </p>
              <p className="text-sm text-gray-900">
                {patient.emergency_contact_phone ?? "Not set"}
              </p>
            </div>

            <div className="rounded-lg border border-gray-100 p-4">
              <p className="text-xs font-medium uppercase text-gray-400">
                Registered
              </p>
              <p className="text-sm text-gray-900">
                {patient.created_at ? formatDate(patient.created_at) : "Not set"}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}