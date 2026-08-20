import "server-only";

import { redirect } from "next/navigation";
import type { SupabaseClient } from "@supabase/supabase-js";
import { createClient } from "@/lib/supabase/server";
import type {
  DoctorDirectoryItem,
  Medication,
} from "@/lib/patient-types";
import type { Database, Json } from "@/types/database";

export async function requirePatientContext() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  const [{ data: profile }, { data: patient }] = await Promise.all([
    supabase.from("profiles").select("*").eq("id", user.id).single(),
    supabase.from("patients").select("*").eq("profile_id", user.id).single(),
  ]);

  if (!profile) {
    redirect("/login");
  }

  if (profile.role !== "patient") {
    redirect(`/${profile.role}`);
  }

  if (!patient) {
    throw new Error("Patient profile is missing for the authenticated account.");
  }

  return {
    supabase,
    user,
    profile,
    patient,
  };
}

export async function getDoctorDirectory(
  supabase: SupabaseClient<Database>,
) {
  const { data, error } = await supabase.rpc("list_active_doctors");

  if (error) {
    throw new Error(`Unable to load doctors: ${error.message}`);
  }

  return data ?? [];
}

export function doctorMap(doctors: DoctorDirectoryItem[]) {
  return new Map(doctors.map((doctor) => [doctor.doctor_id, doctor]));
}

export function doctorName(doctor?: DoctorDirectoryItem) {
  return doctor ? `Dr. ${doctor.full_name}` : "Healthcare provider";
}

export function parseMedications(value: Json): Medication[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      return [];
    }

    const name = typeof item.name === "string" ? item.name.trim() : "";
    if (!name) {
      return [];
    }

    return [{
      name,
      dosage: typeof item.dosage === "string" ? item.dosage : undefined,
      morning: item.morning === true,
      evening: item.evening === true,
      withFood: item.withFood === true || item.with_food === true,
    }];
  });
}

export function prescriptionValidityPercent(
  createdAt: string,
  validUntil: string | null,
) {
  if (!validUntil) {
    return 100;
  }

  const start = new Date(createdAt).getTime();
  const end = new Date(`${validUntil}T23:59:59`).getTime();
  const now = Date.now();

  if (end <= start || now >= end) {
    return 0;
  }

  return Math.max(
    0,
    Math.min(100, Math.round(((end - now) / (end - start)) * 100)),
  );
}

export function firstName(fullName: string) {
  return fullName.trim().split(/\s+/)[0] || "Patient";
}
