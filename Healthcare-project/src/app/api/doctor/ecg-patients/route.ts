import { NextResponse } from "next/server";

import { requireApiRole } from "@/lib/api-auth";
import { createAdminClient } from "@/lib/supabase/admin";

export const dynamic = "force-dynamic";

type PatientProfile = {
  full_name: string | null;
  email: string | null;
};

function firstItem<T>(
  value: T | T[] | null | undefined,
): T | null {
  return Array.isArray(value)
    ? value[0] ?? null
    : value ?? null;
}

export async function GET() {
  const auth = await requireApiRole(["doctor", "admin"]);

  if (!auth.ok) {
    return NextResponse.json(
      { error: auth.error },
      { status: auth.status },
    );
  }

  const supabase = createAdminClient();

  const { data, error } = await supabase
    .from("patients")
    .select(`
      id,
      profiles (
        full_name,
        email
      )
    `)
    .order("created_at", { ascending: false });

  if (error) {
    return NextResponse.json(
      {
        error: "Unable to load patients.",
        details: error.message,
      },
      { status: 500 },
    );
  }

  const patients = (data ?? []).map((patient) => {
    const profile = firstItem(
      patient.profiles as
        | PatientProfile
        | PatientProfile[]
        | null,
    );

    return {
      id: patient.id,
      name: profile?.full_name ?? "Unknown Patient",
      email: profile?.email ?? null,
    };
  });

  return NextResponse.json({ patients });
}
