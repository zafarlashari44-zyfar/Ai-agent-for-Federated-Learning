import { NextResponse } from "next/server";
import { createAdminClient } from "@/lib/supabase/admin";

type PatientRegistration = {
  email?: unknown;
  fullName?: unknown;
  password?: unknown;
};

function validEmail(value: string) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

export async function POST(request: Request) {
  let payload: PatientRegistration;

  try {
    payload = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const fullName =
    typeof payload.fullName === "string" ? payload.fullName.trim() : "";
  const email =
    typeof payload.email === "string"
      ? payload.email.trim().toLowerCase()
      : "";
  const password =
    typeof payload.password === "string" ? payload.password : "";

  if (fullName.length < 2 || fullName.length > 100) {
    return NextResponse.json(
      { error: "Full name must be between 2 and 100 characters." },
      { status: 400 },
    );
  }

  if (!validEmail(email) || email.length > 254) {
    return NextResponse.json(
      { error: "Enter a valid email address." },
      { status: 400 },
    );
  }

  if (password.length < 8 || password.length > 128) {
    return NextResponse.json(
      { error: "Password must be between 8 and 128 characters." },
      { status: 400 },
    );
  }

  try {
    const supabase = createAdminClient();
    const { data, error } = await supabase.auth.admin.createUser({
      email,
      password,
      email_confirm: true,
      user_metadata: {
        full_name: fullName,
        requested_role: "patient",
      },
    });

    if (error) {
      const duplicateAccount =
        error.status === 422 ||
        /already (registered|exists)|duplicate/i.test(error.message);

      return NextResponse.json(
        {
          error: duplicateAccount
            ? "An account with this email already exists."
            : error.message,
        },
        { status: duplicateAccount ? 409 : 400 },
      );
    }

    return NextResponse.json({ userId: data.user.id }, { status: 201 });
  } catch (error) {
    console.error("Patient registration failed:", error);
    return NextResponse.json(
      { error: "Unable to create the patient account right now." },
      { status: 500 },
    );
  }
}
