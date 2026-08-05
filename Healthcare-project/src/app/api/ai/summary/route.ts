import { NextRequest, NextResponse } from "next/server";
import { requireApiRole } from "@/lib/api-auth";
import { generatePatientSummary } from "@/lib/ollama";

export async function POST(req: NextRequest) {
  const auth = await requireApiRole(["doctor", "admin"]);
  if (!auth.ok) {
    return NextResponse.json(
      { error: auth.error },
      { status: auth.status },
    );
  }

  let body: unknown;

  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  if (
    !body ||
    typeof body !== "object" ||
    !("name" in body) ||
    typeof body.name !== "string" ||
    body.name.trim().length === 0 ||
    body.name.length > 200 ||
    !("age" in body) ||
    typeof body.age !== "number" ||
    !Number.isInteger(body.age) ||
    body.age < 0 ||
    body.age > 130 ||
    !("conditions" in body) ||
    !Array.isArray(body.conditions) ||
    !("medications" in body) ||
    !Array.isArray(body.medications)
  ) {
    return NextResponse.json(
      {
        error:
          "A valid name, age, conditions array, and medications array are required.",
      },
      { status: 400 },
    );
  }

  const conditions = body.conditions;
  const medications = body.medications;
  const validList = (items: unknown[]) =>
    items.length <= 50 &&
    items.every(
      (item) =>
        typeof item === "string" &&
        item.trim().length > 0 &&
        item.length <= 200,
    );

  if (!validList(conditions) || !validList(medications)) {
    return NextResponse.json(
      {
        error:
          "Conditions and medications must contain at most 50 non-empty strings.",
      },
      { status: 400 },
    );
  }

  const recentVisit =
    "recentVisit" in body && typeof body.recentVisit === "string"
      ? body.recentVisit.trim()
      : undefined;

  if (recentVisit && recentVisit.length > 2000) {
    return NextResponse.json(
      { error: "Recent visit notes must be at most 2000 characters." },
      { status: 400 },
    );
  }

  try {
    const result = await generatePatientSummary({
      name: body.name.trim(),
      age: body.age,
      conditions: conditions.map((item) => (item as string).trim()),
      medications: medications.map((item) => (item as string).trim()),
      recentVisit,
    });
    return NextResponse.json({ result });
  } catch (error) {
    console.error("AI summary error:", error);
    return NextResponse.json(
      {
        error:
          "AI service unavailable. Run `npm run ai:check` and verify Ollama is ready.",
      },
      { status: 503 },
    );
  }
}
