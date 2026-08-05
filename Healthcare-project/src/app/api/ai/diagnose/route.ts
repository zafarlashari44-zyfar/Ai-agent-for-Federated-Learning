import { NextRequest, NextResponse } from "next/server";
import { requireApiRole } from "@/lib/api-auth";
import { aiDiagnosisSupport } from "@/lib/ollama";

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

  if (!body || typeof body !== "object" || !("symptoms" in body)) {
    return NextResponse.json(
      { error: "A symptoms array is required." },
      { status: 400 },
    );
  }

  const rawSymptoms = body.symptoms;
  if (!Array.isArray(rawSymptoms) || rawSymptoms.length === 0) {
    return NextResponse.json(
      { error: "Provide at least one symptom." },
      { status: 400 },
    );
  }

  const symptoms = rawSymptoms
    .filter((symptom): symptom is string => typeof symptom === "string")
    .map((symptom) => symptom.trim())
    .filter(Boolean);

  if (
    symptoms.length !== rawSymptoms.length ||
    symptoms.length > 20 ||
    symptoms.some((symptom) => symptom.length > 200)
  ) {
    return NextResponse.json(
      {
        error:
          "Symptoms must contain 1-20 non-empty strings of at most 200 characters.",
      },
      { status: 400 },
    );
  }

  try {
    const result = await aiDiagnosisSupport(symptoms);
    return NextResponse.json({ result });
  } catch (error) {
    console.error("AI diagnose error:", error);
    return NextResponse.json(
      {
        error:
          "AI service unavailable. Run `npm run ai:check` and verify Ollama is ready.",
      },
      { status: 503 },
    );
  }
}
