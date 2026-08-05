import { NextRequest, NextResponse } from "next/server";
import { requireApiRole } from "@/lib/api-auth";
import { generateReport } from "@/lib/ollama";

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
    !("type" in body) ||
    typeof body.type !== "string" ||
    body.type.trim().length === 0 ||
    body.type.length > 100 ||
    !("period" in body) ||
    typeof body.period !== "string" ||
    body.period.trim().length === 0 ||
    body.period.length > 100 ||
    !("metrics" in body) ||
    !body.metrics ||
    typeof body.metrics !== "object" ||
    Array.isArray(body.metrics) ||
    JSON.stringify(body.metrics).length > 20000
  ) {
    return NextResponse.json(
      { error: "A valid report type, period, and metrics object are required." },
      { status: 400 },
    );
  }

  try {
    const result = await generateReport({
      type: body.type.trim(),
      period: body.period.trim(),
      metrics: body.metrics as Record<string, unknown>,
    });
    return NextResponse.json({ result });
  } catch (error) {
    console.error("AI report error:", error);
    return NextResponse.json(
      {
        error:
          "AI service unavailable. Run `npm run ai:check` and verify Ollama is ready.",
      },
      { status: 503 },
    );
  }
}
