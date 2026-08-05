import { NextResponse } from "next/server";
import { getN8nHealth } from "@/lib/n8n";

export async function GET() {
  const health = await getN8nHealth();
  return NextResponse.json(health, {
    status: health.status === "ready" ? 200 : 503,
  });
}
