import { NextResponse } from "next/server";
import { getOllamaHealth } from "@/lib/ollama";

export async function GET() {
  const health = await getOllamaHealth();
  return NextResponse.json(health, {
    status: health.status === "offline" ? 503 : 200,
  });
}
