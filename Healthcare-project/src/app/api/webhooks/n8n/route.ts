import { timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";
import { createAdminClient } from "@/lib/supabase/admin";

type EventData = Record<string, unknown>;

function secureEqual(actual: string, expected: string) {
  const actualBuffer = Buffer.from(actual);
  const expectedBuffer = Buffer.from(expected);

  return (
    actualBuffer.length === expectedBuffer.length &&
    timingSafeEqual(actualBuffer, expectedBuffer)
  );
}

function requiredString(
  data: EventData,
  key: string,
  maxLength = 200,
) {
  const value = data[key];
  if (
    typeof value !== "string" ||
    value.trim().length === 0 ||
    value.length > maxLength
  ) {
    throw new Error(`${key} must be a non-empty string.`);
  }
  return value.trim();
}

function requiredUserId(data: EventData) {
  const userId = requiredString(data, "userId", 36);
  if (
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
      userId,
    )
  ) {
    throw new Error("userId must be a valid UUID.");
  }
  return userId;
}

export async function POST(req: NextRequest) {
  try {
    const expectedSecret =
      process.env.N8N_WEBHOOK_SECRET || process.env.N8N_API_KEY;
    const authorization = req.headers.get("authorization") || "";
    const expectedAuthorization = `Bearer ${expectedSecret || ""}`;

    if (
      !expectedSecret ||
      !secureEqual(authorization, expectedAuthorization)
    ) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    let payload: unknown;
    try {
      payload = await req.json();
    } catch {
      return NextResponse.json(
        { error: "Request body must be valid JSON." },
        { status: 400 },
      );
    }

    if (
      !payload ||
      typeof payload !== "object" ||
      !("event" in payload) ||
      typeof payload.event !== "string" ||
      !("data" in payload) ||
      !payload.data ||
      typeof payload.data !== "object" ||
      Array.isArray(payload.data)
    ) {
      return NextResponse.json(
        { error: "Invalid webhook payload." },
        { status: 400 },
      );
    }

    const event = payload.event;
    const data = payload.data as EventData;
    const supabase = createAdminClient();
    const userId = requiredUserId(data);

    let notification: {
      user_id: string;
      title: string;
      message: string;
      type: string;
      link?: string;
    };

    switch (event) {
      case "appointment.reminder":
        notification = {
          user_id: userId,
          title: "Appointment Reminder",
          message: `You have an appointment on ${requiredString(data, "appointmentDate", 100)} at ${requiredString(data, "appointmentTime", 50)}.`,
          type: "appointment",
          link: "/patient/appointments",
        };
        break;

      case "prescription.refill":
        notification = {
          user_id: userId,
          title: "Prescription Refill Due",
          message: `${requiredString(data, "medication")} needs a refill by ${requiredString(data, "refillDate", 100)}.`,
          type: "reminder",
          link: "/patient/prescriptions",
        };
        break;

      case "prescription.created":
        notification = {
          user_id: userId,
          title: "New Prescription",
          message: `A prescription for ${requiredString(data, "medication")} is ready to review.`,
          type: "reminder",
          link: "/patient/prescriptions",
        };
        break;

      case "report.ready":
        notification = {
          user_id: userId,
          title: "Report Ready",
          message: `Your ${requiredString(data, "reportType")} report is ready for review.`,
          type: "report",
          link: "/patient/records",
        };
        break;

      case "patient.registered":
        notification = {
          user_id: userId,
          title: "New Patient Registered",
          message: `${requiredString(data, "patientName")} has joined the platform.`,
          type: "info",
          link: "/doctor/patients",
        };
        break;

      case "billing.alert":
        notification = {
          user_id: userId,
          title: "Billing Reminder",
          message: `Invoice ${requiredString(data, "invoiceNumber", 100)} is due on ${requiredString(data, "dueDate", 100)}.`,
          type: "alert",
        };
        break;

      default:
        return NextResponse.json(
          { error: "Unknown event type." },
          { status: 400 },
        );
    }

    const { error } = await supabase
      .from("notifications")
      .insert(notification);

    if (error) {
      throw new Error(error.message);
    }

    return NextResponse.json({ success: true, event });
  } catch (error) {
    console.error("n8n webhook error:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Webhook processing failed.",
      },
      { status: 500 },
    );
  }
}
