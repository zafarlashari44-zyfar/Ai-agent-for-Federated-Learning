const N8N_URL = (process.env.N8N_URL || "http://localhost:5678").replace(
  /\/+$/,
  "",
);
const N8N_WEBHOOK_PATH =
  process.env.N8N_WEBHOOK_PATH || "medicare-events";
const N8N_WEBHOOK_SECRET =
  process.env.N8N_WEBHOOK_SECRET || process.env.N8N_API_KEY || "";

export type AutomationEvent =
  | {
      event: "appointment.reminder";
      data: {
        userId: string;
        appointmentDate: string;
        appointmentTime: string;
      };
    }
  | {
      event: "prescription.refill";
      data: {
        userId: string;
        medication: string;
        refillDate: string;
      };
    }
  | {
      event: "prescription.created";
      data: {
        userId: string;
        medication: string;
      };
    }
  | {
      event: "report.ready";
      data: {
        userId: string;
        reportType: string;
      };
    }
  | {
      event: "patient.registered";
      data: {
        userId: string;
        patientName: string;
      };
    }
  | {
      event: "billing.alert";
      data: {
        userId: string;
        invoiceNumber: string;
        dueDate: string;
      };
    };

export interface N8nHealth {
  status: "ready" | "offline";
  url: string;
  webhookPath: string;
  webhookSecretConfigured: boolean;
  message: string;
}

async function responseMessage(response: Response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    try {
      const body = (await response.json()) as {
        message?: string;
        error?: string;
      };
      return body.message || body.error || response.statusText;
    } catch {
      return response.statusText;
    }
  }

  const text = await response.text();
  return text.trim() || response.statusText;
}

export async function getN8nHealth(): Promise<N8nHealth> {
  try {
    const response = await fetch(`${N8N_URL}/healthz`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      return {
        status: "offline",
        url: N8N_URL,
        webhookPath: N8N_WEBHOOK_PATH,
        webhookSecretConfigured: Boolean(N8N_WEBHOOK_SECRET),
        message: `n8n returned ${response.status}: ${await responseMessage(response)}`,
      };
    }

    return {
      status: "ready",
      url: N8N_URL,
      webhookPath: N8N_WEBHOOK_PATH,
      webhookSecretConfigured: Boolean(N8N_WEBHOOK_SECRET),
      message: N8N_WEBHOOK_SECRET
        ? "n8n is running and webhook authentication is configured."
        : "n8n is running, but the webhook secret is not configured.",
    };
  } catch (error) {
    const reason =
      error instanceof Error ? error.message : "Unable to reach n8n.";

    return {
      status: "offline",
      url: N8N_URL,
      webhookPath: N8N_WEBHOOK_PATH,
      webhookSecretConfigured: Boolean(N8N_WEBHOOK_SECRET),
      message: `Unable to reach n8n at ${N8N_URL}. ${reason}`,
    };
  }
}

export async function triggerN8nEvent(payload: AutomationEvent) {
  if (!N8N_WEBHOOK_SECRET) {
    throw new Error(
      "N8N_WEBHOOK_SECRET is not configured. Set it in .env.local before triggering workflows.",
    );
  }

  const response = await fetch(
    `${N8N_URL}/webhook/${N8N_WEBHOOK_PATH}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${N8N_WEBHOOK_SECRET}`,
      },
      body: JSON.stringify(payload),
      cache: "no-store",
      signal: AbortSignal.timeout(15000),
    },
  );

  if (!response.ok) {
    throw new Error(
      `n8n webhook error ${response.status}: ${await responseMessage(response)}`,
    );
  }

  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json")
    ? response.json()
    : response.text();
}

export const workflows = {
  appointmentReminder: (
    data: Extract<
      AutomationEvent,
      { event: "appointment.reminder" }
    >["data"],
  ) => triggerN8nEvent({ event: "appointment.reminder", data }),

  prescriptionRefill: (
    data: Extract<
      AutomationEvent,
      { event: "prescription.refill" }
    >["data"],
  ) => triggerN8nEvent({ event: "prescription.refill", data }),

  prescriptionCreated: (
    data: Extract<
      AutomationEvent,
      { event: "prescription.created" }
    >["data"],
  ) => triggerN8nEvent({ event: "prescription.created", data }),

  reportReady: (
    data: Extract<AutomationEvent, { event: "report.ready" }>["data"],
  ) => triggerN8nEvent({ event: "report.ready", data }),

  patientRegistered: (
    data: Extract<
      AutomationEvent,
      { event: "patient.registered" }
    >["data"],
  ) => triggerN8nEvent({ event: "patient.registered", data }),

  billingAlert: (
    data: Extract<AutomationEvent, { event: "billing.alert" }>["data"],
  ) => triggerN8nEvent({ event: "billing.alert", data }),
};
