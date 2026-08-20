const OLLAMA_URL = (
  process.env.OLLAMA_URL || "http://localhost:11434"
).replace(/\/+$/, "");
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "llama3.2";
const OLLAMA_TIMEOUT_MS = 120000;

export interface OllamaMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface OllamaChatResponse {
  message?: {
    content?: string;
  };
  error?: string;
}

interface OllamaTagsResponse {
  models?: Array<{
    model?: string;
    name?: string;
  }>;
}

export interface OllamaHealth {
  status: "ready" | "model_missing" | "offline";
  model: string;
  availableModels: string[];
  message: string;
}

function normalizeModelName(model: string) {
  return model.replace(/:latest$/, "");
}

async function getErrorMessage(response: Response) {
  try {
    const data = (await response.json()) as { error?: string };
    return data.error || response.statusText;
  } catch {
    return response.statusText;
  }
}

export async function getOllamaHealth(): Promise<OllamaHealth> {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/tags`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      return {
        status: "offline",
        model: OLLAMA_MODEL,
        availableModels: [],
        message: `Ollama returned ${response.status}: ${await getErrorMessage(response)}`,
      };
    }

    const data = (await response.json()) as OllamaTagsResponse;
    const availableModels = (data.models ?? [])
      .map((item) => item.model || item.name || "")
      .filter(Boolean);
    const configuredModel = normalizeModelName(OLLAMA_MODEL);
    const modelAvailable = availableModels.some(
      (model) => normalizeModelName(model) === configuredModel,
    );

    if (!modelAvailable) {
      return {
        status: "model_missing",
        model: OLLAMA_MODEL,
        availableModels,
        message: `Ollama is running, but ${OLLAMA_MODEL} is not installed.`,
      };
    }

    return {
      status: "ready",
      model: OLLAMA_MODEL,
      availableModels,
      message: "Ollama is running and the configured model is available.",
    };
  } catch (error) {
    const reason =
      error instanceof Error ? error.message : "Unable to reach Ollama.";

    return {
      status: "offline",
      model: OLLAMA_MODEL,
      availableModels: [],
      message: `Unable to reach Ollama at ${OLLAMA_URL}. ${reason}`,
    };
  }
}

export async function ollamaChat(
  messages: OllamaMessage[],
  model = OLLAMA_MODEL,
): Promise<string> {
  const response = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, stream: false }),
    cache: "no-store",
    signal: AbortSignal.timeout(OLLAMA_TIMEOUT_MS),
  });

  if (!response.ok) {
    throw new Error(
      `Ollama error ${response.status}: ${await getErrorMessage(response)}`,
    );
  }

  const data = (await response.json()) as OllamaChatResponse;
  const content = data.message?.content?.trim();

  if (!content) {
    throw new Error(data.error || "Ollama returned an empty response.");
  }

  return content;
}

export async function generatePatientSummary(patientData: {
  name: string;
  age: number;
  conditions: string[];
  medications: string[];
  recentVisit?: string;
}): Promise<string> {
  const messages: OllamaMessage[] = [
    {
      role: "system",
      content:
        "You are a clinical documentation assistant. Use only the supplied data. Generate a concise, professional patient summary in 2-3 sentences, focusing on health status and recent changes. Do not invent findings. The summary must be reviewed by a licensed clinician.",
    },
    {
      role: "user",
      content: `Generate a summary for patient: ${JSON.stringify(patientData)}`,
    },
  ];
  return ollamaChat(messages);
}

export async function aiDiagnosisSupport(symptoms: string[]): Promise<string> {
  const messages: OllamaMessage[] = [
    {
      role: "system",
      content:
        "You provide clinical decision support to licensed clinicians. Use only the supplied symptoms. Return possible differential diagnoses, brief supporting reasoning, recommended investigations, and urgent red flags. Do not present a definitive diagnosis or invent patient facts. End with a reminder that a clinician must independently verify the result.",
    },
    {
      role: "user",
      content: `Patient presents with: ${symptoms.join(", ")}. What are the differential diagnoses and recommended investigations?`,
    },
  ];
  return ollamaChat(messages);
}

export async function generateReport(data: {
  type: string;
  period: string;
  metrics: Record<string, unknown>;
}): Promise<string> {
  const messages: OllamaMessage[] = [
    {
      role: "system",
      content:
        "You are a healthcare analytics assistant. Generate a clear, professional report using only the supplied metrics. Use structured headings, identify data limitations, and do not invent statistics.",
    },
    {
      role: "user",
      content: `Generate a ${data.type} report for ${data.period}: ${JSON.stringify(data.metrics)}`,
    },
  ];
  return ollamaChat(messages);
}
