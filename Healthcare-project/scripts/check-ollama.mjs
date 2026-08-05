const ollamaUrl = (
  process.env.OLLAMA_URL || "http://localhost:11434"
).replace(/\/+$/, "");
const model = process.env.OLLAMA_MODEL || "llama3.2";
const shouldGenerate = process.argv.includes("--generate");

function normalizeModelName(value) {
  return value.replace(/:latest$/, "");
}

async function main() {
  console.log(`Checking Ollama at ${ollamaUrl}`);

  let tagsResponse;
  try {
    tagsResponse = await fetch(`${ollamaUrl}/api/tags`, {
      signal: AbortSignal.timeout(5000),
    });
  } catch (error) {
    console.error("Ollama is not reachable.");
    console.error("Install or start Ollama, then run this command again.");
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
    return;
  }

  if (!tagsResponse.ok) {
    console.error(
      `Ollama returned ${tagsResponse.status} ${tagsResponse.statusText}.`,
    );
    process.exitCode = 1;
    return;
  }

  const tags = await tagsResponse.json();
  const models = (tags.models || [])
    .map((item) => item.model || item.name)
    .filter(Boolean);
  const configuredModel = normalizeModelName(model);
  const modelAvailable = models.some(
    (item) => normalizeModelName(item) === configuredModel,
  );

  console.log(`Installed models: ${models.join(", ") || "none"}`);

  if (!modelAvailable) {
    console.error(`Configured model "${model}" is not installed.`);
    console.error(`Run: ollama pull ${model}`);
    process.exitCode = 1;
    return;
  }

  console.log(`Configured model "${model}" is ready.`);

  if (!shouldGenerate) {
    return;
  }

  console.log("Running a short generation test...");
  const chatResponse = await fetch(`${ollamaUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: "user",
          content: "Reply exactly with: Ollama works",
        },
      ],
      stream: false,
    }),
    signal: AbortSignal.timeout(120000),
  });

  if (!chatResponse.ok) {
    const errorBody = await chatResponse.text();
    throw new Error(
      `Generation failed: ${chatResponse.status} ${errorBody}`,
    );
  }

  const result = await chatResponse.json();
  const content = result.message?.content?.trim();
  if (!content) {
    throw new Error("Generation completed without a text response.");
  }

  console.log(`Model response: ${content}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
