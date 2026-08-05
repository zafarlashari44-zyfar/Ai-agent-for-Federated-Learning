const n8nUrl = (process.env.N8N_URL || "http://localhost:5678").replace(
  /\/+$/,
  "",
);
const webhookPath =
  process.env.N8N_WEBHOOK_PATH || "medicare-events";
const webhookSecret =
  process.env.N8N_WEBHOOK_SECRET || process.env.N8N_API_KEY || "";
const shouldTestWebhook = process.argv.includes("--webhook");

async function main() {
  console.log(`Checking n8n at ${n8nUrl}`);

  let healthResponse;
  try {
    healthResponse = await fetch(`${n8nUrl}/healthz`, {
      signal: AbortSignal.timeout(5000),
    });
  } catch (error) {
    console.error("n8n is not reachable.");
    console.error("Start it with: npm run automation:start");
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
    return;
  }

  if (!healthResponse.ok) {
    console.error(
      `n8n returned ${healthResponse.status} ${healthResponse.statusText}.`,
    );
    process.exitCode = 1;
    return;
  }

  console.log("n8n health check passed.");

  if (!shouldTestWebhook) {
    return;
  }

  if (!webhookSecret) {
    console.error(
      "N8N_WEBHOOK_SECRET is missing. Add it to .env.local first.",
    );
    process.exitCode = 1;
    return;
  }

  console.log(`Testing production webhook /webhook/${webhookPath}...`);
  const webhookResponse = await fetch(
    `${n8nUrl}/webhook/${webhookPath}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${webhookSecret}`,
      },
      body: JSON.stringify({
        event: "automation.health",
        data: {
          checkedAt: new Date().toISOString(),
        },
      }),
      signal: AbortSignal.timeout(15000),
    },
  );

  const responseText = await webhookResponse.text();
  if (!webhookResponse.ok) {
    throw new Error(
      `Webhook test failed: ${webhookResponse.status} ${responseText}`,
    );
  }

  console.log(`Webhook response: ${responseText}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
