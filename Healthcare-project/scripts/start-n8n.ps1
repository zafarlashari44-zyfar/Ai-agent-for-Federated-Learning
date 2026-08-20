$ErrorActionPreference = "Stop"

$projectRoot = Split-Path $PSScriptRoot -Parent
$envFile = Join-Path $projectRoot ".env.local"

if (-not (Test-Path $envFile)) {
  throw ".env.local was not found at $envFile"
}

Get-Content $envFile | ForEach-Object {
  $line = $_.Trim()
  if (-not $line -or $line.StartsWith("#")) {
    return
  }

  $separator = $line.IndexOf("=")
  if ($separator -lt 1) {
    return
  }

  $name = $line.Substring(0, $separator).Trim()
  $value = $line.Substring($separator + 1).Trim().Trim('"').Trim("'")
  [Environment]::SetEnvironmentVariable($name, $value, "Process")
}

if (-not $env:N8N_WEBHOOK_SECRET) {
  $env:N8N_WEBHOOK_SECRET = $env:N8N_API_KEY
}

if (-not $env:N8N_WEBHOOK_SECRET) {
  throw "Set N8N_WEBHOOK_SECRET in .env.local before starting n8n."
}

if (-not $env:MEDICARE_APP_URL) {
  $env:MEDICARE_APP_URL = $env:NEXT_PUBLIC_APP_URL
}

if (-not $env:MEDICARE_APP_URL) {
  $env:MEDICARE_APP_URL = "http://localhost:3000"
}

$env:N8N_USER_FOLDER = Join-Path $projectRoot ".n8n-data"
$env:N8N_HOST = "localhost"
$env:N8N_PORT = "5678"
$env:N8N_PROTOCOL = "http"
$env:N8N_SECURE_COOKIE = "false"
$env:N8N_DIAGNOSTICS_ENABLED = "false"
$env:N8N_PERSONALIZATION_ENABLED = "false"
$env:N8N_BLOCK_ENV_ACCESS_IN_NODE = "false"
$env:EXECUTIONS_DATA_PRUNE = "true"
$env:EXECUTIONS_DATA_MAX_AGE = "168"
$env:GENERIC_TIMEZONE = "Asia/Kolkata"
$env:TZ = "Asia/Kolkata"

Write-Host "Starting n8n at http://localhost:5678"
Write-Host "Persistent data: $env:N8N_USER_FOLDER"
Write-Host "Application callback: $env:MEDICARE_APP_URL/api/webhooks/n8n"

$n8n = Get-Command n8n -ErrorAction SilentlyContinue
if ($n8n) {
  & n8n start
} else {
  & npx --yes n8n start
}
