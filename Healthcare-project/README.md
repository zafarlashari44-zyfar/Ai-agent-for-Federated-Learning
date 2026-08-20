# MediCare Pro — Healthcare Management System

A full-stack healthcare management platform with three role-based dashboards: **Doctor**, **Admin**, and **Patient**.

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 16.2.7 (App Router) |
| Styling | Tailwind CSS v4 + ShadCN UI |
| Database | Supabase (PostgreSQL) |
| Auth | Supabase Auth |
| File Storage | Supabase Storage |
| AI | Ollama (local) |
| Automation | n8n (local) |
| Deployment | Vercel |

---

## Getting Started

### 1. Install dependencies

```bash
cd healthcare-system
npm install
```

### 2. Set up environment variables

Copy `.env.example` to `.env.local` and fill in your values:

```bash
cp .env.example .env.local
```

```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project-id.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

N8N_URL=http://localhost:5678
N8N_WEBHOOK_PATH=medicare-events
N8N_WEBHOOK_SECRET=your-long-random-webhook-secret
N8N_API_KEY=

NEXT_PUBLIC_APP_URL=http://localhost:3000
```

> Get your Supabase keys from: **supabase.com → Project → Settings → API**

### 3. Set up the database

Link the Supabase CLI and apply the versioned migrations:

```bash
npx supabase login
npx supabase link --project-ref YOUR_PROJECT_REF
npm run db:push
npm run db:types
```

Database migrations are stored in `supabase/migrations/`.

### 4. Bootstrap the first administrator

Register the account normally, then promote it with the server-only role command:

```bash
npm run user:set-role -- admin@example.com admin
```

To create a doctor account, include specialization and license number:

```bash
npm run user:set-role -- doctor@example.com doctor "Cardiology" "LICENSE-123"
```

### 5. Run the development server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Demo Accounts

These Supabase Auth accounts are for demonstration and testing only. Do not
reuse these passwords for production or personal accounts.

| Role | Email | Password | Dashboard |
|---|---|---|---|
| Doctor | `doctor.demo@medicarepro.com` | `Doctor@12345` | `/doctor` |
| Admin | `admin.demo@medicarepro.com` | `Admin@12345` | `/admin` |
| Patient | `explocg@gmail.com` | `Testuser@01` | `/patient` |

---

## Dashboards

### Doctor Dashboard `/doctor`
- Patient list (50+ patients) with search and filters
- Appointment calendar and daily schedule
- AI-assisted diagnosis (via Ollama)
- Prescription management
- Reports with charts
- Analytics (weekly stats, performance radar)

### Admin Dashboard `/admin`
- System overview with revenue and health metrics
- Doctor and patient management
- Billing and invoices
- Role and permission management
- Activity logs
- Integrations settings (Ollama, n8n, Supabase)

### Patient Dashboard `/patient`
- Personal health overview and vitals
- Medical records (accordion view)
- Appointments (book, view upcoming/past)
- Active prescriptions with refill tracking
- Document uploads
- Messaging with doctors
- Health tracking charts

---

## Project Structure

```
src/
├── app/
│   ├── (auth)/           # Login, register, forgot-password
│   ├── (dashboard)/
│   │   ├── admin/        # Admin dashboard pages
│   │   ├── doctor/       # Doctor dashboard pages
│   │   └── patient/      # Patient dashboard pages
│   ├── api/
│   │   ├── ai/           # Ollama AI routes (diagnose, report, summary)
│   │   └── webhooks/n8n/ # n8n automation webhook
│   └── page.tsx          # Landing page
├── components/
│   ├── layout/           # Sidebar, TopNav, DashboardShell
│   └── ui/               # ShadCN-style UI components
├── lib/
│   ├── supabase/         # Browser + server Supabase clients
│   ├── ollama.ts         # Ollama AI helpers
│   └── n8n.ts            # n8n workflow triggers
├── types/
│   └── database.ts       # Full Supabase Database type
└── proxy.ts              # Auth routing (Next.js 16 middleware)
```

---

## AI Features (Ollama)

Ollama runs the AI model locally on the same Windows computer as the
Next.js development server.

1. Install Ollama for Windows from https://ollama.com/download/windows.
2. Open a new PowerShell window.
3. Download the configured model:

```powershell
ollama pull llama3.2
```

The Windows Ollama app normally starts the API automatically. If it is not
running, open the Ollama app or run `ollama serve` in a separate terminal.

Verify the service and model:

```powershell
npm run ai:check
npm run ai:test
```

The live status endpoint is `GET /api/ai/health`.

AI is used for:
- Diagnosis support (Doctor → Diagnoses page)
- Patient summary generation
- Report generation

AI output is clinical decision support only. A licensed clinician must review
all generated content before it is used in patient care.

---

## Automation (n8n)

The project includes a secure, importable notification bridge in
`n8n/workflows/medicare-notification-bridge.json`.

Install n8n globally:

```powershell
npm install n8n -g
```

Start the local instance with the project environment:

```powershell
npm run automation:start
```

Open `http://localhost:5678`, create the local owner account, then import the
workflow JSON using **Import from File**. Open the imported workflow and
publish it so the production webhook is registered.

Verify the local service and published webhook:

```powershell
npm run automation:check
npm run automation:test
```

App-to-n8n endpoint:
`POST http://localhost:5678/webhook/medicare-events`

n8n-to-app endpoint:
`POST http://localhost:3000/api/webhooks/n8n`

Supported events: `appointment.reminder`, `prescription.refill`,
`prescription.created`, `report.ready`, `patient.registered`, and
`billing.alert`.

---

## Deployment (Vercel)

```bash
vercel deploy
```

Set all environment variables from `.env.local` in the Vercel project settings.
