create extension if not exists pgcrypto with schema extensions;

create type public.user_role as enum ('admin', 'doctor', 'patient');
create type public.appointment_status as enum (
  'scheduled',
  'confirmed',
  'in_progress',
  'completed',
  'cancelled',
  'no_show'
);
create type public.gender as enum ('male', 'female', 'other');
create type public.invoice_status as enum ('pending', 'paid', 'overdue', 'cancelled');
create type public.health_goal_status as enum ('in_progress', 'on_track', 'completed', 'paused');

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text not null,
  full_name text not null check (char_length(btrim(full_name)) between 1 and 150),
  role public.user_role not null default 'patient',
  avatar_url text,
  phone text,
  date_of_birth date,
  gender public.gender,
  address text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index profiles_email_lower_key on public.profiles (lower(email));

create table public.doctors (
  id uuid primary key default extensions.gen_random_uuid(),
  profile_id uuid not null unique references public.profiles(id) on delete cascade,
  specialization text not null check (char_length(btrim(specialization)) > 0),
  license_number text not null unique check (char_length(btrim(license_number)) > 0),
  department text,
  experience_years integer not null default 0 check (experience_years >= 0),
  consultation_fee numeric(10,2) not null default 0 check (consultation_fee >= 0),
  available_hours jsonb,
  bio text,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.patients (
  id uuid primary key default extensions.gen_random_uuid(),
  profile_id uuid not null unique references public.profiles(id) on delete cascade,
  blood_type text check (
    blood_type is null or blood_type in ('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-')
  ),
  allergies text[] not null default '{}',
  emergency_contact_name text,
  emergency_contact_phone text,
  insurance_provider text,
  insurance_number text,
  assigned_doctor_id uuid references public.doctors(id) on delete set null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.appointments (
  id uuid primary key default extensions.gen_random_uuid(),
  patient_id uuid not null references public.patients(id) on delete cascade,
  doctor_id uuid not null references public.doctors(id) on delete restrict,
  scheduled_at timestamptz not null,
  duration_minutes integer not null default 30 check (duration_minutes between 5 and 480),
  status public.appointment_status not null default 'scheduled',
  type text not null default 'consultation' check (char_length(btrim(type)) > 0),
  notes text,
  reason text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint appointments_unique_slot unique (doctor_id, scheduled_at)
);

create table public.medical_records (
  id uuid primary key default extensions.gen_random_uuid(),
  patient_id uuid not null references public.patients(id) on delete cascade,
  doctor_id uuid not null references public.doctors(id) on delete restrict,
  appointment_id uuid references public.appointments(id) on delete set null,
  diagnosis text not null check (char_length(btrim(diagnosis)) > 0),
  symptoms text[] not null default '{}',
  treatment_plan text,
  notes text,
  vitals jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.prescriptions (
  id uuid primary key default extensions.gen_random_uuid(),
  patient_id uuid not null references public.patients(id) on delete cascade,
  doctor_id uuid not null references public.doctors(id) on delete restrict,
  medical_record_id uuid references public.medical_records(id) on delete set null,
  medications jsonb not null default '[]'::jsonb
    check (jsonb_typeof(medications) = 'array'),
  instructions text,
  valid_until date,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.documents (
  id uuid primary key default extensions.gen_random_uuid(),
  patient_id uuid not null references public.patients(id) on delete cascade,
  uploaded_by uuid not null references public.profiles(id) on delete restrict,
  name text not null check (char_length(btrim(name)) > 0),
  type text not null check (char_length(btrim(type)) > 0),
  storage_path text not null unique check (char_length(btrim(storage_path)) > 0),
  mime_type text,
  size_bytes bigint check (size_bytes is null or size_bytes >= 0),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.messages (
  id uuid primary key default extensions.gen_random_uuid(),
  sender_id uuid not null references public.profiles(id) on delete cascade,
  receiver_id uuid not null references public.profiles(id) on delete cascade,
  content text not null check (char_length(btrim(content)) between 1 and 10000),
  is_read boolean not null default false,
  created_at timestamptz not null default now(),
  read_at timestamptz,
  constraint messages_distinct_participants check (sender_id <> receiver_id)
);

create table public.notifications (
  id uuid primary key default extensions.gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  title text not null check (char_length(btrim(title)) > 0),
  message text not null check (char_length(btrim(message)) > 0),
  type text not null default 'info',
  is_read boolean not null default false,
  link text,
  created_at timestamptz not null default now(),
  read_at timestamptz
);

create table public.activity_logs (
  id uuid primary key default extensions.gen_random_uuid(),
  user_id uuid references public.profiles(id) on delete set null,
  action text not null,
  resource_type text not null,
  resource_id text,
  details jsonb,
  ip_address inet,
  created_at timestamptz not null default now()
);

create table public.invoices (
  id uuid primary key default extensions.gen_random_uuid(),
  patient_id uuid not null references public.patients(id) on delete cascade,
  appointment_id uuid unique references public.appointments(id) on delete set null,
  amount numeric(10,2) not null check (amount >= 0),
  status public.invoice_status not null default 'pending',
  due_date date,
  paid_at timestamptz,
  description text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint invoices_paid_at_check check (
    (status = 'paid' and paid_at is not null) or status <> 'paid'
  )
);

create table public.vital_readings (
  id uuid primary key default extensions.gen_random_uuid(),
  patient_id uuid not null references public.patients(id) on delete cascade,
  recorded_by uuid references public.profiles(id) on delete set null,
  blood_pressure_systolic integer check (blood_pressure_systolic between 40 and 300),
  blood_pressure_diastolic integer check (blood_pressure_diastolic between 20 and 200),
  heart_rate integer check (heart_rate between 20 and 300),
  temperature_celsius numeric(4,1) check (temperature_celsius between 25 and 45),
  blood_glucose_mg_dl numeric(6,2) check (blood_glucose_mg_dl between 0 and 1000),
  oxygen_saturation numeric(5,2) check (oxygen_saturation between 0 and 100),
  weight_kg numeric(6,2) check (weight_kg between 0 and 1000),
  notes text,
  recorded_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.health_goals (
  id uuid primary key default extensions.gen_random_uuid(),
  patient_id uuid not null references public.patients(id) on delete cascade,
  title text not null check (char_length(btrim(title)) > 0),
  description text,
  target_value numeric,
  current_value numeric,
  unit text,
  target_date date,
  status public.health_goal_status not null default 'in_progress',
  created_by uuid references public.profiles(id) on delete set null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.notification_preferences (
  user_id uuid primary key references public.profiles(id) on delete cascade,
  appointment_reminders boolean not null default true,
  prescription_reminders boolean not null default true,
  lab_results boolean not null default true,
  new_messages boolean not null default true,
  billing_notifications boolean not null default true,
  health_tips boolean not null default false,
  email_enabled boolean not null default true,
  sms_enabled boolean not null default false,
  updated_at timestamptz not null default now()
);

create table public.system_settings (
  key text primary key check (char_length(btrim(key)) > 0),
  value jsonb not null,
  description text,
  updated_by uuid references public.profiles(id) on delete set null,
  updated_at timestamptz not null default now()
);

create index appointments_patient_scheduled_idx
  on public.appointments (patient_id, scheduled_at desc);
create index appointments_doctor_scheduled_idx
  on public.appointments (doctor_id, scheduled_at desc);
create index appointments_status_idx on public.appointments (status);
create index medical_records_patient_created_idx
  on public.medical_records (patient_id, created_at desc);
create index medical_records_doctor_idx on public.medical_records (doctor_id);
create index prescriptions_patient_active_idx
  on public.prescriptions (patient_id, is_active, created_at desc);
create index prescriptions_doctor_idx on public.prescriptions (doctor_id);
create index documents_patient_created_idx
  on public.documents (patient_id, created_at desc);
create index messages_sender_created_idx
  on public.messages (sender_id, created_at desc);
create index messages_receiver_created_idx
  on public.messages (receiver_id, created_at desc);
create index notifications_user_read_created_idx
  on public.notifications (user_id, is_read, created_at desc);
create index activity_logs_created_idx on public.activity_logs (created_at desc);
create index activity_logs_user_idx on public.activity_logs (user_id, created_at desc);
create index invoices_patient_status_idx on public.invoices (patient_id, status);
create index vital_readings_patient_recorded_idx
  on public.vital_readings (patient_id, recorded_at desc);
create index health_goals_patient_status_idx
  on public.health_goals (patient_id, status);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger profiles_set_updated_at
before update on public.profiles
for each row execute function public.set_updated_at();

create trigger doctors_set_updated_at
before update on public.doctors
for each row execute function public.set_updated_at();

create trigger patients_set_updated_at
before update on public.patients
for each row execute function public.set_updated_at();

create trigger appointments_set_updated_at
before update on public.appointments
for each row execute function public.set_updated_at();

create trigger medical_records_set_updated_at
before update on public.medical_records
for each row execute function public.set_updated_at();

create trigger prescriptions_set_updated_at
before update on public.prescriptions
for each row execute function public.set_updated_at();

create trigger documents_set_updated_at
before update on public.documents
for each row execute function public.set_updated_at();

create trigger invoices_set_updated_at
before update on public.invoices
for each row execute function public.set_updated_at();

create trigger vital_readings_set_updated_at
before update on public.vital_readings
for each row execute function public.set_updated_at();

create trigger health_goals_set_updated_at
before update on public.health_goals
for each row execute function public.set_updated_at();

create trigger notification_preferences_set_updated_at
before update on public.notification_preferences
for each row execute function public.set_updated_at();

create trigger system_settings_set_updated_at
before update on public.system_settings
for each row execute function public.set_updated_at();

