create schema if not exists private;

revoke all on schema private from public;
grant usage on schema private to authenticated, service_role;

create or replace function private.current_user_role()
returns public.user_role
language sql
stable
security definer
set search_path = ''
as $$
  select role
  from public.profiles
  where id = (select auth.uid())
$$;

create or replace function private.is_admin()
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select coalesce(private.current_user_role() = 'admin', false)
$$;

create or replace function private.current_doctor_id()
returns uuid
language sql
stable
security definer
set search_path = ''
as $$
  select id
  from public.doctors
  where profile_id = (select auth.uid())
    and is_active
$$;

create or replace function private.current_patient_id()
returns uuid
language sql
stable
security definer
set search_path = ''
as $$
  select id
  from public.patients
  where profile_id = (select auth.uid())
$$;

create or replace function private.can_access_patient(target_patient_id uuid)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select
    private.is_admin()
    or exists (
      select 1
      from public.patients patient
      where patient.id = target_patient_id
        and patient.profile_id = (select auth.uid())
    )
    or exists (
      select 1
      from public.patients patient
      where patient.id = target_patient_id
        and patient.assigned_doctor_id = private.current_doctor_id()
    )
$$;

revoke all on function private.current_user_role() from public;
revoke all on function private.is_admin() from public;
revoke all on function private.current_doctor_id() from public;
revoke all on function private.current_patient_id() from public;
revoke all on function private.can_access_patient(uuid) from public;

grant execute on function private.current_user_role() to authenticated, service_role;
grant execute on function private.is_admin() to authenticated, service_role;
grant execute on function private.current_doctor_id() to authenticated, service_role;
grant execute on function private.current_patient_id() to authenticated, service_role;
grant execute on function private.can_access_patient(uuid) to authenticated, service_role;

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  profile_name text;
begin
  profile_name := coalesce(
    nullif(btrim(new.raw_user_meta_data ->> 'full_name'), ''),
    split_part(coalesce(new.email, 'Patient'), '@', 1)
  );

  insert into public.profiles (id, email, full_name, role)
  values (new.id, coalesce(new.email, ''), profile_name, 'patient');

  insert into public.patients (profile_id)
  values (new.id);

  insert into public.notification_preferences (user_id)
  values (new.id);

  return new;
end;
$$;

create trigger on_auth_user_created
after insert on auth.users
for each row execute function public.handle_new_user();

create or replace function public.handle_auth_user_updated()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  update public.profiles
  set email = coalesce(new.email, email)
  where id = new.id
    and email is distinct from new.email;

  return new;
end;
$$;

create trigger on_auth_user_updated
after update of email on auth.users
for each row execute function public.handle_auth_user_updated();

insert into public.profiles (id, email, full_name, role)
select
  user_account.id,
  coalesce(user_account.email, ''),
  coalesce(
    nullif(btrim(user_account.raw_user_meta_data ->> 'full_name'), ''),
    split_part(coalesce(user_account.email, 'Patient'), '@', 1)
  ),
  'patient'
from auth.users user_account
on conflict (id) do nothing;

insert into public.patients (profile_id)
select profile.id
from public.profiles profile
where profile.role = 'patient'
on conflict (profile_id) do nothing;

insert into public.notification_preferences (user_id)
select profile.id
from public.profiles profile
on conflict (user_id) do nothing;

create or replace function public.list_active_doctors()
returns table (
  doctor_id uuid,
  profile_id uuid,
  full_name text,
  avatar_url text,
  specialization text,
  department text,
  experience_years integer,
  consultation_fee numeric,
  available_hours jsonb,
  bio text
)
language sql
stable
security definer
set search_path = ''
as $$
  select
    doctor.id,
    doctor.profile_id,
    profile.full_name,
    profile.avatar_url,
    doctor.specialization,
    doctor.department,
    doctor.experience_years,
    doctor.consultation_fee,
    doctor.available_hours,
    doctor.bio
  from public.doctors doctor
  join public.profiles profile on profile.id = doctor.profile_id
  where doctor.is_active
  order by profile.full_name
$$;

revoke all on function public.list_active_doctors() from public;
grant execute on function public.list_active_doctors() to anon, authenticated, service_role;

create or replace function public.admin_set_user_role(
  target_user_id uuid,
  new_role public.user_role,
  doctor_specialization text default null,
  doctor_license_number text default null
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
  if not private.is_admin() then
    raise exception 'Administrator access required' using errcode = '42501';
  end if;

  if not exists (select 1 from public.profiles where id = target_user_id) then
    raise exception 'Profile not found' using errcode = 'P0002';
  end if;

  if new_role = 'doctor' then
    if nullif(btrim(doctor_specialization), '') is null
      or nullif(btrim(doctor_license_number), '') is null then
      raise exception 'Doctor specialization and license number are required'
        using errcode = '22023';
    end if;

    insert into public.doctors (
      profile_id,
      specialization,
      license_number,
      is_active
    )
    values (
      target_user_id,
      btrim(doctor_specialization),
      btrim(doctor_license_number),
      true
    )
    on conflict (profile_id) do update
      set specialization = excluded.specialization,
          license_number = excluded.license_number,
          is_active = true;
  elsif new_role = 'patient' then
    insert into public.patients (profile_id)
    values (target_user_id)
    on conflict (profile_id) do nothing;
  end if;

  if new_role <> 'doctor' then
    update public.doctors
    set is_active = false
    where profile_id = target_user_id;
  end if;

  update public.profiles
  set role = new_role
  where id = target_user_id;
end;
$$;

create or replace function public.admin_assign_doctor(
  target_patient_id uuid,
  target_doctor_id uuid
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
  if not private.is_admin() then
    raise exception 'Administrator access required' using errcode = '42501';
  end if;

  if not exists (
    select 1 from public.doctors
    where id = target_doctor_id and is_active
  ) then
    raise exception 'Active doctor not found' using errcode = 'P0002';
  end if;

  update public.patients
  set assigned_doctor_id = target_doctor_id
  where id = target_patient_id;

  if not found then
    raise exception 'Patient not found' using errcode = 'P0002';
  end if;
end;
$$;

revoke all on function public.admin_set_user_role(uuid, public.user_role, text, text) from public;
revoke all on function public.admin_assign_doctor(uuid, uuid) from public;
grant execute on function public.admin_set_user_role(uuid, public.user_role, text, text)
  to authenticated, service_role;
grant execute on function public.admin_assign_doctor(uuid, uuid)
  to authenticated, service_role;

create or replace function private.enforce_appointment_write()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  actor_role public.user_role;
begin
  actor_role := private.current_user_role();

  if actor_role = 'admin' or (select auth.role()) = 'service_role' then
    return new;
  end if;

  if tg_op = 'UPDATE'
    and (
      new.patient_id is distinct from old.patient_id
      or new.doctor_id is distinct from old.doctor_id
      or new.created_at is distinct from old.created_at
    ) then
    raise exception 'Appointment ownership fields cannot be changed'
      using errcode = '42501';
  end if;

  if actor_role = 'patient' then
    if new.patient_id <> private.current_patient_id() then
      raise exception 'Patients may only manage their own appointments'
        using errcode = '42501';
    end if;

    if tg_op = 'INSERT' and new.status <> 'scheduled' then
      raise exception 'New patient appointments must be scheduled'
        using errcode = '42501';
    end if;

    if tg_op = 'UPDATE'
      and new.status is distinct from old.status
      and new.status <> 'cancelled' then
      raise exception 'Patients may only cancel appointment status'
        using errcode = '42501';
    end if;

    return new;
  end if;

  if actor_role = 'doctor' then
    if new.doctor_id <> private.current_doctor_id()
      or not private.can_access_patient(new.patient_id) then
      raise exception 'Doctors may only manage assigned-patient appointments'
        using errcode = '42501';
    end if;

    return new;
  end if;

  raise exception 'Appointment access denied' using errcode = '42501';
end;
$$;

create trigger appointments_enforce_write
before insert or update on public.appointments
for each row execute function private.enforce_appointment_write();

create or replace function private.validate_medical_record_links()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  if new.appointment_id is not null and not exists (
    select 1
    from public.appointments appointment
    where appointment.id = new.appointment_id
      and appointment.patient_id = new.patient_id
      and appointment.doctor_id = new.doctor_id
  ) then
    raise exception 'Medical record appointment does not match patient and doctor'
      using errcode = '23514';
  end if;

  return new;
end;
$$;

create trigger medical_records_validate_links
before insert or update on public.medical_records
for each row execute function private.validate_medical_record_links();

create or replace function private.validate_prescription_links()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  if new.medical_record_id is not null and not exists (
    select 1
    from public.medical_records medical_record
    where medical_record.id = new.medical_record_id
      and medical_record.patient_id = new.patient_id
      and medical_record.doctor_id = new.doctor_id
  ) then
    raise exception 'Prescription medical record does not match patient and doctor'
      using errcode = '23514';
  end if;

  return new;
end;
$$;

create trigger prescriptions_validate_links
before insert or update on public.prescriptions
for each row execute function private.validate_prescription_links();

create or replace function private.validate_document_path()
returns trigger
language plpgsql
set search_path = ''
as $$
declare
  patient_profile_id uuid;
begin
  select profile_id
  into patient_profile_id
  from public.patients
  where id = new.patient_id;

  if patient_profile_id is null
    or split_part(new.storage_path, '/', 1) <> patient_profile_id::text then
    raise exception 'Document path must begin with the patient profile ID'
      using errcode = '23514';
  end if;

  return new;
end;
$$;

create trigger documents_validate_path
before insert or update of patient_id, storage_path on public.documents
for each row execute function private.validate_document_path();

alter table public.profiles enable row level security;
alter table public.doctors enable row level security;
alter table public.patients enable row level security;
alter table public.appointments enable row level security;
alter table public.medical_records enable row level security;
alter table public.prescriptions enable row level security;
alter table public.documents enable row level security;
alter table public.messages enable row level security;
alter table public.notifications enable row level security;
alter table public.activity_logs enable row level security;
alter table public.invoices enable row level security;
alter table public.vital_readings enable row level security;
alter table public.health_goals enable row level security;
alter table public.notification_preferences enable row level security;
alter table public.system_settings enable row level security;

create policy profiles_select
on public.profiles for select
to authenticated
using (
  id = (select auth.uid())
  or private.is_admin()
  or exists (
    select 1
    from public.patients patient
    where patient.profile_id = profiles.id
      and patient.assigned_doctor_id = private.current_doctor_id()
  )
);

create policy profiles_update
on public.profiles for update
to authenticated
using (id = (select auth.uid()) or private.is_admin())
with check (id = (select auth.uid()) or private.is_admin());

create policy doctors_select
on public.doctors for select
to authenticated
using (
  profile_id = (select auth.uid())
  or private.is_admin()
  or exists (
    select 1
    from public.patients patient
    where patient.profile_id = (select auth.uid())
      and patient.assigned_doctor_id = doctors.id
  )
);

create policy doctors_admin_all
on public.doctors for all
to authenticated
using (private.is_admin())
with check (private.is_admin());

create policy patients_select
on public.patients for select
to authenticated
using (private.can_access_patient(id));

create policy patients_update
on public.patients for update
to authenticated
using (private.can_access_patient(id))
with check (private.can_access_patient(id));

create policy appointments_select
on public.appointments for select
to authenticated
using (
  private.is_admin()
  or patient_id = private.current_patient_id()
  or doctor_id = private.current_doctor_id()
);

create policy appointments_insert
on public.appointments for insert
to authenticated
with check (
  private.is_admin()
  or (
    patient_id = private.current_patient_id()
    and status = 'scheduled'
  )
  or (
    doctor_id = private.current_doctor_id()
    and private.can_access_patient(patient_id)
  )
);

create policy appointments_update
on public.appointments for update
to authenticated
using (
  private.is_admin()
  or patient_id = private.current_patient_id()
  or doctor_id = private.current_doctor_id()
)
with check (
  private.is_admin()
  or patient_id = private.current_patient_id()
  or doctor_id = private.current_doctor_id()
);

create policy appointments_delete
on public.appointments for delete
to authenticated
using (private.is_admin());

create policy medical_records_select
on public.medical_records for select
to authenticated
using (private.can_access_patient(patient_id));

create policy medical_records_insert
on public.medical_records for insert
to authenticated
with check (
  private.is_admin()
  or (
    doctor_id = private.current_doctor_id()
    and private.can_access_patient(patient_id)
  )
);

create policy medical_records_update
on public.medical_records for update
to authenticated
using (
  private.is_admin()
  or (
    doctor_id = private.current_doctor_id()
    and private.can_access_patient(patient_id)
  )
)
with check (
  private.is_admin()
  or (
    doctor_id = private.current_doctor_id()
    and private.can_access_patient(patient_id)
  )
);

create policy medical_records_delete
on public.medical_records for delete
to authenticated
using (private.is_admin());

create policy prescriptions_select
on public.prescriptions for select
to authenticated
using (private.can_access_patient(patient_id));

create policy prescriptions_insert
on public.prescriptions for insert
to authenticated
with check (
  private.is_admin()
  or (
    doctor_id = private.current_doctor_id()
    and private.can_access_patient(patient_id)
  )
);

create policy prescriptions_update
on public.prescriptions for update
to authenticated
using (
  private.is_admin()
  or (
    doctor_id = private.current_doctor_id()
    and private.can_access_patient(patient_id)
  )
)
with check (
  private.is_admin()
  or (
    doctor_id = private.current_doctor_id()
    and private.can_access_patient(patient_id)
  )
);

create policy prescriptions_delete
on public.prescriptions for delete
to authenticated
using (private.is_admin());

create policy documents_select
on public.documents for select
to authenticated
using (private.can_access_patient(patient_id));

create policy documents_insert
on public.documents for insert
to authenticated
with check (
  uploaded_by = (select auth.uid())
  and private.can_access_patient(patient_id)
);

create policy documents_update
on public.documents for update
to authenticated
using (
  private.is_admin()
  or uploaded_by = (select auth.uid())
  or patient_id = private.current_patient_id()
)
with check (
  private.is_admin()
  or uploaded_by = (select auth.uid())
  or patient_id = private.current_patient_id()
);

create policy documents_delete
on public.documents for delete
to authenticated
using (
  private.is_admin()
  or uploaded_by = (select auth.uid())
  or patient_id = private.current_patient_id()
);

create policy messages_select
on public.messages for select
to authenticated
using (
  sender_id = (select auth.uid())
  or receiver_id = (select auth.uid())
  or private.is_admin()
);

create policy messages_insert
on public.messages for insert
to authenticated
with check (sender_id = (select auth.uid()));

create policy messages_update
on public.messages for update
to authenticated
using (receiver_id = (select auth.uid()) or private.is_admin())
with check (receiver_id = (select auth.uid()) or private.is_admin());

create policy messages_delete
on public.messages for delete
to authenticated
using (sender_id = (select auth.uid()) or private.is_admin());

create policy notifications_select
on public.notifications for select
to authenticated
using (user_id = (select auth.uid()) or private.is_admin());

create policy notifications_update
on public.notifications for update
to authenticated
using (user_id = (select auth.uid()) or private.is_admin())
with check (user_id = (select auth.uid()) or private.is_admin());

create policy notifications_delete
on public.notifications for delete
to authenticated
using (user_id = (select auth.uid()) or private.is_admin());

create policy activity_logs_select
on public.activity_logs for select
to authenticated
using (private.is_admin());

create policy invoices_select
on public.invoices for select
to authenticated
using (
  private.is_admin()
  or patient_id = private.current_patient_id()
);

create policy invoices_admin_all
on public.invoices for all
to authenticated
using (private.is_admin())
with check (private.is_admin());

create policy vital_readings_select
on public.vital_readings for select
to authenticated
using (private.can_access_patient(patient_id));

create policy vital_readings_insert
on public.vital_readings for insert
to authenticated
with check (
  recorded_by = (select auth.uid())
  and private.can_access_patient(patient_id)
);

create policy vital_readings_update
on public.vital_readings for update
to authenticated
using (recorded_by = (select auth.uid()) or private.is_admin())
with check (recorded_by = (select auth.uid()) or private.is_admin());

create policy vital_readings_delete
on public.vital_readings for delete
to authenticated
using (recorded_by = (select auth.uid()) or private.is_admin());

create policy health_goals_select
on public.health_goals for select
to authenticated
using (private.can_access_patient(patient_id));

create policy health_goals_insert
on public.health_goals for insert
to authenticated
with check (
  created_by = (select auth.uid())
  and private.can_access_patient(patient_id)
);

create policy health_goals_update
on public.health_goals for update
to authenticated
using (private.can_access_patient(patient_id))
with check (private.can_access_patient(patient_id));

create policy health_goals_delete
on public.health_goals for delete
to authenticated
using (
  created_by = (select auth.uid())
  or patient_id = private.current_patient_id()
  or private.is_admin()
);

create policy notification_preferences_select
on public.notification_preferences for select
to authenticated
using (user_id = (select auth.uid()) or private.is_admin());

create policy notification_preferences_insert
on public.notification_preferences for insert
to authenticated
with check (user_id = (select auth.uid()) or private.is_admin());

create policy notification_preferences_update
on public.notification_preferences for update
to authenticated
using (user_id = (select auth.uid()) or private.is_admin())
with check (user_id = (select auth.uid()) or private.is_admin());

create policy system_settings_admin_all
on public.system_settings for all
to authenticated
using (private.is_admin())
with check (private.is_admin());

revoke all on all tables in schema public from anon, authenticated;

grant select, update on public.profiles to authenticated;
revoke update on public.profiles from authenticated;
grant update (full_name, avatar_url, phone, date_of_birth, gender, address)
  on public.profiles to authenticated;

grant select on public.doctors to authenticated;
grant select, update on public.patients to authenticated;
revoke update on public.patients from authenticated;
grant update (
  blood_type,
  allergies,
  emergency_contact_name,
  emergency_contact_phone,
  insurance_provider,
  insurance_number
) on public.patients to authenticated;

grant select, insert, update, delete on public.appointments to authenticated;
grant select, insert, update, delete on public.medical_records to authenticated;
grant select, insert, update, delete on public.prescriptions to authenticated;
grant select, insert, delete on public.documents to authenticated;
grant update (name, type) on public.documents to authenticated;
grant select, insert, delete on public.messages to authenticated;
grant update (is_read, read_at) on public.messages to authenticated;
grant select, delete on public.notifications to authenticated;
grant update (is_read, read_at) on public.notifications to authenticated;
grant select on public.activity_logs to authenticated;
grant select on public.invoices to authenticated;
grant select, insert, update, delete on public.vital_readings to authenticated;
grant select, insert, update, delete on public.health_goals to authenticated;
grant select, insert, update on public.notification_preferences to authenticated;
grant select, insert, update, delete on public.system_settings to authenticated;

grant all on all tables in schema public to service_role;

create or replace function private.write_audit_log()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  row_data jsonb;
begin
  row_data := case when tg_op = 'DELETE' then to_jsonb(old) else to_jsonb(new) end;

  insert into public.activity_logs (
    user_id,
    action,
    resource_type,
    resource_id,
    details
  )
  values (
    (select auth.uid()),
    lower(tg_op),
    tg_table_name,
    row_data ->> 'id',
    jsonb_build_object('operation', tg_op)
  );

  if tg_op = 'DELETE' then
    return old;
  end if;

  return new;
end;
$$;

create trigger audit_doctors
after insert or update or delete on public.doctors
for each row execute function private.write_audit_log();

create trigger audit_patients
after insert or update or delete on public.patients
for each row execute function private.write_audit_log();

create trigger audit_appointments
after insert or update or delete on public.appointments
for each row execute function private.write_audit_log();

create trigger audit_medical_records
after insert or update or delete on public.medical_records
for each row execute function private.write_audit_log();

create trigger audit_prescriptions
after insert or update or delete on public.prescriptions
for each row execute function private.write_audit_log();

create trigger audit_documents
after insert or update or delete on public.documents
for each row execute function private.write_audit_log();

create trigger audit_invoices
after insert or update or delete on public.invoices
for each row execute function private.write_audit_log();
