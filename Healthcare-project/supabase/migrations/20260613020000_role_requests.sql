create table public.role_requests (
  id uuid primary key default extensions.gen_random_uuid(),
  user_id uuid not null unique references public.profiles(id) on delete cascade,
  requested_role public.user_role not null
    check (requested_role in ('doctor', 'admin')),
  specialization text,
  license_number text,
  status text not null default 'pending'
    check (status in ('pending', 'approved', 'rejected')),
  review_note text,
  reviewed_by uuid references public.profiles(id) on delete set null,
  reviewed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint role_requests_doctor_fields_check check (
    requested_role <> 'doctor'
    or (
      nullif(btrim(specialization), '') is not null
      and nullif(btrim(license_number), '') is not null
    )
  )
);

create index role_requests_status_created_idx
  on public.role_requests (status, created_at desc);

create trigger role_requests_set_updated_at
before update on public.role_requests
for each row execute function public.set_updated_at();

alter table public.role_requests enable row level security;

create policy role_requests_select
on public.role_requests for select
to authenticated
using (
  user_id = (select auth.uid())
  or private.is_admin()
);

grant select on public.role_requests to authenticated;

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  profile_name text;
  requested_role_text text;
  doctor_specialization text;
  doctor_license_number text;
begin
  profile_name := coalesce(
    nullif(btrim(new.raw_user_meta_data ->> 'full_name'), ''),
    split_part(coalesce(new.email, 'Patient'), '@', 1)
  );
  requested_role_text := lower(
    coalesce(new.raw_user_meta_data ->> 'requested_role', 'patient')
  );
  doctor_specialization :=
    nullif(btrim(new.raw_user_meta_data ->> 'doctor_specialization'), '');
  doctor_license_number :=
    nullif(btrim(new.raw_user_meta_data ->> 'doctor_license_number'), '');

  insert into public.profiles (id, email, full_name, role)
  values (new.id, coalesce(new.email, ''), profile_name, 'patient');

  insert into public.patients (profile_id)
  values (new.id);

  insert into public.notification_preferences (user_id)
  values (new.id);

  if requested_role_text = 'admin' then
    insert into public.role_requests (user_id, requested_role)
    values (new.id, 'admin');
  elsif requested_role_text = 'doctor'
    and doctor_specialization is not null
    and doctor_license_number is not null then
    insert into public.role_requests (
      user_id,
      requested_role,
      specialization,
      license_number
    )
    values (
      new.id,
      'doctor',
      doctor_specialization,
      doctor_license_number
    );
  end if;

  return new;
end;
$$;

create or replace function public.admin_review_role_request(
  target_request_id uuid,
  approve_request boolean,
  review_message text default null
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  role_request public.role_requests%rowtype;
begin
  if not private.is_admin() then
    raise exception 'Administrator access required' using errcode = '42501';
  end if;

  select *
  into role_request
  from public.role_requests
  where id = target_request_id
  for update;

  if role_request.id is null then
    raise exception 'Role request not found' using errcode = 'P0002';
  end if;

  if role_request.status <> 'pending' then
    raise exception 'Role request has already been reviewed'
      using errcode = '22023';
  end if;

  if approve_request then
    perform private.set_user_role(
      role_request.user_id,
      role_request.requested_role,
      role_request.specialization,
      role_request.license_number
    );

    update public.role_requests
    set status = 'approved',
        review_note = nullif(btrim(review_message), ''),
        reviewed_by = (select auth.uid()),
        reviewed_at = now()
    where id = target_request_id;

    insert into public.notifications (
      user_id,
      title,
      message,
      type,
      link
    )
    values (
      role_request.user_id,
      'Role request approved',
      format(
        'Your %s account has been approved. Sign in again to open the correct portal.',
        role_request.requested_role
      ),
      'info',
      '/login'
    );
  else
    update public.role_requests
    set status = 'rejected',
        review_note = nullif(btrim(review_message), ''),
        reviewed_by = (select auth.uid()),
        reviewed_at = now()
    where id = target_request_id;

    insert into public.notifications (
      user_id,
      title,
      message,
      type,
      link
    )
    values (
      role_request.user_id,
      'Role request reviewed',
      coalesce(
        nullif(btrim(review_message), ''),
        'Your requested role was not approved. Contact an administrator for details.'
      ),
      'alert',
      '/patient/notifications'
    );
  end if;
end;
$$;

revoke all on function public.admin_review_role_request(uuid, boolean, text)
  from public, anon;
grant execute on function public.admin_review_role_request(uuid, boolean, text)
  to authenticated, service_role;
