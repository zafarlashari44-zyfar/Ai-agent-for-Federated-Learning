create or replace function private.set_user_role(
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

revoke all on function private.set_user_role(uuid, public.user_role, text, text)
  from public, anon, authenticated, service_role;

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

  perform private.set_user_role(
    target_user_id,
    new_role,
    doctor_specialization,
    doctor_license_number
  );
end;
$$;

create or replace function public.service_set_user_role(
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
  if (select auth.role()) <> 'service_role' then
    raise exception 'Service role access required' using errcode = '42501';
  end if;

  perform private.set_user_role(
    target_user_id,
    new_role,
    doctor_specialization,
    doctor_license_number
  );
end;
$$;

revoke all on function public.service_set_user_role(uuid, public.user_role, text, text)
  from public, anon, authenticated;
grant execute on function public.service_set_user_role(uuid, public.user_role, text, text)
  to service_role;
