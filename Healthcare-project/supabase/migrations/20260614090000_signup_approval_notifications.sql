create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  profile_name text;
  requested_role_text text;
  requested_role_value public.user_role;
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

  if requested_role_text = 'admin' then
    requested_role_value := 'admin';
  elsif requested_role_text = 'doctor' then
    if doctor_specialization is null or doctor_license_number is null then
      raise exception 'Doctor specialization and license number are required'
        using errcode = '22023';
    end if;
    requested_role_value := 'doctor';
  end if;

  insert into public.profiles (id, email, full_name, role)
  values (new.id, coalesce(new.email, ''), profile_name, 'patient');

  insert into public.notification_preferences (user_id)
  values (new.id);

  if requested_role_value is null then
    insert into public.patients (profile_id)
    values (new.id);
  else
    insert into public.role_requests (
      user_id,
      requested_role,
      specialization,
      license_number
    )
    values (
      new.id,
      requested_role_value,
      case
        when requested_role_value = 'doctor' then doctor_specialization
        else null
      end,
      case
        when requested_role_value = 'doctor' then doctor_license_number
        else null
      end
    );

    insert into public.notifications (
      user_id,
      title,
      message,
      type,
      link
    )
    select
      administrator.id,
      format('New %s registration', initcap(requested_role_value::text)),
      format(
        '%s (%s) registered as a %s and is waiting for approval.',
        profile_name,
        coalesce(new.email, ''),
        requested_role_value::text
      ),
      'user',
      '/admin/roles'
    from public.profiles administrator
    where administrator.role = 'admin';
  end if;

  return new;
end;
$$;

insert into public.notifications (
  user_id,
  title,
  message,
  type,
  link
)
select
  administrator.id,
  format('New %s registration', initcap(role_request.requested_role::text)),
  format(
    '%s (%s) registered as a %s and is waiting for approval.',
    applicant.full_name,
    applicant.email,
    role_request.requested_role::text
  ),
  'user',
  '/admin/roles'
from public.role_requests role_request
join public.profiles applicant
  on applicant.id = role_request.user_id
cross join public.profiles administrator
where role_request.status = 'pending'
  and administrator.role = 'admin'
  and not exists (
    select 1
    from public.notifications notification
    where notification.user_id = administrator.id
      and notification.link = '/admin/roles'
      and notification.message = format(
        '%s (%s) registered as a %s and is waiting for approval.',
        applicant.full_name,
        applicant.email,
        role_request.requested_role::text
      )
  );
