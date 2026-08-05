insert into storage.buckets (
  id,
  name,
  public,
  file_size_limit,
  allowed_mime_types
)
values (
  'patient-documents',
  'patient-documents',
  false,
  20971520,
  array[
    'application/pdf',
    'image/jpeg',
    'image/png',
    'image/webp',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  ]
)
on conflict (id) do update
set public = excluded.public,
    file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;

create policy patient_documents_select
on storage.objects for select
to authenticated
using (
  bucket_id = 'patient-documents'
  and (
    private.is_admin()
    or (storage.foldername(name))[1] = (select auth.uid())::text
    or exists (
      select 1
      from public.patients patient
      where patient.profile_id::text = (storage.foldername(name))[1]
        and patient.assigned_doctor_id = private.current_doctor_id()
    )
  )
);

create policy patient_documents_insert
on storage.objects for insert
to authenticated
with check (
  bucket_id = 'patient-documents'
  and (
    private.is_admin()
    or (storage.foldername(name))[1] = (select auth.uid())::text
    or exists (
      select 1
      from public.patients patient
      where patient.profile_id::text = (storage.foldername(name))[1]
        and patient.assigned_doctor_id = private.current_doctor_id()
    )
  )
);

create policy patient_documents_update
on storage.objects for update
to authenticated
using (
  bucket_id = 'patient-documents'
  and (
    private.is_admin()
    or owner_id = (select auth.uid())::text
  )
)
with check (
  bucket_id = 'patient-documents'
  and (
    private.is_admin()
    or owner_id = (select auth.uid())::text
  )
);

create policy patient_documents_delete
on storage.objects for delete
to authenticated
using (
  bucket_id = 'patient-documents'
  and (
    private.is_admin()
    or owner_id = (select auth.uid())::text
    or (storage.foldername(name))[1] = (select auth.uid())::text
  )
);

