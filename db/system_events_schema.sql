create table public.system_events (
  id serial not null,
  run_id uuid null,
  event_type text null,
  status text null,
  message text null,
  timestamp timestamp without time zone null default CURRENT_TIMESTAMP,
  constraint system_events_pkey primary key (id)
) TABLESPACE pg_default;