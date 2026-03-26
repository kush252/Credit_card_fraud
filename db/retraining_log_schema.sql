create table public.retraining_logs (
  id serial not null,
  run_id uuid null,
  old_model_version text null,
  new_model_version text null,
  dataset_size integer null,
  metrics_before jsonb null,
  metrics_after jsonb null,
  retraining_trigger boolean null,
  status text null,
  timestamp timestamp without time zone null default CURRENT_TIMESTAMP,
  constraint retraining_logs_pkey primary key (id)
) TABLESPACE pg_default;