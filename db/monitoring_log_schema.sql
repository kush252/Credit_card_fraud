create table public.monitoring_logs (
  id serial not null,
  run_id uuid null,
  model_version text null,
  data_drift_detected boolean null,
  concept_drift_detected boolean null,
  retraining_triggered boolean null,
  timestamp timestamp without time zone null default CURRENT_TIMESTAMP,
  constraint monitoring_logs_pkey primary key (id)
) TABLESPACE pg_default;