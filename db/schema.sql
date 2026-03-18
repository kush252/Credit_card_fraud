-- prediction_logs table
create table prediction_logs (
    id bigserial primary key,
    timestamp timestamptz default now(),

    v1 double precision,
    v2 double precision,
    v3 double precision,
    v4 double precision,
    v5 double precision,
    v6 double precision,
    v7 double precision,
    v8 double precision,
    v9 double precision,
    v10 double precision,
    v11 double precision,
    v12 double precision,
    v13 double precision,
    v14 double precision,
    v15 double precision,
    v16 double precision,
    v17 double precision,
    v18 double precision,
    v19 double precision,
    v20 double precision,
    v21 double precision,
    v22 double precision,
    v23 double precision,
    v24 double precision,
    v25 double precision,
    v26 double precision,
    v27 double precision,
    v28 double precision,

    prediction integer,
    probability double precision,
    model_version text,
    scaled_amount double precision,
    scaled_time double precision
);

-- index for time-based queries
create index idx_prediction_logs_timestamp 
on prediction_logs (timestamp);

-- index for filtering predictions
create index idx_prediction_logs_prediction 
on prediction_logs (prediction);

-- documentation
comment on table prediction_logs is 
'Stores all prediction logs for fraud detection monitoring (data drift, concept drift)';