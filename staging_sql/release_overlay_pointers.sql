C:\Users\rdfer\Projects\fantasy-projections\.venv\Lib\site-packages\polars\meta\build.py:5: UserWarning: Polars binary is missing!
  from polars._utils.polars_version import get_polars_version
BEGIN;

-- Running upgrade d4a1f6c28b57 -> f1e2d3c4b5a6

CREATE TABLE release_pointer (
    id VARCHAR(36) NOT NULL, 
    season INTEGER NOT NULL, 
    namespace VARCHAR(128) NOT NULL, 
    release_id VARCHAR(64) NOT NULL, 
    manifest_sha256 VARCHAR(64) NOT NULL, 
    manifest_storage_uri TEXT, 
    status VARCHAR(32) DEFAULT 'active' NOT NULL, 
    pointer_json JSON DEFAULT '{}' NOT NULL, 
    activated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (id), 
    UNIQUE (season)
);

CREATE INDEX ix_release_pointer_season ON release_pointer (season);

CREATE TABLE release_pointer_history (
    id VARCHAR(36) NOT NULL, 
    season INTEGER NOT NULL, 
    pointer_json JSON DEFAULT '{}' NOT NULL, 
    reason VARCHAR(64) DEFAULT 'promote' NOT NULL, 
    activated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_release_pointer_history_season ON release_pointer_history (season);

CREATE TABLE status_overlay_pointer (
    id VARCHAR(36) NOT NULL, 
    season INTEGER NOT NULL, 
    overlay_hash VARCHAR(64) NOT NULL, 
    base_release_id VARCHAR(64) NOT NULL, 
    base_manifest_sha256 VARCHAR(64) NOT NULL, 
    artifact_uri TEXT NOT NULL, 
    adjustment_count INTEGER DEFAULT '0' NOT NULL, 
    algorithm_version VARCHAR(64) NOT NULL, 
    pointer_json JSON DEFAULT '{}' NOT NULL, 
    activated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (id), 
    UNIQUE (season)
);

CREATE INDEX ix_status_overlay_pointer_season ON status_overlay_pointer (season);

CREATE TABLE status_overlay_pointer_history (
    id VARCHAR(36) NOT NULL, 
    season INTEGER NOT NULL, 
    pointer_json JSON DEFAULT '{}' NOT NULL, 
    reason VARCHAR(64) DEFAULT 'promote' NOT NULL, 
    activated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_status_overlay_pointer_history_season ON status_overlay_pointer_history (season);

CREATE TABLE job_lease (
    job_name VARCHAR(64) NOT NULL, 
    holder_id VARCHAR(64) NOT NULL, 
    lease_until TIMESTAMP WITH TIME ZONE NOT NULL, 
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (job_name)
);

CREATE TABLE job_outbox (
    id VARCHAR(36) NOT NULL, 
    job_name VARCHAR(64) NOT NULL, 
    idempotency_key VARCHAR(128) NOT NULL, 
    status VARCHAR(32) DEFAULT 'queued' NOT NULL, 
    holder_id VARCHAR(64), 
    scheduled_at TIMESTAMP WITH TIME ZONE, 
    claimed_at TIMESTAMP WITH TIME ZONE, 
    started_at TIMESTAMP WITH TIME ZONE, 
    finished_at TIMESTAMP WITH TIME ZONE, 
    attempt INTEGER DEFAULT '0' NOT NULL, 
    error TEXT, 
    metadata_json JSON DEFAULT '{}' NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (id), 
    UNIQUE (idempotency_key)
);

CREATE INDEX ix_job_outbox_job_name ON job_outbox (job_name);

CREATE TABLE rate_limit_bucket (
    bucket_key VARCHAR(256) NOT NULL, 
    window_start TIMESTAMP WITH TIME ZONE NOT NULL, 
    event_count INTEGER DEFAULT '0' NOT NULL, 
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (bucket_key)
);

UPDATE alembic_version SET version_num='f1e2d3c4b5a6' WHERE alembic_version.version_num = 'd4a1f6c28b57';