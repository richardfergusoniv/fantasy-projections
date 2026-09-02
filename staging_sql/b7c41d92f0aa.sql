C:\Users\rdfer\Projects\fantasy-projections\.venv\Lib\site-packages\polars\meta\build.py:5: UserWarning: Polars binary is missing!
  from polars._utils.polars_version import get_polars_version
BEGIN;

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL, 
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Running upgrade  -> e53ebac3a6e5

CREATE TABLE app_user (
    id VARCHAR(36) NOT NULL, 
    email VARCHAR(320) NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    UNIQUE (email)
);

CREATE TABLE assistant_audit (
    id VARCHAR(36) NOT NULL, 
    user_hash VARCHAR(64) NOT NULL, 
    request_class VARCHAR(64) NOT NULL, 
    tools_called JSON NOT NULL, 
    source_ids JSON NOT NULL, 
    model_id VARCHAR(64), 
    token_usage JSON NOT NULL, 
    estimated_cost_usd FLOAT, 
    latency_ms INTEGER, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id)
);

CREATE TABLE availability_event (
    id VARCHAR(36) NOT NULL, 
    player_id VARCHAR(64) NOT NULL, 
    event_type VARCHAR(32) NOT NULL, 
    active_from TIMESTAMP WITH TIME ZONE NOT NULL, 
    active_until TIMESTAMP WITH TIME ZONE, 
    cleared_at TIMESTAMP WITH TIME ZONE, 
    source_snapshot_id VARCHAR(36), 
    evidence_ids JSON NOT NULL, 
    policy_json JSON NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_availability_event_player_id ON availability_event (player_id);

CREATE TABLE decision_snapshot (
    id VARCHAR(36) NOT NULL, 
    kind VARCHAR(32) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    week INTEGER, 
    projection_run_id VARCHAR(36) NOT NULL, 
    roster_snapshot_id VARCHAR(36), 
    result_json JSON NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_decision_snapshot_league_id ON decision_snapshot (league_id);

CREATE TABLE depth_snapshot (
    id VARCHAR(36) NOT NULL, 
    season INTEGER NOT NULL, 
    as_of TIMESTAMP WITH TIME ZONE NOT NULL, 
    artifact_uri TEXT NOT NULL, 
    content_hash VARCHAR(64) NOT NULL, 
    PRIMARY KEY (id)
);

CREATE TABLE injury_evidence (
    id VARCHAR(36) NOT NULL, 
    player_id VARCHAR(64) NOT NULL, 
    published_at TIMESTAMP WITH TIME ZONE, 
    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    source_url TEXT NOT NULL, 
    source_title VARCHAR(512) NOT NULL, 
    claim_json JSON NOT NULL, 
    confidence FLOAT NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_injury_evidence_player_id ON injury_evidence (player_id);

CREATE TABLE job_run (
    id VARCHAR(36) NOT NULL, 
    job_name VARCHAR(64) NOT NULL, 
    correlation_id VARCHAR(64) NOT NULL, 
    status VARCHAR(32) NOT NULL, 
    attempt INTEGER NOT NULL, 
    idempotency_key VARCHAR(128), 
    started_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    finished_at TIMESTAMP WITH TIME ZONE, 
    duration_ms INTEGER, 
    error TEXT, 
    metadata_json JSON NOT NULL, 
    PRIMARY KEY (id), 
    UNIQUE (idempotency_key)
);

CREATE INDEX ix_job_run_correlation_id ON job_run (correlation_id);

CREATE INDEX ix_job_run_job_name ON job_run (job_name);

CREATE TABLE league (
    id VARCHAR(36) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    season INTEGER NOT NULL, 
    name VARCHAR(256) NOT NULL, 
    league_type VARCHAR(32) NOT NULL, 
    status VARCHAR(32) NOT NULL, 
    previous_league_id VARCHAR(64), 
    raw_json JSON NOT NULL, 
    PRIMARY KEY (id), 
    CONSTRAINT uq_league_season UNIQUE (league_id, season)
);

CREATE INDEX ix_league_league_id ON league (league_id);

CREATE TABLE league_draft_rule (
    id VARCHAR(36) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    rule VARCHAR(32) NOT NULL, 
    confirmed_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_league_draft_rule_league_id ON league_draft_rule (league_id);

CREATE TABLE league_member (
    id VARCHAR(36) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    user_id VARCHAR(64) NOT NULL, 
    roster_id INTEGER NOT NULL, 
    display_name VARCHAR(256) NOT NULL, 
    PRIMARY KEY (id), 
    CONSTRAINT uq_league_roster UNIQUE (league_id, roster_id)
);

CREATE INDEX ix_league_member_league_id ON league_member (league_id);

CREATE TABLE league_rule_snapshot (
    id VARCHAR(36) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    raw_json JSON NOT NULL, 
    normalized_json JSON NOT NULL, 
    contract_hash VARCHAR(64) NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_league_rule_snapshot_league_id ON league_rule_snapshot (league_id);

CREATE TABLE league_transaction (
    id VARCHAR(36) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    transaction_id VARCHAR(64) NOT NULL, 
    txn_type VARCHAR(32) NOT NULL, 
    status VARCHAR(32) NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    payload JSON NOT NULL, 
    PRIMARY KEY (id), 
    UNIQUE (transaction_id)
);

CREATE INDEX ix_league_transaction_league_id ON league_transaction (league_id);

CREATE TABLE magic_link_token (
    id VARCHAR(36) NOT NULL, 
    email VARCHAR(320) NOT NULL, 
    token_hash VARCHAR(128) NOT NULL, 
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    used_at TIMESTAMP WITH TIME ZONE, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    UNIQUE (token_hash)
);

CREATE INDEX ix_magic_link_token_email ON magic_link_token (email);

CREATE TABLE manager_state (
    id VARCHAR(36) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    roster_id INTEGER NOT NULL, 
    as_of TIMESTAMP WITH TIME ZONE NOT NULL, 
    label VARCHAR(32) NOT NULL, 
    probabilities_json JSON NOT NULL, 
    features_json JSON NOT NULL, 
    overridden_label VARCHAR(32), 
    PRIMARY KEY (id)
);

CREATE INDEX ix_manager_state_league_id ON manager_state (league_id);

CREATE TABLE manager_tendency (
    id VARCHAR(36) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    roster_id INTEGER NOT NULL, 
    as_of TIMESTAMP WITH TIME ZONE NOT NULL, 
    sample_size INTEGER NOT NULL, 
    features_json JSON NOT NULL, 
    model_version VARCHAR(32) NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_manager_tendency_league_id ON manager_tendency (league_id);

CREATE TABLE matchup_snapshot (
    id VARCHAR(36) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    week INTEGER NOT NULL, 
    roster_id INTEGER NOT NULL, 
    matchup_id INTEGER NOT NULL, 
    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    points FLOAT, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_matchup_snapshot_league_id ON matchup_snapshot (league_id);

CREATE TABLE player_identity (
    player_id VARCHAR(64) NOT NULL, 
    sleeper_id VARCHAR(64), 
    gsis_id VARCHAR(64), 
    name VARCHAR(256) NOT NULL, 
    position VARCHAR(8) NOT NULL, 
    team VARCHAR(8), 
    PRIMARY KEY (player_id)
);

CREATE INDEX ix_player_identity_sleeper_id ON player_identity (sleeper_id);

CREATE TABLE player_status_snapshot (
    id VARCHAR(36) NOT NULL, 
    player_id VARCHAR(64) NOT NULL, 
    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    status VARCHAR(32), 
    injury_status VARCHAR(32), 
    practice VARCHAR(32), 
    raw_json JSON NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_player_status_snapshot_player_id ON player_status_snapshot (player_id);

CREATE TABLE projection_run (
    id VARCHAR(36) NOT NULL, 
    mode VARCHAR(32) NOT NULL, 
    season INTEGER NOT NULL, 
    week INTEGER, 
    as_of TIMESTAMP WITH TIME ZONE NOT NULL, 
    model_version VARCHAR(64) NOT NULL, 
    input_hash VARCHAR(64) NOT NULL, 
    status VARCHAR(32) NOT NULL, 
    manifest_uri TEXT, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_projection_run_mode ON projection_run (mode);

CREATE TABLE promotion_event (
    id VARCHAR(36) NOT NULL, 
    mode VARCHAR(32) NOT NULL, 
    candidate_run_id VARCHAR(36) NOT NULL, 
    previous_run_id VARCHAR(36), 
    promoted BOOLEAN NOT NULL, 
    validation_json JSON NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id)
);

CREATE TABLE roster_snapshot (
    id VARCHAR(36) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    week INTEGER NOT NULL, 
    roster_id INTEGER NOT NULL, 
    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    players JSON NOT NULL, 
    starters JSON NOT NULL, 
    reserve JSON NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_roster_snapshot_league_id ON roster_snapshot (league_id);

CREATE TABLE sleeper_account (
    id VARCHAR(36) NOT NULL, 
    user_id VARCHAR(64) NOT NULL, 
    username VARCHAR(128) NOT NULL, 
    last_synced_at TIMESTAMP WITH TIME ZONE, 
    PRIMARY KEY (id), 
    UNIQUE (user_id)
);

CREATE TABLE source_snapshot (
    id VARCHAR(36) NOT NULL, 
    endpoint VARCHAR(256) NOT NULL, 
    request_params_json JSON NOT NULL, 
    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    body_hash VARCHAR(64) NOT NULL, 
    artifact_uri TEXT NOT NULL, 
    health_verdict VARCHAR(32) NOT NULL, 
    is_complete BOOLEAN NOT NULL, 
    PRIMARY KEY (id)
);

CREATE TABLE trade_proposal (
    id VARCHAR(36) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    created_by_roster_id INTEGER NOT NULL, 
    sides_json JSON NOT NULL, 
    direction VARCHAR(16) NOT NULL, 
    status VARCHAR(32) NOT NULL, 
    countered_by_id VARCHAR(36), 
    PRIMARY KEY (id)
);

CREATE INDEX ix_trade_proposal_league_id ON trade_proposal (league_id);

CREATE TABLE traded_pick (
    id VARCHAR(36) NOT NULL, 
    league_id VARCHAR(64) NOT NULL, 
    season INTEGER NOT NULL, 
    round INTEGER NOT NULL, 
    original_roster_id INTEGER NOT NULL, 
    owner_roster_id INTEGER NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_traded_pick_league_id ON traded_pick (league_id);

CREATE TABLE active_projection_pointer (
    id VARCHAR(36) NOT NULL, 
    mode VARCHAR(32) NOT NULL, 
    season INTEGER NOT NULL, 
    week INTEGER, 
    run_id VARCHAR(36) NOT NULL, 
    activated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    previous_run_id VARCHAR(36), 
    PRIMARY KEY (id), 
    FOREIGN KEY(run_id) REFERENCES projection_run (id), 
    CONSTRAINT uq_active_pointer UNIQUE (mode, season, week)
);

CREATE TABLE player_projection (
    id VARCHAR(36) NOT NULL, 
    run_id VARCHAR(36) NOT NULL, 
    player_id VARCHAR(64) NOT NULL, 
    team VARCHAR(8), 
    opponent VARCHAR(8), 
    availability_probability FLOAT, 
    mean_json JSON NOT NULL, 
    quantiles_json JSON NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(run_id) REFERENCES projection_run (id)
);

CREATE INDEX ix_player_projection_player_id ON player_projection (player_id);

CREATE INDEX ix_player_projection_run_id ON player_projection (run_id);

CREATE TABLE session_record (
    id VARCHAR(36) NOT NULL, 
    user_id VARCHAR(36) NOT NULL, 
    session_hash VARCHAR(128) NOT NULL, 
    csrf_token VARCHAR(64) NOT NULL, 
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES app_user (id), 
    UNIQUE (session_hash)
);

CREATE TABLE simulation_partition (
    id VARCHAR(36) NOT NULL, 
    run_id VARCHAR(36) NOT NULL, 
    partition_key VARCHAR(128) NOT NULL, 
    uri TEXT NOT NULL, 
    sha256 VARCHAR(64) NOT NULL, 
    draw_count INTEGER NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(run_id) REFERENCES projection_run (id)
);

CREATE INDEX ix_simulation_partition_run_id ON simulation_partition (run_id);

CREATE TABLE trade_evaluation (
    id VARCHAR(36) NOT NULL, 
    proposal_id VARCHAR(36) NOT NULL, 
    projection_run_id VARCHAR(36) NOT NULL, 
    objective_json JSON NOT NULL, 
    fairness_json JSON NOT NULL, 
    acceptance_json JSON NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(proposal_id) REFERENCES trade_proposal (id)
);

INSERT INTO alembic_version (version_num) VALUES ('e53ebac3a6e5') RETURNING alembic_version.version_num;

-- Running upgrade e53ebac3a6e5 -> b7c41d92f0aa

ALTER TABLE projection_run ADD COLUMN artifact_mode VARCHAR(32) DEFAULT 'derived';

CREATE UNIQUE INDEX uq_player_projection_run_player ON player_projection (run_id, player_id);

CREATE UNIQUE INDEX uq_simulation_partition_run_key ON simulation_partition (run_id, partition_key);

CREATE UNIQUE INDEX uq_active_pointer_season_long ON active_projection_pointer (mode, season) WHERE week IS NULL;

ALTER TABLE projection_run ALTER COLUMN id TYPE VARCHAR(128);

ALTER TABLE player_projection ALTER COLUMN run_id TYPE VARCHAR(128);

ALTER TABLE simulation_partition ALTER COLUMN run_id TYPE VARCHAR(128);

ALTER TABLE active_projection_pointer ALTER COLUMN run_id TYPE VARCHAR(128);

ALTER TABLE active_projection_pointer ALTER COLUMN previous_run_id TYPE VARCHAR(128);

ALTER TABLE promotion_event ALTER COLUMN candidate_run_id TYPE VARCHAR(128);

ALTER TABLE promotion_event ALTER COLUMN previous_run_id TYPE VARCHAR(128);

ALTER TABLE decision_snapshot ALTER COLUMN projection_run_id TYPE VARCHAR(128);

ALTER TABLE trade_evaluation ALTER COLUMN projection_run_id TYPE VARCHAR(128);

ALTER TABLE app_user ALTER COLUMN created_at SET DEFAULT now();

ALTER TABLE magic_link_token ALTER COLUMN created_at SET DEFAULT now();

ALTER TABLE session_record ALTER COLUMN created_at SET DEFAULT now();

ALTER TABLE league ALTER COLUMN status SET DEFAULT 'active';

ALTER TABLE league ALTER COLUMN raw_json SET DEFAULT '{}';

ALTER TABLE league_draft_rule ALTER COLUMN confirmed_at SET DEFAULT now();

ALTER TABLE league_rule_snapshot ALTER COLUMN raw_json SET DEFAULT '{}';

ALTER TABLE league_rule_snapshot ALTER COLUMN normalized_json SET DEFAULT '{}';

ALTER TABLE roster_snapshot ALTER COLUMN players SET DEFAULT '[]';

ALTER TABLE roster_snapshot ALTER COLUMN starters SET DEFAULT '[]';

ALTER TABLE roster_snapshot ALTER COLUMN reserve SET DEFAULT '[]';

ALTER TABLE league_transaction ALTER COLUMN payload SET DEFAULT '{}';

ALTER TABLE player_status_snapshot ALTER COLUMN raw_json SET DEFAULT '{}';

ALTER TABLE injury_evidence ALTER COLUMN claim_json SET DEFAULT '{}';

ALTER TABLE injury_evidence ALTER COLUMN confidence SET DEFAULT 0.5;

ALTER TABLE availability_event ALTER COLUMN evidence_ids SET DEFAULT '[]';

ALTER TABLE availability_event ALTER COLUMN policy_json SET DEFAULT '{}';

ALTER TABLE projection_run ALTER COLUMN status SET DEFAULT 'candidate';

ALTER TABLE player_projection ALTER COLUMN mean_json SET DEFAULT '{}';

ALTER TABLE player_projection ALTER COLUMN quantiles_json SET DEFAULT '{}';

ALTER TABLE active_projection_pointer ALTER COLUMN activated_at SET DEFAULT now();

ALTER TABLE decision_snapshot ALTER COLUMN result_json SET DEFAULT '{}';

ALTER TABLE decision_snapshot ALTER COLUMN created_at SET DEFAULT now();

ALTER TABLE manager_state ALTER COLUMN probabilities_json SET DEFAULT '{}';

ALTER TABLE manager_state ALTER COLUMN features_json SET DEFAULT '{}';

ALTER TABLE trade_proposal ALTER COLUMN created_at SET DEFAULT now();

ALTER TABLE trade_proposal ALTER COLUMN sides_json SET DEFAULT '{}';

ALTER TABLE trade_proposal ALTER COLUMN status SET DEFAULT 'offered';

ALTER TABLE trade_evaluation ALTER COLUMN objective_json SET DEFAULT '{}';

ALTER TABLE trade_evaluation ALTER COLUMN fairness_json SET DEFAULT '{}';

ALTER TABLE trade_evaluation ALTER COLUMN acceptance_json SET DEFAULT '{}';

ALTER TABLE manager_tendency ALTER COLUMN sample_size SET DEFAULT 0;

ALTER TABLE manager_tendency ALTER COLUMN features_json SET DEFAULT '{}';

ALTER TABLE job_run ALTER COLUMN status SET DEFAULT 'running';

ALTER TABLE job_run ALTER COLUMN attempt SET DEFAULT 1;

ALTER TABLE job_run ALTER COLUMN started_at SET DEFAULT now();

ALTER TABLE job_run ALTER COLUMN metadata_json SET DEFAULT '{}';

ALTER TABLE source_snapshot ALTER COLUMN health_verdict SET DEFAULT 'healthy';

ALTER TABLE source_snapshot ALTER COLUMN is_complete SET DEFAULT true;

ALTER TABLE promotion_event ALTER COLUMN promoted SET DEFAULT false;

ALTER TABLE promotion_event ALTER COLUMN validation_json SET DEFAULT '{}';

ALTER TABLE promotion_event ALTER COLUMN created_at SET DEFAULT now();

ALTER TABLE assistant_audit ALTER COLUMN tools_called SET DEFAULT '[]';

ALTER TABLE assistant_audit ALTER COLUMN source_ids SET DEFAULT '[]';

ALTER TABLE assistant_audit ALTER COLUMN token_usage SET DEFAULT '{}';

ALTER TABLE assistant_audit ALTER COLUMN created_at SET DEFAULT now();

ALTER TABLE decision_snapshot ADD CONSTRAINT fk_decision_snapshot_projection_run FOREIGN KEY(projection_run_id) REFERENCES projection_run (id);

ALTER TABLE trade_evaluation ADD CONSTRAINT fk_trade_evaluation_projection_run FOREIGN KEY(projection_run_id) REFERENCES projection_run (id);

ALTER TABLE promotion_event ADD CONSTRAINT fk_promotion_event_candidate_run FOREIGN KEY(candidate_run_id) REFERENCES projection_run (id);

ALTER TABLE promotion_event ADD CONSTRAINT fk_promotion_event_previous_run FOREIGN KEY(previous_run_id) REFERENCES projection_run (id);

ALTER TABLE active_projection_pointer ADD CONSTRAINT fk_active_pointer_previous_run FOREIGN KEY(previous_run_id) REFERENCES projection_run (id);

UPDATE alembic_version SET version_num='b7c41d92f0aa' WHERE alembic_version.version_num = 'e53ebac3a6e5';