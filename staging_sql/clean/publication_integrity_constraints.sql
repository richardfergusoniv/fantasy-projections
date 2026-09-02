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
