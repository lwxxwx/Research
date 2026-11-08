BEGIN;

-- Running upgrade s2_f07_case_benchmark -> s2j_g2_roles

ALTER TABLE users ADD COLUMN role VARCHAR(32) DEFAULT 'engineer' NOT NULL;

ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT true NOT NULL;

ALTER TABLE users ADD CONSTRAINT check_users_role CHECK (role IN ('admin','engineer','reviewer','rule_admin'));

CREATE INDEX idx_users_role ON users (role);

ALTER TABLE review_result ADD COLUMN created_by BIGINT;

ALTER TABLE review_result ADD CONSTRAINT fk_review_result_created_by_users FOREIGN KEY(created_by) REFERENCES users (id) ON DELETE SET NULL;

CREATE INDEX idx_review_result_created_by ON review_result (created_by);

ALTER TABLE rule_candidates ADD COLUMN reviewed_by BIGINT;

ALTER TABLE rule_candidates ADD CONSTRAINT fk_rule_candidates_reviewed_by_users FOREIGN KEY(reviewed_by) REFERENCES users (id) ON DELETE SET NULL;

CREATE INDEX idx_rule_candidates_reviewed_by ON rule_candidates (reviewed_by);

UPDATE alembic_version SET version_num='s2j_g2_roles' WHERE alembic_version.version_num = 's2_f07_case_benchmark';

COMMIT;


