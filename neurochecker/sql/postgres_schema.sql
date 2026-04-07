CREATE TABLE IF NOT EXISTS datasets (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    source_root TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS neurons (
    id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    external_id TEXT,
    source_type TEXT NOT NULL,
    export_root TEXT,
    pixel_size_xy DOUBLE PRECISION,
    slice_thickness_z DOUBLE PRECISION,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (dataset_id, name)
);

CREATE TABLE IF NOT EXISTS segmentation_runs (
    id BIGSERIAL PRIMARY KEY,
    neuron_id BIGINT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL,
    direction TEXT,
    run_started TEXT,
    color_r INTEGER,
    color_g INTEGER,
    color_b INTEGER,
    max_width INTEGER NOT NULL DEFAULT 0,
    max_height INTEGER NOT NULL DEFAULT 0,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (neuron_id, run_id)
);

CREATE TABLE IF NOT EXISTS source_masks (
    id BIGSERIAL PRIMARY KEY,
    neuron_id BIGINT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
    run_db_id BIGINT REFERENCES segmentation_runs(id) ON DELETE SET NULL,
    mask_id TEXT NOT NULL,
    frame INTEGER NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    full_width INTEGER NOT NULL DEFAULT 0,
    full_height INTEGER NOT NULL DEFAULT 0,
    path TEXT,
    color_r INTEGER,
    color_g INTEGER,
    color_b INTEGER,
    flagged BOOLEAN NOT NULL DEFAULT FALSE,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (neuron_id, mask_id)
);

CREATE TABLE IF NOT EXISTS segments (
    id BIGSERIAL PRIMARY KEY,
    neuron_id BIGINT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL,
    segment_path TEXT NOT NULL,
    branch_label TEXT NOT NULL,
    parent_segment_index INTEGER,
    node_count INTEGER NOT NULL DEFAULT 0,
    frame_count INTEGER NOT NULL DEFAULT 0,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (neuron_id, segment_index),
    UNIQUE (neuron_id, segment_path)
);

CREATE TABLE IF NOT EXISTS skeleton_nodes (
    id BIGSERIAL PRIMARY KEY,
    neuron_id BIGINT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
    node_index INTEGER NOT NULL,
    frame INTEGER NOT NULL,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    z DOUBLE PRECISION,
    x_px DOUBLE PRECISION,
    y_px DOUBLE PRECISION,
    label TEXT,
    degree INTEGER,
    run_id TEXT,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (neuron_id, node_index)
);

CREATE TABLE IF NOT EXISTS skeleton_edges (
    id BIGSERIAL PRIMARY KEY,
    neuron_id BIGINT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
    source_node_db_id BIGINT NOT NULL REFERENCES skeleton_nodes(id) ON DELETE CASCADE,
    target_node_db_id BIGINT NOT NULL REFERENCES skeleton_nodes(id) ON DELETE CASCADE,
    UNIQUE (neuron_id, source_node_db_id, target_node_db_id)
);

CREATE TABLE IF NOT EXISTS segment_nodes (
    id BIGSERIAL PRIMARY KEY,
    segment_id BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    node_db_id BIGINT NOT NULL REFERENCES skeleton_nodes(id) ON DELETE CASCADE,
    node_order INTEGER NOT NULL,
    UNIQUE (segment_id, node_db_id),
    UNIQUE (segment_id, node_order)
);

CREATE TABLE IF NOT EXISTS segment_samples (
    id BIGSERIAL PRIMARY KEY,
    neuron_id BIGINT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
    segment_id BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    node_db_id BIGINT REFERENCES skeleton_nodes(id) ON DELETE SET NULL,
    run_db_id BIGINT REFERENCES segmentation_runs(id) ON DELETE SET NULL,
    source_mask_id BIGINT REFERENCES source_masks(id) ON DELETE SET NULL,
    run_id TEXT,
    mask_id TEXT,
    frame INTEGER NOT NULL,
    step INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    mask_path TEXT,
    has_mask BOOLEAN NOT NULL DEFAULT FALSE,
    flagged BOOLEAN NOT NULL DEFAULT FALSE,
    crop_x INTEGER,
    crop_y INTEGER,
    crop_w INTEGER,
    crop_h INTEGER,
    scale_factor DOUBLE PRECISION,
    centerline_x_px DOUBLE PRECISION,
    centerline_y_px DOUBLE PRECISION,
    centerline_x DOUBLE PRECISION,
    centerline_y DOUBLE PRECISION,
    centerline_z DOUBLE PRECISION,
    centroid_x_px DOUBLE PRECISION,
    centroid_y_px DOUBLE PRECISION,
    sample_center_x_px DOUBLE PRECISION,
    sample_center_y_px DOUBLE PRECISION,
    mask_x INTEGER,
    mask_y INTEGER,
    mask_w INTEGER,
    mask_h INTEGER,
    full_width INTEGER,
    full_height INTEGER,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (segment_id, frame, step)
);

CREATE TABLE IF NOT EXISTS reviewers (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS review_assignments (
    id BIGSERIAL PRIMARY KEY,
    sample_id BIGINT NOT NULL REFERENCES segment_samples(id) ON DELETE CASCADE,
    reviewer_id BIGINT NOT NULL REFERENCES reviewers(id) ON DELETE CASCADE,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source TEXT NOT NULL DEFAULT 'import',
    UNIQUE (sample_id, reviewer_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    id BIGSERIAL PRIMARY KEY,
    sample_id BIGINT NOT NULL REFERENCES segment_samples(id) ON DELETE CASCADE,
    reviewer_id BIGINT NOT NULL REFERENCES reviewers(id) ON DELETE CASCADE,
    verdict TEXT NOT NULL CHECK (verdict IN ('good', 'bad', 'pending')),
    delta_mask_path TEXT,
    delta_mask_png BYTEA,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_json JSONB,
    UNIQUE (sample_id, reviewer_id)
);

CREATE INDEX IF NOT EXISTS idx_neurons_dataset_name
    ON neurons (dataset_id, name);

CREATE INDEX IF NOT EXISTS idx_runs_neuron_run
    ON segmentation_runs (neuron_id, run_id);

CREATE INDEX IF NOT EXISTS idx_source_masks_neuron_frame
    ON source_masks (neuron_id, frame);

CREATE INDEX IF NOT EXISTS idx_segments_neuron_path
    ON segments (neuron_id, segment_path);

CREATE INDEX IF NOT EXISTS idx_skeleton_nodes_neuron_frame
    ON skeleton_nodes (neuron_id, frame);

CREATE INDEX IF NOT EXISTS idx_segment_samples_segment_step
    ON segment_samples (segment_id, step, frame);

CREATE INDEX IF NOT EXISTS idx_segment_samples_neuron_frame
    ON segment_samples (neuron_id, frame);

CREATE INDEX IF NOT EXISTS idx_review_assignments_sample
    ON review_assignments (sample_id);

CREATE INDEX IF NOT EXISTS idx_reviews_sample
    ON reviews (sample_id);

CREATE INDEX IF NOT EXISTS idx_reviews_reviewer
    ON reviews (reviewer_id);
