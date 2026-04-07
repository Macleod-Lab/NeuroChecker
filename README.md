# NeuroChecker

NeuroChecker is a companion tool for NeuroTracer. It computes per-mask
centroids, builds a simplified skeleton graph (MST or frame-linked), and
provides a GUI to navigate crops centered on the segmentation.

## Inputs
- NeuroTracer output folders like `neuron_<ID>/masks/` with `index.json` and
  `.npz` mask patches.
- The raw image stack directory for the same dataset.

## Outputs
- `skeleton.json`: 3D graph with endpoint and branchpoint labels (physical units).
- `sample_points.json`: per-mask centroid list (pixel + physical coords).

## GUI Usage
From inside this folder:

```bash
python -m neurochecker.gui
```

In the GUI:
1. Select the NeuroTracer data root (folder containing `neuron_<ID>` folders).
2. Select the image stack directory.
3. Pick a neuron from the list.
4. Use Prev/Next (or left/right arrow keys) to navigate frames.
5. Press `M` to open the interactive 3D map in your browser.

## CLI Usage

```bash
python -m neurochecker.cli --neuron-id 0 --data-root "C:\Users\Macleod Lab\Desktop\neurotracer-standalone"
```

Common options:

```bash
python -m neurochecker.cli --neuron-id 0 --data-root "C:\Users\Macleod Lab\Desktop\neurotracer-standalone" \
  --graph-mode mst --k-neighbors 6 --max-frame-gap 3 \
  --pixel-size-xy 0.00495 --slice-thickness-z 0.059
```

## PostgreSQL Review Database

For high-volume reviewer input, use the PostgreSQL-backed database tools:

```bash
python -m neurochecker.db_cli --dsn "postgresql://user:pass@host:5432/neurochecker" init
python -m neurochecker.db_cli --dsn "postgresql://user:pass@host:5432/neurochecker" import-export --export-root "C:\path\to\neurochecker_export_neuron_Bruce_20260301_184259"
python -m neurochecker.db_cli --dsn "postgresql://user:pass@host:5432/neurochecker" score --neuron neuron_Bruce --segment root/branch_1
```

The database schema stores:
- NeuroTracer runs and source masks
- NeuroChecker segment trees, skeleton nodes, edges, image crops, and sample metadata
- Reviewer assignments, verdicts, and optional delta-mask PNG payloads

Integration tests for the database are gated behind `NEUROCHECKER_TEST_DSN`.

## Notes
- Pixel coordinates are stored as `x_px`, `y_px`, and `z_frame`.
- Physical coordinates use `pixel_size_xy` and `slice_thickness_z` in the same units.
- The tool does not modify NeuroTracer masks on disk.
- The 3D map uses Plotly and opens in your default browser.

## Skeleton Format
`skeleton.json` is a graph:

```json
{
  "neuron_id": "0",
  "spacing": { "pixel_size_xy": 0.00495, "slice_thickness_z": 0.059 },
  "graph_mode": "mst",
  "nodes": [
    {
      "id": 0,
      "frame": 1717,
      "run_id": "c729a957...",
      "x_px": 12123.4,
      "y_px": 12678.0,
      "z_frame": 1717,
      "x": 60.01,
      "y": 62.76,
      "z": 101.30,
      "degree": 1,
      "label": "endpoint"
    }
  ],
  "edges": [[0, 1]],
  "counts": { "nodes": 100, "edges": 99, "endpoints": 12, "branchpoints": 6, "isolated": 1 }
}
```

