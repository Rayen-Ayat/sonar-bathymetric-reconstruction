"""
jsf_overlap_check.py — Overlap-zone self-consistency check for the four
surface reconstruction algorithms reviewed in Ayat (2026).

Survey lawnmower patterns create zones where adjacent track lines both measure
the same patch of lakebed. Points in these zones come from very different moments
in the collection sequence and thus provide genuinely independent measurements of
the same depth. This script finds those zones and measures how consistently each
reconstruction algorithm interpolates depth within them.

Algorithms evaluated (per Literature Review, Sections 3–6):
  1. grid      — SciPy griddata interpolation
  2. delaunay  — 2-D Delaunay + percentile edge filter
  3. poisson   — Poisson Surface Reconstruction
  4. bpa       — Ball Pivoting Algorithm

Method:
  1. Group points into ping blocks (400 pts/block = one sonar ping).
  2. Build a 1m cell grid. Flag cells containing points from blocks separated by
     more than 50 pings as "overlap zones" (inter-swath overlaps).
  3. For each algorithm, reconstruct from the full point cloud.
  4. At each overlap-zone point, interpolate the mesh surface (linear
     interpolation on mesh vertices) and compute residual: predicted_Z − actual_Z.
  5. Report bias (mean residual), precision (std), and RMSE per algorithm.
  6. Export residual distributions as violin plots.

Usage:
    python jsf_overlap_check.py --lake brunette [options]

Options:
    --lake           brunette / hunters / mclain  (required)
    --block-size N   Points per ping block (default: 400)
    --min-block-gap  Minimum ping-block gap to count as overlap (default: 50)
    --cell-size C    Grid cell size in metres (default: 1.0)
    --output-dir D   Output directory (default: evaluation_out/)
    --grid-step S    Grid interpolation step in metres (default: 1.0)
    --edge-pct P     Delaunay edge filter percentile (default: 95)
    --poisson-depth  Poisson octree depth (default: 9)
    --poisson-trim T Poisson density trim percentile (default: 5)
    --normal-k K     kNN for normal estimation, poisson/bpa (default: 30)
    --verbose        Show Open3D output

Outputs:
    overlap_results.csv     Per-algorithm consistency metrics
    overlap_residuals.png   Violin plots of residual distributions
"""

import os
import sys
import gc
import csv
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import jsf_merge as jm

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

# Re-use algorithm runners from jsf_crossval
from jsf_crossval import run_algo, LAKE_CONFIG


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OverlapResult:
    name:          str
    n_overlap_pts: int   = 0
    n_covered:     int   = 0
    coverage:      float = 0.0
    bias_m:        float = float('nan')
    std_m:         float = float('nan')
    rmse_m:        float = float('nan')
    elapsed_s:     float = 0.0
    error:         str   = ''


# ---------------------------------------------------------------------------
# Step 1 — Load & ENU projection
# ---------------------------------------------------------------------------

def load_enu(npy_path: Path, z_scale: float = 1.0) -> Tuple[np.ndarray, float, float]:
    pts = np.load(npy_path)
    enu, lat_ref, lon_ref = jm.to_enu(pts, z_scale=z_scale)
    return enu.astype(np.float64), lat_ref, lon_ref


# ---------------------------------------------------------------------------
# Step 2 — Overlap zone detection
# ---------------------------------------------------------------------------

def detect_overlap_zones(
    enu_pts:       np.ndarray,
    block_size:    int   = 400,
    cell_size:     float = 1.0,
    min_block_gap: int   = 50,
) -> np.ndarray:
    """
    Find points that lie in grid cells also visited by pings far apart in
    collection order (i.e. inter-swath overlap zones).

    Args:
        enu_pts:       (N, 3) ENU point array in collection order.
        block_size:    Number of points per ping block (default: 400).
        cell_size:     Grid cell size in metres (default: 1.0 m).
        min_block_gap: Minimum block-index separation to count as overlap (default: 50).

    Returns:
        overlap_mask: Boolean array of length N; True = point in an overlap zone.
    """
    n         = len(enu_pts)
    block_idx = np.arange(n) // block_size

    x_min = enu_pts[:, 0].min()
    y_min = enu_pts[:, 1].min()
    ci = ((enu_pts[:, 0] - x_min) / cell_size).astype(np.int32)
    cj = ((enu_pts[:, 1] - y_min) / cell_size).astype(np.int32)

    max_cj   = int(cj.max()) + 1
    cell_key = ci * max_cj + cj

    order   = np.argsort(cell_key, kind='stable')
    s_keys  = cell_key[order]
    s_bidx  = block_idx[order]

    _, starts, counts = np.unique(s_keys, return_index=True, return_counts=True)

    overlap_mask = np.zeros(n, dtype=bool)
    for start, count in zip(starts, counts):
        if count < 2:
            continue
        seg = s_bidx[start : start + count]
        if int(seg.max()) - int(seg.min()) > min_block_gap:
            overlap_mask[order[start : start + count]] = True

    return overlap_mask


# ---------------------------------------------------------------------------
# Step 3 — Per-algorithm evaluation at overlap points
# ---------------------------------------------------------------------------

def evaluate_at_overlap(
    vertices:     np.ndarray,
    faces:        np.ndarray,
    overlap_enu:  np.ndarray,
) -> dict:
    """
    Predict Z at overlap-zone points via linear interpolation, then compute
    residuals (predicted − actual).
    """
    nan  = float('nan')
    base = dict(n_overlap_pts=len(overlap_enu), n_covered=0, coverage=0.0,
                bias_m=nan, std_m=nan, rmse_m=nan, residuals=np.array([]))

    if len(vertices) < 4 or len(faces) == 0 or len(overlap_enu) == 0:
        return base

    interp    = LinearNDInterpolator(vertices[:, :2], vertices[:, 2])
    z_pred    = interp(overlap_enu[:, :2])
    z_true    = overlap_enu[:, 2]

    covered   = ~np.isnan(z_pred)
    n_covered = int(covered.sum())

    if n_covered == 0:
        return dict(base, coverage=0.0)

    residuals = z_pred[covered] - z_true[covered]
    return dict(
        n_overlap_pts = len(overlap_enu),
        n_covered     = n_covered,
        coverage      = n_covered / len(overlap_enu),
        bias_m        = float(np.mean(residuals)),
        std_m         = float(np.std(residuals)),
        rmse_m        = float(np.sqrt(np.mean(residuals ** 2))),
        residuals     = residuals,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(results: List[OverlapResult], label: str) -> None:
    W = 78
    print(f"\n  Overlap-zone consistency — {label}")
    print("  " + "=" * W)
    hdr = (
        f"  {'Algorithm':<12} {'OvlpPts':>8} {'Cov%':>6} "
        f"{'Bias(m)':>9} {'Std(m)':>8} {'RMSE(m)':>9} {'Time(s)':>8}"
    )
    print(hdr)
    print("  " + "-" * W)
    for r in results:
        if r.error:
            print(f"  {r.name:<12}  ERROR: {r.error[:65]}")
            continue
        bias_s = f"{r.bias_m:+.3f}" if not np.isnan(r.bias_m) else "n/a"
        std_s  = f"{r.std_m:.3f}"   if not np.isnan(r.std_m)  else "n/a"
        rmse_s = f"{r.rmse_m:.3f}"  if not np.isnan(r.rmse_m) else "n/a"
        print(
            f"  {r.name:<12} {r.n_overlap_pts:>8,} {r.coverage*100:>6.1f} "
            f"{bias_s:>9} {std_s:>8} {rmse_s:>9} {r.elapsed_s:>8.1f}"
        )
    print("  " + "=" * W)


def save_csv(results: List[OverlapResult], path: Path) -> None:
    fieldnames = [
        'algorithm', 'n_overlap_pts', 'n_covered', 'coverage_pct',
        'bias_m', 'std_m', 'rmse_m', 'time_s', 'error',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'algorithm':     r.name,
                'n_overlap_pts': r.n_overlap_pts,
                'n_covered':     r.n_covered,
                'coverage_pct':  round(r.coverage * 100, 2),
                'bias_m':        '' if np.isnan(r.bias_m)  else round(r.bias_m, 4),
                'std_m':         '' if np.isnan(r.std_m)   else round(r.std_m, 4),
                'rmse_m':        '' if np.isnan(r.rmse_m)  else round(r.rmse_m, 4),
                'time_s':        round(r.elapsed_s, 2),
                'error':         r.error,
            })
    print(f"  CSV  → {path}")


def plot_violin(results: List[OverlapResult], out_dir: Path, label: str) -> None:
    """
    Save violin plots of residual distributions per algorithm.
    A violin plot shows full distribution shape — useful for spotting fat-tailed
    algorithms with occasional large errors.
    """
    valid = [(r.name, r._residuals) for r in results
             if not r.error and len(getattr(r, '_residuals', np.array([]))) > 0]
    if not valid:
        print("  No residual data to plot.")
        return

    names     = [v[0] for v in valid]
    residuals = [v[1] for v in valid]

    fig, ax = plt.subplots(figsize=(10, 5))
    parts   = ax.violinplot(residuals, showmedians=True, showextrema=True)

    cmap   = plt.cm.tab10
    colors = [cmap(i / max(len(names) - 1, 1)) for i in range(len(names))]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', label='zero bias')
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Residual — predicted Z − actual Z  (m)', fontsize=10)
    ax.set_title(f'Overlap-Zone Residual Distributions — {label}\n'
                 f'(grid · delaunay · poisson · bpa)', fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = out_dir / 'overlap_residuals.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Overlap-zone self-consistency check for four '
                    'surface reconstruction algorithms (Ayat 2026).'
    )
    parser.add_argument('--lake', required=True, choices=list(LAKE_CONFIG.keys()))
    parser.add_argument('--block-size',    type=int,   default=400,  metavar='N')
    parser.add_argument('--min-block-gap', type=int,   default=50,   metavar='G')
    parser.add_argument('--cell-size',     type=float, default=1.0,  metavar='C')
    parser.add_argument('--output-dir',    default='evaluation_out', metavar='DIR')
    parser.add_argument('--grid-step',     type=float, default=1.0,  metavar='S')
    parser.add_argument('--edge-pct',      type=float, default=95,   metavar='P')
    parser.add_argument('--poisson-depth', type=int,   default=9,    metavar='D')
    parser.add_argument('--poisson-trim',  type=int,   default=5,    metavar='T')
    parser.add_argument('--normal-k',      type=int,   default=30,   metavar='K')
    parser.add_argument('--verbose',       action='store_true')
    args = parser.parse_args()

    if HAS_OPEN3D and not args.verbose:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg      = LAKE_CONFIG[args.lake]
    npy_path = cfg['npy']
    label    = cfg['label']

    if not npy_path.exists():
        print(f"ERROR: {npy_path} not found.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  Overlap-zone check — {label}")
    print(f"{'='*65}")

    print("  Loading point cloud ...", end='  ', flush=True)
    enu, lat_ref, lon_ref = load_enu(npy_path)
    print(f"{len(enu):,} points")

    print(
        f"  Detecting overlap zones "
        f"(block_size={args.block_size}, cell={args.cell_size}m, "
        f"min_gap={args.min_block_gap} blocks) ...",
        end='  ', flush=True,
    )
    overlap_mask = detect_overlap_zones(
        enu,
        block_size    = args.block_size,
        cell_size     = args.cell_size,
        min_block_gap = args.min_block_gap,
    )
    n_overlap = int(overlap_mask.sum())
    pct       = 100 * n_overlap / len(enu)
    print(f"{n_overlap:,} overlap points ({pct:.1f}% of survey)")

    if n_overlap == 0:
        print("  No overlap zones detected. Try reducing --min-block-gap.")
        sys.exit(0)

    overlap_enu = enu[overlap_mask]

    algo_names = ['grid', 'delaunay']
    if HAS_OPEN3D:
        algo_names += ['poisson', 'bpa']
    else:
        print("  open3d not available — skipping poisson and bpa")

    results: List[OverlapResult] = []

    for name in algo_names:
        print(f"\n  [{name}] reconstructing from full cloud ...", end='  ', flush=True)
        r  = OverlapResult(name=name, n_overlap_pts=n_overlap)
        t0 = time.perf_counter()

        try:
            vertices, faces = run_algo(
                name, enu,
                grid_step     = args.grid_step,
                edge_pct      = args.edge_pct,
                poisson_depth = args.poisson_depth,
                poisson_trim  = args.poisson_trim,
                normal_k      = args.normal_k,
            )
            recon_time = time.perf_counter() - t0
            gc.collect()

            print(f"done ({recon_time:.1f}s)")
            print(f"    evaluating at {n_overlap:,} overlap points ...", end='  ', flush=True)

            metrics = evaluate_at_overlap(vertices, faces, overlap_enu)
            r.n_covered  = metrics['n_covered']
            r.coverage   = metrics['coverage']
            r.bias_m     = metrics['bias_m']
            r.std_m      = metrics['std_m']
            r.rmse_m     = metrics['rmse_m']
            r._residuals = metrics['residuals']
            r.elapsed_s  = time.perf_counter() - t0

            bias_s = f"{r.bias_m:+.3f}m" if not np.isnan(r.bias_m) else "n/a"
            print(
                f"coverage={r.coverage*100:.1f}%  "
                f"bias={bias_s}  std={r.std_m:.3f}m  RMSE={r.rmse_m:.3f}m"
            )

        except Exception as exc:
            r.elapsed_s  = time.perf_counter() - t0
            r.error      = str(exc)
            r._residuals = np.array([])
            print(f"FAILED: {exc}")

        results.append(r)
        del vertices, faces
        gc.collect()

    print_results_table(results, label)
    save_csv(results, out_dir / 'overlap_results.csv')
    plot_violin(results, out_dir, label)
    print(f"\n  All outputs in: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
