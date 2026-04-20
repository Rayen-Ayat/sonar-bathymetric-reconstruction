"""
jsf_crossval.py — Ping-level cross-validation for the four surface reconstruction
algorithms reviewed in Ayat (2026).

Splits the merged point cloud into 80% train / 20% test using ping-aware blocks
(default: 400 pts/block, matching one sonar ping). Reconstructs from the train set
only, then evaluates each mesh by vertically interpolating Z at the held-out test
point (East, North) positions.

Because test pings are genuinely spatially distinct from training pings (they come
from different moments in the survey lawnmower pattern), RMSE is non-zero and
meaningful for ALL four algorithms, including Delaunay and BPA.

Algorithms evaluated (per Literature Review, Sections 3–6):
  1. grid      — SciPy griddata interpolation
  2. delaunay  — 2-D Delaunay + percentile edge filter
  3. poisson   — Poisson Surface Reconstruction
  4. bpa       — Ball Pivoting Algorithm

Usage:
    python jsf_crossval.py --lake brunette [options]
    python jsf_crossval.py --lake hunters
    python jsf_crossval.py --lake mclain

Options:
    --lake           brunette / hunters / mclain  (required)
    --block-size N   Points per ping block (default: 400)
    --train-ratio R  Fraction of blocks used for training (default: 0.8)
    --output-dir D   Where to write results (default: evaluation_out/)
    --grid-step S    Grid interpolation spacing in metres (default: 1.0)
    --edge-pct P     Delaunay edge-filter percentile (default: 95)
    --poisson-depth  Poisson octree depth (default: 9)
    --poisson-trim T Poisson density trim percentile (default: 5)
    --normal-k K     kNN for normal estimation, poisson/bpa (default: 30)
    --verbose        Print Open3D output (suppressed by default)

Outputs (in --output-dir):
    crossval_results.csv    Per-algorithm metrics
    crossval_results.png    Bar chart comparison
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import gc
import csv
import time
import argparse
from dataclasses import dataclass, field
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

# ---------------------------------------------------------------------------
# Lake configuration
# ---------------------------------------------------------------------------

LAKE_CONFIG: Dict[str, dict] = {
    'brunette': {
        'npy':   _HERE / 'output'         / 'brunette_park_merged.npy',
        'label': 'Brunette Park',
    },
    'hunters': {
        'npy':   _HERE / 'output_hunters' / 'hunters_point_merged.npy',
        'label': 'Hunters Point',
    },
    'mclain': {
        'npy':   _HERE / 'output_mclain'  / 'mclain_merged.npy',
        'label': 'McLain',
    },
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """Cross-validation metrics for one algorithm."""
    name:       str
    n_train:    int   = 0
    n_test:     int   = 0
    n_covered:  int   = 0
    coverage:   float = 0.0
    cv_rmse:    float = float('nan')
    mae:        float = float('nan')
    p90_error:  float = float('nan')
    combined:   float = float('nan')
    elapsed_s:  float = 0.0
    error:      str   = ''


# ---------------------------------------------------------------------------
# Step 1 — Load & ping-aware split
# ---------------------------------------------------------------------------

def load_and_split(
    npy_path:    Path,
    block_size:  int   = 400,
    train_ratio: float = 0.80,
    z_scale:     float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Load a merged .npy point cloud, project to ENU, and split into train/test
    using a ping-aware block strategy.

    Points are grouped into consecutive blocks of `block_size` points each, and
    every ceil(1 / (1-train_ratio))-th block is assigned to the test set.
    This spreads test pings throughout the survey rather than concentrating them
    at the end, ensuring genuine spatial independence between train and test sets.
    """
    pts = np.load(npy_path)
    enu, lat_ref, lon_ref = jm.to_enu(pts, z_scale=z_scale)
    enu = enu.astype(np.float64)
    del pts; gc.collect()

    n_pts    = len(enu)
    n_blocks = n_pts // block_size
    usable   = n_blocks * block_size

    block_of = np.arange(usable) // block_size
    k        = max(2, round(1.0 / max(1e-6, 1.0 - train_ratio)))
    test_blocks = set(range(k - 1, n_blocks, k))

    test_mask  = np.array([b in test_blocks for b in block_of], dtype=bool)
    train_mask = ~test_mask

    train_enu = enu[:usable][train_mask]
    test_enu  = enu[:usable][test_mask]

    n_test_blocks  = len(test_blocks)
    n_train_blocks = n_blocks - n_test_blocks
    print(
        f"  Split: {len(train_enu):,} train pts ({n_train_blocks} blocks)  |  "
        f"{len(test_enu):,} test pts ({n_test_blocks} blocks)  "
        f"[block_size={block_size}, k={k}]"
    )
    return train_enu, test_enu, lat_ref, lon_ref


# ---------------------------------------------------------------------------
# Step 2 — Algorithm runners (train set only, four algorithms)
# ---------------------------------------------------------------------------

def run_algo(
    name:          str,
    train_enu:     np.ndarray,
    grid_step:     float = 1.0,
    edge_pct:      float = 95.0,
    poisson_depth: int   = 9,
    poisson_trim:  int   = 5,
    normal_k:      int   = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one of the four literature-reviewed surface reconstruction algorithms
    on the training point set.

    Args:
        name:          Algorithm key: 'grid' | 'delaunay' | 'poisson' | 'bpa'.
        train_enu:     (N, 3) float64 ENU point array for training.
        grid_step:     Grid interpolation cell size in metres (grid algorithm).
        edge_pct:      Edge-length percentile filter for Delaunay (Section 4.4).
        poisson_depth: Octree depth for Poisson (Section 5.4; default 9).
        poisson_trim:  Density trim percentile for Poisson (Section 5.2; default 5).
        normal_k:      kNN for normal estimation, Poisson/BPA (Section 5.4; default 30).

    Returns:
        vertices: (V, 3) float64 mesh vertex array.
        faces:    (F, 3) int32   mesh face index array.
    """
    if name == 'grid':
        return jm.mesh_from_grid(train_enu, step=grid_step)

    if name == 'delaunay':
        faces = jm.mesh_from_delaunay(train_enu, edge_pct=edge_pct)
        return train_enu, faces

    if not HAS_OPEN3D:
        raise RuntimeError('open3d not available — install with: pip install open3d')

    if name == 'poisson':
        return jm.mesh_from_poisson(
            train_enu,
            depth=poisson_depth,
            trim_percentile=poisson_trim,
            normal_k=normal_k,
        )

    if name == 'bpa':
        return jm.mesh_from_bpa(train_enu, normal_k=normal_k)

    raise ValueError(f'Unknown algorithm: {name!r}. Choose grid | delaunay | poisson | bpa')


# ---------------------------------------------------------------------------
# Step 3 — Surface interpolation & evaluation
# ---------------------------------------------------------------------------

def compute_gap_fraction(vertices: np.ndarray, faces: np.ndarray,
                         cell_size: float = 1.0) -> float:
    if len(faces) == 0 or len(vertices) == 0:
        return 1.0
    xy    = vertices[:, :2]
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    nx = max(1, int((x_max - x_min) / cell_size) + 1)
    ny = max(1, int((y_max - y_min) / cell_size) + 1)
    covered = np.zeros((ny, nx), dtype=bool)
    v0 = xy[faces[:, 0]]; v1 = xy[faces[:, 1]]; v2 = xy[faces[:, 2]]
    xl = np.minimum(np.minimum(v0[:, 0], v1[:, 0]), v2[:, 0])
    xh = np.maximum(np.maximum(v0[:, 0], v1[:, 0]), v2[:, 0])
    yl = np.minimum(np.minimum(v0[:, 1], v1[:, 1]), v2[:, 1])
    yh = np.maximum(np.maximum(v0[:, 1], v1[:, 1]), v2[:, 1])
    ci0 = ((xl - x_min) / cell_size).astype(int).clip(0, nx - 1)
    ci1 = ((xh - x_min) / cell_size).astype(int).clip(0, nx - 1)
    cj0 = ((yl - y_min) / cell_size).astype(int).clip(0, ny - 1)
    cj1 = ((yh - y_min) / cell_size).astype(int).clip(0, ny - 1)
    for i in range(len(faces)):
        covered[cj0[i]:cj1[i] + 1, ci0[i]:ci1[i] + 1] = True
    return 1.0 - float(covered.sum()) / (nx * ny)


def make_surface_interpolator(vertices: np.ndarray):
    """
    Build a 2D→Z surface interpolator using scipy LinearNDInterpolator.
    Returns NaN for queries outside the convex hull of the mesh vertex footprint.
    """
    return LinearNDInterpolator(vertices[:, :2], vertices[:, 2])


def evaluate_algorithm(
    vertices:  np.ndarray,
    faces:     np.ndarray,
    test_enu:  np.ndarray,
) -> dict:
    """
    Evaluate a reconstructed mesh against held-out test points.

    Metrics:
        coverage:   fraction of test points with a valid prediction [0, 1]
        cv_rmse:    RMSE on covered points (metres)
        mae:        mean absolute error on covered points (metres)
        p90_error:  90th-percentile absolute error on covered points (metres)
        combined:   cv_rmse / (1 - gap_fraction)  [lower is better]
    """
    nan   = float('nan')
    empty = dict(coverage=0.0, cv_rmse=nan, mae=nan, p90_error=nan, combined=nan,
                 n_covered=0)

    if len(vertices) < 4 or len(faces) == 0:
        return empty

    interp    = make_surface_interpolator(vertices)
    z_pred    = interp(test_enu[:, :2])
    z_true    = test_enu[:, 2]

    covered   = ~np.isnan(z_pred)
    n_covered = int(covered.sum())
    coverage  = n_covered / len(test_enu)

    if n_covered == 0:
        return dict(empty, coverage=coverage)

    residuals    = np.abs(z_pred[covered] - z_true[covered])
    cv_rmse      = float(np.sqrt(np.mean(residuals ** 2)))
    mae          = float(np.mean(residuals))
    p90          = float(np.percentile(residuals, 90))
    gap_fraction = compute_gap_fraction(vertices, faces)
    combined     = cv_rmse / max(1.0 - gap_fraction, 1e-6)

    return dict(
        coverage=coverage, cv_rmse=cv_rmse, mae=mae,
        p90_error=p90, combined=combined, n_covered=n_covered,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(results: List[CVResult], label: str) -> None:
    W = 84
    print(f"\n  Cross-validation results — {label}")
    print("  " + "=" * W)
    hdr = (
        f"  {'Algorithm':<12} {'Train':>7} {'Test':>7} {'Cov%':>6} "
        f"{'CV-RMSE(m)':>11} {'MAE(m)':>8} {'P90(m)':>8} {'Combined':>10} {'Time(s)':>8}"
    )
    print(hdr)
    print("  " + "-" * W)
    for r in results:
        if r.error:
            print(f"  {r.name:<12}  ERROR: {r.error[:65]}")
            continue
        rmse_s  = f"{r.cv_rmse:.3f}"   if not np.isnan(r.cv_rmse)   else "n/a"
        mae_s   = f"{r.mae:.3f}"       if not np.isnan(r.mae)        else "n/a"
        p90_s   = f"{r.p90_error:.3f}" if not np.isnan(r.p90_error)  else "n/a"
        comb_s  = f"{r.combined:.3f}"  if not np.isnan(r.combined)   else "n/a"
        print(
            f"  {r.name:<12} {r.n_train:>7,} {r.n_test:>7,} "
            f"{r.coverage*100:>6.1f} {rmse_s:>11} {mae_s:>8} "
            f"{p90_s:>8} {comb_s:>10} {r.elapsed_s:>8.1f}"
        )
    print("  " + "=" * W)

    valid = [r for r in results if not r.error and not np.isnan(r.combined)]
    if valid:
        ranked = sorted(valid, key=lambda r: r.combined)
        print(f"\n  Ranking by combined score (CV-RMSE / (1 - gap_fraction), lower = better):")
        for i, r in enumerate(ranked, 1):
            flag = ""
            if r.coverage < 0.80:
                flag = "  ⚠ coverage < 80% — gaps present"
            if r.cv_rmse > 1.0:
                flag += "  ⚠ CV-RMSE > 1 m — poor accuracy"
            print(f"    #{i}  {r.name:<12}  combined={r.combined:.3f}{flag}")


def save_csv(results: List[CVResult], path: Path, lake_key: str = '') -> None:
    fieldnames = [
        'lake', 'algorithm', 'n_train', 'n_test', 'n_covered', 'coverage_pct',
        'cv_rmse_m', 'mae_m', 'p90_error_m', 'combined_score', 'time_s', 'error',
    ]
    write_header = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow({
                'lake':          lake_key,
                'algorithm':     r.name,
                'n_train':       r.n_train,
                'n_test':        r.n_test,
                'n_covered':     r.n_covered,
                'coverage_pct':  round(r.coverage * 100, 2),
                'cv_rmse_m':     '' if np.isnan(r.cv_rmse)   else round(r.cv_rmse, 4),
                'mae_m':         '' if np.isnan(r.mae)        else round(r.mae, 4),
                'p90_error_m':   '' if np.isnan(r.p90_error)  else round(r.p90_error, 4),
                'combined_score':'' if np.isnan(r.combined)   else round(r.combined, 4),
                'time_s':        round(r.elapsed_s, 2),
                'error':         r.error,
            })
    print(f"  CSV  → {path}")


def plot_results(results: List[CVResult], out_dir: Path, label: str) -> None:
    """Save a 3-panel bar chart: CV-RMSE, coverage, combined score."""
    valid = [r for r in results if not r.error]
    if not valid:
        print("  No valid results to plot.")
        return

    algos    = [r.name for r in valid]
    cv_rmse  = [r.cv_rmse   if not np.isnan(r.cv_rmse)  else 0 for r in valid]
    cov      = [r.coverage * 100                                for r in valid]
    combined = [r.combined  if not np.isnan(r.combined) else 0 for r in valid]

    cmap   = plt.cm.tab10
    colors = [cmap(i / max(len(algos) - 1, 1)) for i in range(len(algos))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    panels = [
        (axes[0], cv_rmse,  'CV-RMSE (m)',           False),
        (axes[1], cov,      'Coverage (%)',           True),
        (axes[2], combined, 'Combined score\n(lower is better)', False),
    ]
    for ax, vals, ylabel, higher_better in panels:
        bars = ax.bar(algos, vals, color=colors, edgecolor='white')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        if vals:
            best = int(np.argmin(vals) if not higher_better else np.argmax(vals))
            bars[best].set_edgecolor('gold')
            bars[best].set_linewidth(2.5)

    fig.suptitle(f'Cross-Validation Results — {label}\n'
                 f'(grid · delaunay · poisson · bpa)', fontsize=12)
    plt.tight_layout()
    out_path = out_dir / 'crossval_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Ping-level cross-validation for four lake bathymetry '
                    'reconstruction algorithms (Ayat 2026).'
    )
    parser.add_argument('--lake', required=True, choices=list(LAKE_CONFIG.keys()))
    parser.add_argument('--block-size',    type=int,   default=400,  metavar='N')
    parser.add_argument('--train-ratio',   type=float, default=0.80, metavar='R')
    parser.add_argument('--output-dir',    default='evaluation_out', metavar='DIR')
    parser.add_argument('--grid-step',     type=float, default=1.0,  metavar='S')
    parser.add_argument('--edge-pct',      type=float, default=95,   metavar='P')
    parser.add_argument('--poisson-depth', type=int,   default=9,    metavar='D',
                        help='Octree depth (default 9; Section 5.4 recommends 9-10)')
    parser.add_argument('--poisson-trim',  type=int,   default=5,    metavar='T',
                        help='Density trim percentile (default 5; Section 5.2)')
    parser.add_argument('--normal-k',      type=int,   default=30,   metavar='K',
                        help='kNN for normals, poisson/bpa (default 30; Section 5.4)')
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
    print(f"  Cross-validation — {label}")
    print(f"{'='*65}")

    train_enu, test_enu, lat_ref, lon_ref = load_and_split(
        npy_path, block_size=args.block_size, train_ratio=args.train_ratio
    )

    algo_names = ['grid', 'delaunay']
    if HAS_OPEN3D:
        algo_names += ['poisson', 'bpa']
    else:
        print("  open3d not available — skipping poisson and bpa")

    results: List[CVResult] = []

    for name in algo_names:
        print(f"\n  [{name}] reconstructing from train set ...", end='  ', flush=True)
        r  = CVResult(name=name, n_train=len(train_enu), n_test=len(test_enu))
        t0 = time.perf_counter()

        try:
            vertices, faces = run_algo(
                name, train_enu,
                grid_step     = args.grid_step,
                edge_pct      = args.edge_pct,
                poisson_depth = args.poisson_depth,
                poisson_trim  = args.poisson_trim,
                normal_k      = args.normal_k,
            )
            r.elapsed_s = time.perf_counter() - t0
            gc.collect()

            print(f"done ({r.elapsed_s:.1f}s, {len(vertices):,}v {len(faces):,}f)")
            print(f"    evaluating on {len(test_enu):,} test points ...", end='  ', flush=True)

            metrics = evaluate_algorithm(vertices, faces, test_enu)
            r.n_covered = metrics['n_covered']
            r.coverage  = metrics['coverage']
            r.cv_rmse   = metrics['cv_rmse']
            r.mae       = metrics['mae']
            r.p90_error = metrics['p90_error']
            r.combined  = metrics['combined']

            rmse_s = f"{r.cv_rmse:.3f}m" if not np.isnan(r.cv_rmse) else "n/a"
            print(
                f"coverage={r.coverage*100:.1f}%  CV-RMSE={rmse_s}  "
                f"combined={r.combined:.3f}"
            )

        except Exception as exc:
            r.elapsed_s = time.perf_counter() - t0
            r.error     = str(exc)
            print(f"FAILED: {exc}")

        results.append(r)
        del vertices, faces
        gc.collect()

    print_results_table(results, label)
    save_csv(results, out_dir / 'crossval_results.csv', lake_key=args.lake)
    plot_results(results, out_dir, label)
    print(f"\n  All outputs in: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
