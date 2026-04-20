"""
benchmark_reconstruction.py — Benchmark the four surface reconstruction algorithms
reviewed in Ayat (2026) on lake bathymetric data.

Algorithms (per Literature Review, Sections 3–6):
  1. grid      — SciPy griddata linear interpolation (Section 3)
  2. delaunay  — 2-D Delaunay + percentile edge-length filter (Section 4)
  3. poisson   — Poisson Surface Reconstruction, Open3D (Section 5)
  4. bpa       — Ball Pivoting Algorithm, Open3D (Section 6)

Quality metrics:
  - Reconstruction time (s)
  - Vertex / triangle count
  - Triangle quality  (mean equilateral-ratio, 0–1, higher = better)
  - Coverage          (% of input points within 1 m of the mesh)
  - Gap fraction      (fraction of bounding-box area with no mesh)
  - Median edge length (m)
  - Watertightness

Usage:
    python benchmark_reconstruction.py [options]

    --lakes    brunette hunters mclain     (default: all three)
    --output-dir DIR                       (default: benchmark_out/)
    --grid-step S     grid interpolation spacing in metres (default 1.0)
    --edge-pct  P     Delaunay edge filter percentile      (default 95)
    --poisson-depth D Poisson octree depth                 (default 9)
    --poisson-trim T  Poisson density trim percentile      (default 5)
    --normal-k  K     kNN for normal estimation            (default 30)
    --max-pts   N     Subsample to N pts before Open3D algorithms (0=off)

Requires: numpy scipy matplotlib open3d (for poisson and bpa)
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
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).parent))
import jsf_merge as jm
import jsf_parser as jsp

# ---------------------------------------------------------------------------
# Optional Open3D import
# ---------------------------------------------------------------------------

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("WARNING: open3d not installed — Poisson and BPA algorithms unavailable.")
    print("         Install with:  pip install open3d\n")


# ---------------------------------------------------------------------------
# Lake configuration
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent

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
class AlgoResult:
    name:             str
    vertices:         int   = 0
    triangles:        int   = 0
    elapsed_s:        float = 0.0
    triangle_quality: float = 0.0
    coverage_pct:     float = 0.0
    gap_fraction:     float = 0.0
    median_edge_m:    float = 0.0
    is_watertight:    bool  = False
    cv_rmse:          float = float('nan')
    combined_score:   float = float('nan')
    error:            str   = ''


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_enu(npy_path: Path, z_scale: float = 1.0) -> Tuple[np.ndarray, float, float]:
    pts = np.load(npy_path)
    enu, lat_ref, lon_ref = jm.to_enu(pts, z_scale=z_scale)
    return enu.astype(np.float64), lat_ref, lon_ref


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------

def triangle_quality_score(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Mean equilateral-ratio quality over all triangles.
    q = (4 * sqrt(3) * area) / (e0^2 + e1^2 + e2^2)
    Perfect equilateral triangle → 1.0; degenerate → 0.
    """
    if len(faces) == 0:
        return 0.0
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    s  = (e0 + e1 + e2) / 2
    area_sq = np.maximum(s * (s - e0) * (s - e1) * (s - e2), 0.0)
    area    = np.sqrt(area_sq)
    denom   = e0**2 + e1**2 + e2**2
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.where(denom > 0, (4 * np.sqrt(3) * area) / denom, 0.0)
    return float(np.mean(q))


def coverage_and_watertight(
    vertices: np.ndarray,
    faces: np.ndarray,
    enu_pts: np.ndarray,
    sample_n: int = 10_000,
) -> Tuple[float, bool]:
    """
    Compute (coverage_pct, is_watertight) from the mesh and original point cloud.
    coverage_pct: fraction of enu_pts whose nearest mesh vertex is within 1 m.
    """
    if len(faces) == 0 or len(vertices) == 0:
        return 0.0, False

    is_wt = False
    if HAS_OPEN3D:
        try:
            is_wt = bool(numpy_to_o3d(vertices, faces).is_watertight())
        except Exception:
            pass

    from scipy.spatial import cKDTree
    n   = min(sample_n, len(enu_pts))
    idx = np.random.default_rng(42).choice(len(enu_pts), n, replace=False)
    pts = enu_pts[idx]
    tree  = cKDTree(vertices)
    dists, _ = tree.query(pts, k=1, workers=-1)
    coverage  = float(np.mean(dists < 1.0) * 100.0)
    return coverage, is_wt


def compute_gap_fraction(vertices: np.ndarray, faces: np.ndarray,
                         cell_size: float = 1.0) -> float:
    """
    Fraction of the survey bounding-box area not covered by any mesh triangle.
    """
    if len(faces) == 0 or len(vertices) == 0:
        return 1.0
    xy    = vertices[:, :2]
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    nx = max(1, int((x_max - x_min) / cell_size) + 1)
    ny = max(1, int((y_max - y_min) / cell_size) + 1)
    covered = np.zeros((ny, nx), dtype=bool)
    v0 = xy[faces[:, 0]]; v1 = xy[faces[:, 1]]; v2 = xy[faces[:, 2]]
    xl = np.minimum(np.minimum(v0[:,0], v1[:,0]), v2[:,0])
    xh = np.maximum(np.maximum(v0[:,0], v1[:,0]), v2[:,0])
    yl = np.minimum(np.minimum(v0[:,1], v1[:,1]), v2[:,1])
    yh = np.maximum(np.maximum(v0[:,1], v1[:,1]), v2[:,1])
    ci0 = ((xl - x_min) / cell_size).astype(int).clip(0, nx - 1)
    ci1 = ((xh - x_min) / cell_size).astype(int).clip(0, nx - 1)
    cj0 = ((yl - y_min) / cell_size).astype(int).clip(0, ny - 1)
    cj1 = ((yh - y_min) / cell_size).astype(int).clip(0, ny - 1)
    for i in range(len(faces)):
        covered[cj0[i]:cj1[i]+1, ci0[i]:ci1[i]+1] = True
    return 1.0 - float(covered.sum()) / (nx * ny)


def compute_median_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 0.0
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    return float(np.median(np.concatenate([e0, e1, e2])))


# ---------------------------------------------------------------------------
# Mesh conversion helpers
# ---------------------------------------------------------------------------

def numpy_to_o3d(vertices: np.ndarray, faces: np.ndarray):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    return mesh


def o3d_to_numpy(mesh) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


# ---------------------------------------------------------------------------
# Algorithm implementations (Section 3–6 of literature review)
# ---------------------------------------------------------------------------

def run_grid(enu_pts: np.ndarray, step: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 1 — SciPy griddata interpolation (Section 3).

    Linear mode: preserves measured values exactly, avoids oscillation on
    noisy data. Grid resolution of 1.0 m is within the recommended 0.5–2.0 m
    range for dense multibeam surveys. NaN fill explicitly marks uninterpolated
    regions rather than extrapolating beyond the convex hull.
    """
    return jm.mesh_from_grid(enu_pts, step=step)


def run_delaunay(enu_pts: np.ndarray, edge_pct: float = 95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 2 — 2-D Delaunay + percentile edge filter (Section 4).

    The 95th-percentile edge-length threshold is the recommended starting
    point per Section 4.4. The resulting mesh is open (has boundaries),
    faithfully representing gaps in survey coverage — a feature, not a bug,
    for hydrographic applications.
    """
    faces = jm.mesh_from_delaunay(enu_pts, edge_pct=edge_pct)
    return enu_pts, faces


def run_poisson(enu_pts: np.ndarray, depth: int = 9,
                trim_percentile: int = 5,
                normal_k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 3 — Poisson Surface Reconstruction (Section 5).

    Key parameters per Section 5.4:
    - octree depth 9–10 for 1–2 m point spacing
    - normal estimation via PCA on k=30 nearest neighbours
    - density-based trimming at 5th percentile to handle the watertight problem

    Best suited for enclosed underwater structures where the watertight
    assumption is physically valid (Section 5.2 & 5.3).
    """
    return jm.mesh_from_poisson(
        enu_pts,
        depth=depth,
        trim_percentile=trim_percentile,
        normal_k=normal_k,
    )


def run_bpa(enu_pts: np.ndarray,
            normal_k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 4 — Ball Pivoting Algorithm (Section 6).

    Multi-scale radii [ρ, 2ρ, 4ρ] per Section 6.3 — first meshes dense
    near-nadir regions at fine scale then fills swath edges at coarser scales.
    Radius ρ is set to the average nearest-neighbour distance (Section 6.2).
    Produces an open mesh; holes represent insufficient point density.
    """
    return jm.mesh_from_bpa(enu_pts, normal_k=normal_k)


# ---------------------------------------------------------------------------
# Per-lake benchmark
# ---------------------------------------------------------------------------

def benchmark_lake(
    lake_key: str,
    config:   dict,
    args,
    out_dir:  Path,
) -> List[AlgoResult]:

    npy_path = Path(config['npy'])
    label    = config['label']
    lake_dir = out_dir / lake_key
    lake_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Lake: {label}  ->  {lake_dir}")
    print(f"{'='*65}")

    enu, lat_ref, lon_ref = load_enu(npy_path)
    print(
        f"  Points: {len(enu):,}  "
        f"E {enu[:,0].min():.0f}..{enu[:,0].max():.0f} m  "
        f"N {enu[:,1].min():.0f}..{enu[:,1].max():.0f} m  "
        f"Z {enu[:,2].min():.2f}..{enu[:,2].max():.2f} m"
    )

    # Build algorithm list (strictly the four from the literature review)
    algorithms = [
        ('grid',     lambda: run_grid(enu, step=args.grid_step)),
        ('delaunay', lambda: run_delaunay(enu, edge_pct=args.edge_pct)),
    ]

    if HAS_OPEN3D:
        # Optionally subsample for Open3D algorithms to keep runtimes reasonable
        if args.max_pts > 0 and len(enu) > args.max_pts:
            rng     = np.random.default_rng(42)
            idx_sub = rng.choice(len(enu), args.max_pts, replace=False)
            enu_sub = enu[idx_sub]
            print(f"  [open3d] Subsampled to {len(enu_sub):,} pts (--max-pts {args.max_pts})")
        else:
            enu_sub = enu

        algorithms += [
            ('poisson', lambda: run_poisson(
                enu_sub,
                depth           = args.poisson_depth,
                trim_percentile = args.poisson_trim,
                normal_k        = args.normal_k,
            )),
            ('bpa', lambda: run_bpa(enu_sub, normal_k=args.normal_k)),
        ]
    else:
        print("  open3d not available — skipping poisson and bpa")

    results: List[AlgoResult] = []

    for algo_name, algo_fn in algorithms:
        print(f"\n  [{algo_name}] running ...", end='  ', flush=True)
        r  = AlgoResult(name=algo_name)
        t0 = time.perf_counter()

        try:
            vertices, faces = algo_fn()
            r.elapsed_s = time.perf_counter() - t0
            gc.collect()

            r.vertices  = len(vertices)
            r.triangles = len(faces)

            if len(faces) > 0:
                r.triangle_quality = triangle_quality_score(vertices, faces)
                r.coverage_pct, r.is_watertight = \
                    coverage_and_watertight(vertices, faces, enu)
                r.gap_fraction  = compute_gap_fraction(vertices, faces)
                r.median_edge_m = compute_median_edge_length(vertices, faces)

                obj_path = lake_dir / f'{algo_name}_mesh.obj'
                jm.export_obj(vertices, faces, obj_path, lat_ref, lon_ref, z_scale=1.0)

            print(
                f"done  {r.elapsed_s:.1f}s | "
                f"{r.vertices:,}v {r.triangles:,}f | "
                f"quality={r.triangle_quality:.3f} | "
                f"gap={r.gap_fraction*100:.1f}% | "
                f"edge={r.median_edge_m:.2f}m"
            )

        except Exception as exc:
            r.error     = str(exc)
            r.elapsed_s = time.perf_counter() - t0
            print(f"FAILED: {exc}")

        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: List[AlgoResult], label: str) -> None:
    W = 92
    print(f"\n  Results — {label}")
    print("  " + "=" * W)
    hdr = (
        f"  {'Algorithm':<12} {'Verts':>8} {'Tris':>9} {'Time(s)':>8} "
        f"{'TriQual':>8} {'Cov%':>6} {'Gap%':>6} {'Edge(m)':>8} {'Wtight':>7}"
    )
    print(hdr)
    print("  " + "-" * W)
    for r in results:
        if r.error:
            print(f"  {r.name:<12}  ERROR: {r.error[:75]}")
            continue
        print(
            f"  {r.name:<12} {r.vertices:>8,} {r.triangles:>9,} "
            f"{r.elapsed_s:>8.1f} {r.triangle_quality:>8.4f} "
            f"{r.coverage_pct:>6.1f} {r.gap_fraction*100:>6.1f} "
            f"{r.median_edge_m:>8.2f} "
            f"{'yes' if r.is_watertight else 'no':>7}"
        )
    print("  " + "=" * W)


def save_csv(all_results: Dict[str, List[AlgoResult]], out_dir: Path) -> None:
    out_path   = out_dir / 'benchmark_results.csv'
    fieldnames = [
        'lake', 'algorithm', 'vertices', 'triangles', 'time_s',
        'tri_quality', 'coverage_pct', 'gap_pct', 'median_edge_m',
        'watertight', 'cv_rmse', 'combined_score', 'error',
    ]
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for lake, results in all_results.items():
            for r in results:
                writer.writerow({
                    'lake':          lake,
                    'algorithm':     r.name,
                    'vertices':      r.vertices,
                    'triangles':     r.triangles,
                    'time_s':        round(r.elapsed_s, 2),
                    'tri_quality':   round(r.triangle_quality, 5),
                    'coverage_pct':  round(r.coverage_pct, 2),
                    'gap_pct':       round(r.gap_fraction * 100, 2),
                    'median_edge_m': round(r.median_edge_m, 3),
                    'watertight':    r.is_watertight,
                    'cv_rmse':       '' if np.isnan(r.cv_rmse)        else round(r.cv_rmse, 4),
                    'combined_score':'' if np.isnan(r.combined_score) else round(r.combined_score, 4),
                    'error':         r.error,
                })
    print(f"  CSV  → {out_path}")


def _normalise(vals: list, higher_better: bool) -> list:
    arr = np.array([v if np.isfinite(v) else 0.0 for v in vals], dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return [0.5] * len(vals)
    normed = (arr - lo) / (hi - lo)
    return (normed if higher_better else 1.0 - normed).tolist()


def _radar_panel(ax, categories, algo_names, algo_values, colors):
    """Draw a radar / spider chart on a polar Axes (larger polygon = better)."""
    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=6)
    ax.grid(color='grey', linewidth=0.5, linestyle='--', alpha=0.6)

    for i, (name, vals) in enumerate(zip(algo_names, algo_values)):
        v = vals + vals[:1]
        ax.plot(angles, v, color=colors[i], linewidth=1.8, label=name)
        ax.fill(angles, v, color=colors[i], alpha=0.12)

    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=8)


def plot_results(all_results: Dict[str, List[AlgoResult]], out_dir: Path,
                 has_crossval: bool = False) -> None:
    """
    Save a radar chart comparing all four algorithms per lake.
    Axes (all normalised to [0,1], larger = better):
      Speed, Triangle Quality, Coverage, Gap Fill, CV Accuracy (optional).
    """
    lakes = list(all_results.keys())
    algos = [r.name for r in next(iter(all_results.values()))]

    cats = ['Speed', 'Tri Quality', 'Coverage', 'Gap Fill']
    if has_crossval:
        cats.append('CV Accuracy')

    cmap   = plt.cm.tab10
    colors = [cmap(i / max(len(algos) - 1, 1)) for i in range(len(algos))]

    n_cols = len(lakes)
    fig    = plt.figure(figsize=(5.5 * n_cols, 5.5))

    for col, lake in enumerate(lakes):
        ax           = fig.add_subplot(1, n_cols, col + 1, polar=True)
        results_lake = all_results[lake]
        time_vals    = [r.elapsed_s         for r in results_lake]
        qual_vals    = [r.triangle_quality   for r in results_lake]
        cov_vals     = [r.coverage_pct / 100 for r in results_lake]
        gap_vals     = [1.0 - r.gap_fraction for r in results_lake]

        speed_norm = _normalise([1.0 / max(t, 1e-3) for t in time_vals], True)
        qual_norm  = _normalise(qual_vals, True)
        cov_norm   = _normalise(cov_vals,  True)
        gap_norm   = _normalise(gap_vals,  True)

        algo_values = [
            [speed_norm[i], qual_norm[i], cov_norm[i], gap_norm[i]]
            for i in range(len(algos))
        ]
        if has_crossval:
            cv_vals = [r.cv_rmse if np.isfinite(r.cv_rmse) else 99 for r in results_lake]
            cv_norm = _normalise(cv_vals, False)
            for i in range(len(algos)):
                algo_values[i].append(cv_norm[i])

        _radar_panel(ax, cats, algos, algo_values, colors)
        ax.set_title(LAKE_CONFIG[lake]['label'], fontsize=10, pad=14)

    fig.suptitle(
        'Surface Reconstruction Benchmark — Radar Chart\n'
        '(grid · delaunay · poisson · bpa  |  larger polygon = better)',
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    out_path = out_dir / 'benchmark_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {out_path}")


def print_summary(all_results: Dict[str, List[AlgoResult]]) -> None:
    """Cross-lake average ranking with literature-informed interpretation."""
    agg: Dict[str, Dict[str, list]] = {}
    for results in all_results.values():
        for r in results:
            if r.error or r.triangles == 0:
                continue
            d = agg.setdefault(r.name, {
                'time': [], 'quality': [], 'coverage': [], 'gap': []
            })
            d['time'].append(r.elapsed_s)
            d['quality'].append(r.triangle_quality)
            d['coverage'].append(r.coverage_pct)
            d['gap'].append(r.gap_fraction * 100)

    print("\n  --- Cross-lake averages ---")
    print(f"  {'Algorithm':<14} {'Avg time(s)':>12} {'Avg quality':>12} "
          f"{'Avg cov%':>10} {'Avg gap%':>10}")
    print("  " + "-" * 62)
    for algo in sorted(agg):
        d   = agg[algo]
        t_s = f"{np.mean(d['time']):.1f}"     if d['time']     else 'n/a'
        q_s = f"{np.mean(d['quality']):.4f}"  if d['quality']  else 'n/a'
        c_s = f"{np.mean(d['coverage']):.1f}" if d['coverage'] else 'n/a'
        g_s = f"{np.mean(d['gap']):.1f}"      if d['gap']      else 'n/a'
        print(f"  {algo:<14} {t_s:>12} {q_s:>12} {c_s:>10} {g_s:>10}")

    if agg:
        best_quality = max(agg, key=lambda a: np.mean(agg[a]['quality'])  if agg[a]['quality']  else 0)
        best_cov     = max(agg, key=lambda a: np.mean(agg[a]['coverage']) if agg[a]['coverage'] else 0)
        least_gap    = min(agg, key=lambda a: np.mean(agg[a]['gap'])      if agg[a]['gap']      else 100)
        fastest      = min(agg, key=lambda a: np.mean(agg[a]['time'])     if agg[a]['time']     else 1e9)
        print(f"\n  Best triangle quality : {best_quality}")
        print(f"  Best coverage         : {best_cov}")
        print(f"  Least gap             : {least_gap}")
        print(f"  Fastest               : {fastest}")

    print("""
  --- Literature Review Interpretation (Ayat 2026) ---
  grid     : Best for rapid gridding of dense, clean data → regular-grid DEMs.
             Fills inter-swath gaps via interpolation; NaN beyond convex hull.
  delaunay : Best overall for general bathymetry (Section 8.2).
             Preserves measured depths; open mesh honestly represents gaps.
             Recommended for legal charting and heterogeneous coverage.
  poisson  : Best for enclosed structures (cisterns, ship hulls) where the
             watertight assumption is physically valid (Section 5.2-5.3).
             Density trimming (--poisson-trim) mitigates spurious caps.
  bpa      : Best for high-resolution feature-dense surveys (Section 8.2).
             Holes indicate coverage gaps — useful for real-time QC (Section 6.3).
             Adaptive radii via Saffi et al. (2024) not yet in standard software.

  NOTE: Run jsf_crossval.py and jsf_overlap_check.py for accuracy estimates,
  then use generate_report_summary.py to combine all results.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_crossval_csv(csv_path: Path) -> Dict[str, tuple]:
    if not csv_path.exists():
        return {}
    results = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            algo = row.get('algorithm', '').strip()
            try:
                cv   = float(row['cv_rmse_m'])      if row.get('cv_rmse_m')      else float('nan')
                comb = float(row['combined_score'])  if row.get('combined_score')  else float('nan')
            except ValueError:
                cv, comb = float('nan'), float('nan')
            results[algo] = (cv, comb)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Benchmark the four literature-reviewed surface reconstruction algorithms '
                    'on lake bathymetric data.'
    )
    parser.add_argument(
        '--lakes', nargs='+',
        default=list(LAKE_CONFIG.keys()),
        choices=list(LAKE_CONFIG.keys()),
        help='Lakes to process (default: all)',
    )
    parser.add_argument('--output-dir',    default='benchmark_out', metavar='DIR')
    parser.add_argument('--grid-step',     type=float, default=1.0,  metavar='S',
                        help='Grid interpolation cell size in metres (default 1.0; '
                             'Section 3.3 recommends 0.5–2.0 m)')
    parser.add_argument('--edge-pct',      type=float, default=95,   metavar='P',
                        help='Delaunay edge-filter percentile (default 95; '
                             'Section 4.4 recommends 90–95)')
    parser.add_argument('--poisson-depth', type=int,   default=9,    metavar='D',
                        help='Poisson octree depth (default 9; '
                             'Section 5.4 recommends 9-10 for 1-2 m spacing)')
    parser.add_argument('--poisson-trim',  type=int,   default=5,    metavar='T',
                        help='Poisson density trim percentile (default 5; Section 5.2)')
    parser.add_argument('--normal-k',      type=int,   default=30,   metavar='K',
                        help='kNN for normal estimation, poisson/bpa (default 30; '
                             'Section 5.4 recommends 20-50)')
    parser.add_argument('--max-pts',       type=int,   default=0,    metavar='N',
                        help='Subsample to N pts before Open3D algorithms (0 = no limit)')
    parser.add_argument('--crossval',      action='store_true',
                        help='Load crossval_results.csv and add CV-RMSE to table/chart')
    parser.add_argument('--eval-dir',      default='evaluation_out', metavar='DIR')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crossval_data: Dict[str, tuple] = {}
    if args.crossval:
        csv_path     = Path(args.eval_dir) / 'crossval_results.csv'
        crossval_data = _load_crossval_csv(csv_path)
        if crossval_data:
            print(f"  Loaded CV scores for {len(crossval_data)} algorithms from {csv_path}")
        else:
            print(f"  WARNING: --crossval set but {csv_path} not found or empty.")
            print(f"           Run:  python jsf_crossval.py --lake <name>  first.")

    all_results: Dict[str, List[AlgoResult]] = {}

    for lake_key in args.lakes:
        cfg      = LAKE_CONFIG[lake_key]
        npy_path = Path(cfg['npy'])
        if not npy_path.exists():
            print(f"\nWARNING: {npy_path} not found — skipping {cfg['label']}")
            continue
        results = benchmark_lake(lake_key, cfg, args, out_dir)

        for r in results:
            if r.name in crossval_data:
                r.cv_rmse, r.combined_score = crossval_data[r.name]

        all_results[lake_key] = results
        print_table(results, cfg['label'])

    if not all_results:
        print("No lakes processed. Exiting.")
        return

    has_cv = bool(crossval_data)
    print(f"\n{'='*65}")
    plot_results(all_results, out_dir, has_crossval=has_cv)
    save_csv(all_results, out_dir)
    print_summary(all_results)
    print(f"\n  All outputs in: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
