"""
jsf_merge.py — Multi-file point cloud merge, surface reconstruction, and mesh export.

Implements the four surface reconstruction algorithms evaluated in the literature
review (Ayat, 2026):

  1. grid      — SciPy griddata linear interpolation to regular grid (Section 3)
  2. delaunay  — 2-D Delaunay triangulation + percentile edge-length filter (Section 4)
  3. poisson   — Poisson Surface Reconstruction via Open3D (Section 5)
  4. bpa       — Ball Pivoting Algorithm via Open3D (Section 6)

Usage:
    python jsf_merge.py <npy_dir> [options]

Options:
    --grid-size M    Cell size for point-cloud cleaning (default: 0.2 m)
    --name NAME      Output file stem (default: merged)
    --max-depth D    Discard points deeper than D metres (default: no limit)
    --z-scale Z      Vertical exaggeration for export (default: 1.0)
    --step S         Grid interpolation spacing in metres (default: 1.0)
    --output-dir D   Where to write outputs (default: same as npy_dir)
    --method M       'grid' | 'delaunay' | 'poisson' | 'bpa'  (default: grid)
    --edge-pct P     [delaunay] Edge percentile filter (default: 95)
    --poisson-depth D [poisson] Octree depth (default: 9; use 10-11 for sub-metre surveys)
    --poisson-trim T  [poisson] Density trim percentile (default: 5)
    --normal-k K     [poisson/bpa] Neighbours for normal estimation (default: 30)
    --no-ply / --no-obj

Coordinate system in exported files:
    X = East  (metres from survey centroid)
    Y = North (metres from survey centroid)
    Z = Up    (-depth * z_scale)
    Comment in OBJ/PLY header records the reference lat/lon.
"""

import os
import sys
import gc
import argparse
from math import cos, radians
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, cKDTree

import jsf_parser as jsp


# ---------------------------------------------------------------------------
# Coordinate projection
# ---------------------------------------------------------------------------

def to_enu(pts, lat_ref=None, lon_ref=None, z_scale=1.0):
    """
    Convert POINT_DTYPE structured array → (N,3) local ENU float64 array.
    Returns (enu, lat_ref, lon_ref).
    """
    if lat_ref is None:
        lat_ref = float(np.mean(pts['lat']))
    if lon_ref is None:
        lon_ref = float(np.mean(pts['lon']))

    east  = (pts['lon'] - lon_ref) * (111320.0 * cos(radians(lat_ref)))
    north = (pts['lat'] - lat_ref) * 111320.0
    up    = -pts['depth'] * z_scale

    return np.column_stack([east, north, up]).astype(np.float64), lat_ref, lon_ref


# ---------------------------------------------------------------------------
# Statistical outlier removal
# ---------------------------------------------------------------------------

def sor_filter(enu_pts, k=20, sigma=2.0, min_floor_m=0.5):
    """
    Remove points whose Z (depth) deviates more than sigma × MAD from their
    k nearest neighbours' median Z.

    min_floor_m: minimum MAD floor in metres so flat areas don't get
                 over-aggressively filtered (allows ±sigma*floor deviation).

    Returns filtered (N,3) array and a boolean keep-mask.
    """
    tree = cKDTree(enu_pts[:, :2])
    _, idx = tree.query(enu_pts[:, :2], k=min(k + 1, len(enu_pts)))

    neighbour_z  = enu_pts[:, 2][idx[:, 1:]]
    local_median = np.median(neighbour_z, axis=1)
    local_mad    = np.median(np.abs(neighbour_z - local_median[:, None]), axis=1)

    threshold = sigma * np.maximum(1.4826 * local_mad, min_floor_m)
    keep = np.abs(enu_pts[:, 2] - local_median) <= threshold

    return enu_pts[keep], keep


# ---------------------------------------------------------------------------
# Algorithm 1 — SciPy griddata interpolation  (Literature Review, Section 3)
# ---------------------------------------------------------------------------

def mesh_from_grid(enu_pts, step=1.0):
    """
    SciPy griddata linear interpolation to a regular grid, then triangulated.

    Per Section 3 of the literature review:
    - Linear method preserves measured values exactly and avoids oscillation.
    - Grid resolution should be finer than average point spacing (0.5–2.0 m
      for dense multibeam surveys).
    - fill_value=np.nan explicitly marks uninterpolated (gap) regions.

    The resulting mesh is an open 2.5D surface — a height field — which is
    the natural representation for bathymetric data.

    Args:
        enu_pts: (N, 3) float64 ENU array.
        step:    Grid spacing in metres (default 1.0 m).

    Returns:
        vertices: (V, 3) float64 grid vertex array.
        faces:    (F, 3) int32 triangle index array.
    """
    x, y, z = enu_pts[:, 0], enu_pts[:, 1], enu_pts[:, 2]

    xi = np.arange(x.min(), x.max() + step, step)
    yi = np.arange(y.min(), y.max() + step, step)
    ny, nx = len(yi), len(xi)

    XI, YI = np.meshgrid(xi, yi, indexing='xy')
    ZI = griddata(
        np.column_stack([x, y]), z, (XI, YI),
        method='linear',
        fill_value=np.nan,   # explicitly mark uninterpolated regions (Section 3.3)
    )

    valid = ~np.isnan(ZI)
    vertex_idx = np.full((ny, nx), -1, dtype=np.int32)
    n_valid = int(valid.sum())
    vertex_idx[valid] = np.arange(n_valid, dtype=np.int32)

    vertices = np.column_stack([XI[valid], YI[valid], ZI[valid]])

    rows, cols = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing='ij')
    rows = rows.ravel()
    cols = cols.ravel()

    i00 = vertex_idx[rows,     cols    ]
    i01 = vertex_idx[rows,     cols + 1]
    i10 = vertex_idx[rows + 1, cols    ]
    i11 = vertex_idx[rows + 1, cols + 1]

    m1 = (i00 >= 0) & (i01 >= 0) & (i10 >= 0)
    m2 = (i01 >= 0) & (i11 >= 0) & (i10 >= 0)

    tri1  = np.column_stack([i00[m1], i01[m1], i10[m1]])
    tri2  = np.column_stack([i01[m2], i11[m2], i10[m2]])
    faces = np.vstack([tri1, tri2]).astype(np.int32)

    return vertices.astype(np.float64), faces


# ---------------------------------------------------------------------------
# Algorithm 2 — 2D Delaunay + percentile edge filtering  (Section 4)
# ---------------------------------------------------------------------------

def mesh_from_delaunay(enu_pts, edge_pct=95):
    """
    2-D Delaunay triangulation with percentile-based edge-length filtering.

    Per Section 4 of the literature review:
    - Delaunay triangulation is the natural representation for 2.5D surfaces.
    - Maximises the minimum angle; avoids degenerate slivers.
    - Edge-length filtering (chi-shape framework, Duckham et al. 2008) removes
      long triangles spanning data gaps or survey boundaries.
    - The 90th–95th percentile is the recommended starting threshold (Section 4.4).
    - Resulting mesh is open (has boundaries), honestly representing data gaps.

    Args:
        enu_pts:  (N, 3) float64 ENU array.
        edge_pct: Percentile threshold for edge-length filter (default 95).

    Returns:
        faces: (F, 3) int32 triangle index array.
               Vertices are the original enu_pts (points are interpolated exactly).
    """
    xy  = enu_pts[:, :2]
    tri = Delaunay(xy)

    p   = xy[tri.simplices]
    e01 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    e12 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    e20 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    max_edge = np.maximum(e01, np.maximum(e12, e20))

    threshold = float(np.percentile(max_edge, edge_pct))
    faces     = tri.simplices[max_edge <= threshold].copy().astype(np.int32)
    print(f'  Edge filter ({edge_pct}th pct): {threshold:.2f} m threshold, '
          f'{len(faces):,} / {len(tri.simplices):,} triangles kept')
    return faces


# ---------------------------------------------------------------------------
# Algorithm 3 — Poisson Surface Reconstruction  (Section 5)
# ---------------------------------------------------------------------------

def mesh_from_poisson(enu_pts, depth=9, trim_percentile=5, normal_k=30):
    """
    Poisson Surface Reconstruction via Open3D.

    Per Section 5 of the literature review:
    - Requires oriented point normals; normals must point upward (toward the
      water column) for bathymetric data (Section 5.4).
    - Normal estimation via PCA on k nearest neighbours (typically 20–50).
    - Octree depth 9–10 is appropriate for 1–2 m point spacing (Section 5.4).
    - Density-based trimming removes spurious closing geometry at survey
      boundaries — a fundamental consequence of the algorithm's watertight
      formulation (Section 5.2).

    Args:
        enu_pts:         (N, 3) float64 ENU array.
        depth:           Octree depth (default 9; use 10–11 for sub-metre data).
        trim_percentile: Density percentile below which vertices are removed (default 5).
        normal_k:        Neighbours for PCA normal estimation (default 30).

    Returns:
        vertices: (V, 3) float64 mesh vertex array.
        faces:    (F, 3) int32 triangle index array.

    Raises:
        ImportError if open3d is not installed.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            'open3d is required for Poisson reconstruction. '
            'Install with: pip install open3d'
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(enu_pts)

    # Normal estimation: PCA on k nearest neighbours (Section 5.4)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(normal_k)
    )
    # Consistent upward orientation — essential for open 2.5D surfaces
    pcd.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=False
    )
    densities = np.asarray(densities)

    # Density-based trimming: removes spurious caps at survey boundaries (Section 5.2)
    mesh.remove_vertices_by_mask(densities < np.percentile(densities, trim_percentile))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()

    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


# ---------------------------------------------------------------------------
# Algorithm 4 — Ball Pivoting Algorithm  (Section 6)
# ---------------------------------------------------------------------------

def mesh_from_bpa(enu_pts, normal_k=30):
    """
    Ball Pivoting Algorithm via Open3D with multi-scale radii.

    Per Section 6 of the literature review:
    - Radius must be 1–2× the average nearest-neighbour distance (Section 6.2).
    - Multi-scale pivoting [ρ, 2ρ, 4ρ] first meshes dense regions at fine scale
      then fills larger gaps at coarser scales (Section 6.3).
    - Normal estimation improves results on steep terrain (Section 6.4).
    - Normals oriented upward (toward water surface) for consistent meshing.
    - Produces an open mesh; holes honestly represent insufficient point density.

    Args:
        enu_pts:  (N, 3) float64 ENU array.
        normal_k: Neighbours for PCA normal estimation (default 30).

    Returns:
        vertices: (V, 3) float64 mesh vertex array.
        faces:    (F, 3) int32 triangle index array.

    Raises:
        ImportError if open3d is not installed.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            'open3d is required for BPA reconstruction. '
            'Install with: pip install open3d'
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(enu_pts)

    # Normal estimation (optional for BPA but improves steep-terrain results)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(normal_k)
    )
    pcd.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])

    # Compute characteristic point spacing
    dists    = pcd.compute_nearest_neighbor_distance()
    avg_dist = float(np.mean(dists))

    # Multi-scale radii: [ρ, 2ρ, 4ρ] per Section 6.3
    radii = o3d.utility.DoubleVector([avg_dist, avg_dist * 2.0, avg_dist * 4.0])

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


# ---------------------------------------------------------------------------
# OBJ / PLY export
# ---------------------------------------------------------------------------

def _header_comment(lat_ref, lon_ref, z_scale):
    return (f'local ENU metres; origin lat={lat_ref:.8f} '
            f'lon={lon_ref:.8f}; z_scale={z_scale}')


def export_obj(vertices, faces, out_path, lat_ref, lon_ref, z_scale):
    print(f'  Writing OBJ: {out_path.name}  '
          f'({len(vertices):,} verts, {len(faces):,} faces) ...', end=' ', flush=True)
    BLOCK = 100_000
    with open(out_path, 'w') as f:
        f.write('# Generated by jsf_merge.py\n')
        f.write(f'# {_header_comment(lat_ref, lon_ref, z_scale)}\n')
        f.write('# X=East_m Y=North_m Z=Up_m(-depth*z_scale)\n')
        for s in range(0, len(vertices), BLOCK):
            e = min(s + BLOCK, len(vertices))
            block = vertices[s:e]
            lines = [f'v {r[0]:.3f} {r[1]:.3f} {r[2]:.3f}' for r in block]
            f.write('\n'.join(lines) + '\n')
        for s in range(0, len(faces), BLOCK):
            e = min(s + BLOCK, len(faces))
            block = faces[s:e] + 1
            lines = [f'f {r[0]} {r[1]} {r[2]}' for r in block]
            f.write('\n'.join(lines) + '\n')
    print('done.')


def export_ply(vertices, faces, out_path, lat_ref, lon_ref, z_scale):
    print(f'  Writing PLY: {out_path.name}  '
          f'({len(vertices):,} verts, {len(faces):,} faces) ...', end=' ', flush=True)
    n_vert, n_face = len(vertices), len(faces)
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'comment {_header_comment(lat_ref, lon_ref, z_scale)}\n'
        f'element vertex {n_vert}\n'
        'property float64 x\nproperty float64 y\nproperty float64 z\n'
        f'element face {n_face}\n'
        'property list uint8 int32 vertex_indices\nend_header\n'
    )
    fsd = np.dtype([('count', np.uint8), ('v', np.int32, 3)])
    fs  = np.empty(n_face, dtype=fsd)
    fs['count'] = 3
    fs['v']     = faces.astype(np.int32)
    with open(out_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertices.astype(np.float64).tobytes())
        f.write(fs.tobytes())
    print('done.')


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------

def do_merge(npy_files, output_dir, name='merged', grid_size=0.2,
             min_depth=None, max_depth=None, z_scale=1.0,
             sor_k=20, sor_sigma=None,
             method='grid', step=1.0,
             edge_pct=95,
             poisson_depth=9, poisson_trim=5, normal_k=30,
             write_obj=True, write_ply=True):
    """
    Merge .npy chunk files into a single cleaned 3-D mesh using one of four
    surface reconstruction algorithms from the literature review.

    Args:
        npy_files:     List of .npy file paths (per-chunk point clouds).
        output_dir:    Output directory path.
        name:          File stem for output files.
        grid_size:     Grid-median cleaning cell size in metres.
        min_depth:     Discard points shallower than this (metres).
        max_depth:     Discard points deeper than this (metres).
        z_scale:       Vertical exaggeration factor.
        sor_k:         Neighbours for statistical outlier removal.
        sor_sigma:     SOR threshold in MAD units (None = disabled).
        method:        'grid' | 'delaunay' | 'poisson' | 'bpa'.
        step:          Grid spacing in metres (grid algorithm only).
        edge_pct:      Edge-length percentile filter (delaunay only).
        poisson_depth: Octree depth for Poisson (default 9).
        poisson_trim:  Density trim percentile for Poisson (default 5).
        normal_k:      kNN for normal estimation (poisson/bpa).
        write_obj:     Export OBJ mesh file.
        write_ply:     Export PLY mesh file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_files  = [Path(p) for p in npy_files]

    # --- Load ---
    total_bytes = sum(p.stat().st_size for p in npy_files)
    print(f'Loading {len(npy_files)} chunk file(s) ({total_bytes/1e6:.1f} MB) ...')
    all_pts = np.concatenate([np.load(p) for p in npy_files])
    print(f'  Combined: {len(all_pts):,} points')

    # --- Depth filters ---
    if min_depth is not None:
        before  = len(all_pts)
        all_pts = all_pts[all_pts['depth'] >= min_depth]
        print(f'  Min-depth <{min_depth} m removed: {before - len(all_pts):,}, '
              f'{len(all_pts):,} remain')
    if max_depth is not None:
        before  = len(all_pts)
        all_pts = all_pts[all_pts['depth'] <= max_depth]
        print(f'  Max-depth >{max_depth} m removed: {before - len(all_pts):,}, '
              f'{len(all_pts):,} remain')

    # --- Grid-median clean ---
    print(f'Applying {grid_size} m grid-median filter ...')
    cleaned = jsp.grid_median_clean(all_pts, grid_size)
    del all_pts; gc.collect()
    print(f'  After cleaning: {len(cleaned):,} points')

    if len(cleaned) < 4:
        print('Too few points. Aborting.')
        return

    np.save(output_dir / f'{name}_merged.npy', cleaned)

    # --- Project to ENU metres ---
    enu, lat_ref, lon_ref = to_enu(cleaned, z_scale=z_scale)
    print(f'  ENU origin: lat={lat_ref:.6f} lon={lon_ref:.6f}')
    print(f'  Extent: E {enu[:,0].min():.0f}..{enu[:,0].max():.0f} m  '
          f'N {enu[:,1].min():.0f}..{enu[:,1].max():.0f} m  '
          f'Z {enu[:,2].min():.2f}..{enu[:,2].max():.2f} m')

    # --- Statistical outlier removal ---
    if sor_sigma is not None:
        before = len(enu)
        print(f'Statistical outlier removal (k={sor_k}, sigma={sor_sigma}) ...')
        enu, _ = sor_filter(enu, k=sor_k, sigma=sor_sigma)
        print(f'  Removed {before - len(enu):,} outliers '
              f'({100*(before-len(enu))/before:.1f}%), '
              f'{len(enu):,} points remain')

    # --- Surface reconstruction ---
    if method == 'grid':
        print(f'[grid] SciPy griddata interpolation at {step} m spacing ...')
        vertices, faces = mesh_from_grid(enu, step=step)
        print(f'  Grid vertices: {len(vertices):,}, triangles: {len(faces):,}')

    elif method == 'delaunay':
        print(f'[delaunay] 2-D Delaunay + {edge_pct}th-pct edge filter '
              f'on {len(enu):,} points ...')
        faces    = mesh_from_delaunay(enu, edge_pct=edge_pct)
        vertices = enu
        print(f'  Triangles after filtering: {len(faces):,}')

    elif method == 'poisson':
        print(f'[poisson] Poisson Surface Reconstruction '
              f'(depth={poisson_depth}, trim={poisson_trim}%, '
              f'normal_k={normal_k}) ...')
        vertices, faces = mesh_from_poisson(
            enu, depth=poisson_depth,
            trim_percentile=poisson_trim, normal_k=normal_k,
        )
        print(f'  Poisson vertices: {len(vertices):,}, triangles: {len(faces):,}')

    elif method == 'bpa':
        print(f'[bpa] Ball Pivoting Algorithm '
              f'(multi-scale radii [ρ,2ρ,4ρ], normal_k={normal_k}) ...')
        vertices, faces = mesh_from_bpa(enu, normal_k=normal_k)
        print(f'  BPA vertices: {len(vertices):,}, triangles: {len(faces):,}')

    else:
        raise ValueError(f'Unknown method: {method!r}. '
                         f'Choose grid | delaunay | poisson | bpa')

    if len(faces) == 0:
        print('No triangles produced. Aborting.')
        return

    # --- Export ---
    if write_obj:
        export_obj(vertices, faces, output_dir / f'{name}_{method}_mesh.obj',
                   lat_ref, lon_ref, z_scale)
    if write_ply:
        export_ply(vertices, faces, output_dir / f'{name}_{method}_mesh.ply',
                   lat_ref, lon_ref, z_scale)

    print(f'\nDone.  Method: {method}  '
          f'Vertices: {len(vertices):,}  Triangles: {len(faces):,}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Merge JSF chunk .npy files into a VR-ready 3D mesh '
                    'using one of four literature-reviewed reconstruction algorithms.'
    )
    p.add_argument('npy_source', help='Directory containing _chunk_NNNN.npy files')
    p.add_argument('--grid-size',      type=float, default=0.2,       metavar='M')
    p.add_argument('--name',           default='merged',              metavar='NAME')
    p.add_argument('--min-depth',      type=float, default=None,      metavar='D')
    p.add_argument('--max-depth',      type=float, default=None,      metavar='D')
    p.add_argument('--z-scale',        type=float, default=1.0,       metavar='Z')
    p.add_argument('--step',           type=float, default=1.0,       metavar='S',
                   help='Grid spacing in metres for griddata method (default: 1.0)')
    p.add_argument('--method',         default='grid',
                   choices=['grid', 'delaunay', 'poisson', 'bpa'],
                   help='Reconstruction algorithm (default: grid)')
    p.add_argument('--sor-sigma',      type=float, default=None,      metavar='S')
    p.add_argument('--sor-k',          type=int,   default=20,        metavar='K')
    p.add_argument('--edge-pct',       type=float, default=95,        metavar='P',
                   help='[delaunay] Edge-length percentile filter (default: 95)')
    p.add_argument('--poisson-depth',  type=int,   default=9,         metavar='D',
                   help='[poisson] Octree depth (default: 9; use 10-11 for sub-metre data)')
    p.add_argument('--poisson-trim',   type=int,   default=5,         metavar='T',
                   help='[poisson] Density trim percentile (default: 5)')
    p.add_argument('--normal-k',       type=int,   default=30,        metavar='K',
                   help='[poisson/bpa] Neighbours for normal estimation (default: 30)')
    p.add_argument('--output-dir',     default=None,                  metavar='DIR')
    p.add_argument('--no-ply',  action='store_true')
    p.add_argument('--no-obj',  action='store_true')
    args = p.parse_args()

    source = Path(args.npy_source)
    if source.is_dir():
        npy_files = sorted(source.glob('*_chunk_*.npy'))
        if not npy_files:
            npy_files = [f for f in sorted(source.glob('*.npy'))
                         if '_merged' not in f.stem]
        output_dir = Path(args.output_dir) if args.output_dir else source
    elif source.is_file():
        npy_files  = [source]
        output_dir = Path(args.output_dir) if args.output_dir else source.parent
    else:
        print(f'Error: {source} not found.'); sys.exit(1)

    if not npy_files:
        print(f'No .npy files in {source}'); sys.exit(1)

    do_merge(
        npy_files=npy_files, output_dir=output_dir, name=args.name,
        grid_size=args.grid_size,
        min_depth=args.min_depth, max_depth=args.max_depth,
        z_scale=args.z_scale,
        sor_k=args.sor_k, sor_sigma=args.sor_sigma,
        method=args.method, step=args.step,
        edge_pct=args.edge_pct,
        poisson_depth=args.poisson_depth,
        poisson_trim=args.poisson_trim,
        normal_k=args.normal_k,
        write_obj=not args.no_obj, write_ply=not args.no_ply,
    )


if __name__ == '__main__':
    main()
