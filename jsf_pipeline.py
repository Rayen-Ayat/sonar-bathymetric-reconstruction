"""
jsf_pipeline.py — Chunked JSF bathymetry processor.

Usage:
    python jsf_pipeline.py <file_or_dir> [options]

    <file_or_dir>   Single .jsf file OR directory of .jsf files.
                    Continuation files (.001, .002, ...) are auto-chained.

Options:
    --chunk-size N       Pings per processing chunk (default: 10000)
    --grid-size M        Grid cell size in metres for cleaning (default: 0.2)
    --output-dir DIR     Where to write .npy chunk files (default: ./output)
    --no-merge           Skip the merge/triangulate step after all chunks
    --resume             Skip chunks whose .npy output files already exist
    --name NAME          Stem name for merged output files (default: merged)

    Reconstruction method (passed to jsf_merge.py):
    --method M           grid | delaunay | poisson | bpa  (default: grid)

    [grid options]
    --step S             Grid interpolation spacing in metres (default: 1.0)

    [delaunay options]
    --edge-pct P         Edge percentile for triangle filtering (default: 95)

    [poisson options]
    --poisson-depth D    Octree depth (default: 9; use 10-11 for sub-metre data)
    --poisson-trim T     Density trim percentile (default: 5)

    [poisson / bpa options]
    --normal-k K         Neighbours for normal estimation (default: 30)

Produces:
    <output-dir>/<stem>_chunk_NNNN.npy   per-chunk cleaned point clouds
    <output-dir>/<name>_merged.npy       merged + cleaned point cloud
    <output-dir>/<name>_<method>_mesh.obj  ASCII OBJ mesh
    <output-dir>/<name>_<method>_mesh.ply  Binary PLY mesh
"""

import os
import sys
import gc
import argparse
from pathlib import Path

import numpy as np

import jsf_parser as jsp


# ---------------------------------------------------------------------------
# Chunk flush
# ---------------------------------------------------------------------------

def flush_chunk(ping_buffer, chunk_index, output_dir, stem, grid_size):
    """
    Georeference all pings in ping_buffer, grid-clean, and save as .npy.
    Returns the path of the saved file, or None if no valid points.
    """
    parts = []
    for (lat, lon, heading, beams, is_starboard) in ping_buffer:
        pts = jsp.georef_beams(beams, lat, lon, heading, is_starboard)
        if len(pts):
            parts.append(pts)

    if not parts:
        return None

    chunk_pts = np.concatenate(parts)
    del parts

    cleaned   = jsp.grid_median_clean(chunk_pts, grid_size)
    del chunk_pts
    gc.collect()

    out_path = output_dir / f'{stem}_chunk_{chunk_index:04d}.npy'
    np.save(out_path, cleaned)

    n_out = len(cleaned)
    print(f'  Chunk {chunk_index:04d}: {n_out:,} pts saved -> {out_path.name}')

    del cleaned
    gc.collect()
    return out_path


# ---------------------------------------------------------------------------
# Process one logical file
# ---------------------------------------------------------------------------

def process_logical_file(logical_name, file_paths, output_dir, chunk_size,
                          grid_size, resume):
    """
    Stream-process a logical file (possibly spanning multiple physical files).
    Returns a list of paths to the .npy chunk files produced.
    """
    stem      = logical_name
    nav       = jsp.NavState()
    ping_buf  = []
    chunk_idx = 0
    npy_files = []

    if resume:
        existing = sorted(output_dir.glob(f'{stem}_chunk_*.npy'))
        if existing:
            highest   = max(int(p.stem.rsplit('_', 1)[1]) for p in existing)
            chunk_idx = highest + 1
            npy_files = list(existing)
            print(f'  Resume: found {len(existing)} existing chunk(s), '
                  f'resuming from chunk {chunk_idx:04d}')
            pings_to_skip = chunk_idx * chunk_size
        else:
            pings_to_skip = 0
    else:
        pings_to_skip = 0

    pings_seen = 0

    for msg in jsp.iter_logical_file(file_paths):
        if msg.msg_type == 2002:
            fix = jsp.parse_nav_2002(msg.data)
            if fix:
                nav.update(*fix)

        elif msg.msg_type == 3000 and nav.valid:
            if pings_seen < pings_to_skip:
                pings_seen += 1
                continue

            beams, is_starboard = jsp.parse_bathy_3000(msg.data, msg.channel)
            ping_buf.append((nav.lat, nav.lon, nav.heading, beams, is_starboard))
            pings_seen += 1

            if len(ping_buf) >= chunk_size:
                path = flush_chunk(ping_buf, chunk_idx, output_dir, stem, grid_size)
                if path:
                    npy_files.append(path)
                chunk_idx += 1
                ping_buf   = []

    if ping_buf:
        path = flush_chunk(ping_buf, chunk_idx, output_dir, stem, grid_size)
        if path:
            npy_files.append(path)
        ping_buf = []

    return npy_files


# ---------------------------------------------------------------------------
# Main pipeline driver
# ---------------------------------------------------------------------------

def run_pipeline(target, output_dir, chunk_size, grid_size, resume,
                 do_merge_step, merge_name, merge_args):
    """
    Full pipeline:
      1. Discover logical files (base + continuations) in target
      2. Process each logical file into per-chunk .npy files
      3. Optionally merge all .npy files into a single cleaned mesh
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logical_files = jsp.build_logical_files(target)
    if not logical_files:
        print(f'No JSF files found at: {target}')
        sys.exit(1)

    print(f'Found {len(logical_files)} logical file(s):')
    for name, paths in logical_files:
        sizes    = [os.path.getsize(p) for p in paths]
        total_mb = sum(sizes) / 1e6
        print(f'  {name}  ({len(paths)} file(s), {total_mb:.0f} MB total)')
    print()

    all_npy = []

    for name, paths in logical_files:
        print(f'Processing: {name}')
        for p in paths:
            mb = os.path.getsize(p) / 1e6
            print(f'  File: {os.path.basename(p)}  ({mb:.0f} MB)')

        npy_files = process_logical_file(
            logical_name=name,
            file_paths=paths,
            output_dir=output_dir,
            chunk_size=chunk_size,
            grid_size=grid_size,
            resume=resume,
        )
        all_npy.extend(npy_files)
        print(f'  -> {len(npy_files)} chunk file(s) written.\n')

    print(f'All processing complete. Total chunk files: {len(all_npy)}')

    if do_merge_step and all_npy:
        print()
        from jsf_merge import do_merge
        do_merge(
            npy_files=all_npy,
            output_dir=output_dir,
            name=merge_name,
            grid_size=grid_size,
            **merge_args,
        )
    elif do_merge_step and not all_npy:
        print('No point cloud data produced; skipping merge.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Chunked bathymetry processor for large JSF sonar files.'
    )
    parser.add_argument('target',
        help='Path to a .jsf file or directory of .jsf files')
    parser.add_argument('--chunk-size',     type=int,   default=10_000, metavar='N')
    parser.add_argument('--grid-size',      type=float, default=0.2,    metavar='M')
    parser.add_argument('--output-dir',     default='output',           metavar='DIR')
    parser.add_argument('--no-merge',       action='store_true',
        help='Skip merge/triangulation after processing')
    parser.add_argument('--resume',         action='store_true',
        help='Skip chunks whose .npy output already exists (crash recovery)')
    parser.add_argument('--name',           default='merged',           metavar='NAME')

    # Reconstruction method
    parser.add_argument('--method',         default='grid',
        choices=['grid', 'delaunay', 'poisson', 'bpa'],
        help='Surface reconstruction algorithm (default: grid)')

    # grid options
    parser.add_argument('--step',           type=float, default=1.0,    metavar='S',
        help='[grid] Interpolation spacing in metres (default: 1.0)')

    # delaunay options
    parser.add_argument('--edge-pct',       type=float, default=95,     metavar='P',
        help='[delaunay] Edge-filter percentile (default: 95)')

    # poisson options
    parser.add_argument('--poisson-depth',  type=int,   default=9,      metavar='D',
        help='[poisson] Octree depth (default: 9)')
    parser.add_argument('--poisson-trim',   type=int,   default=5,      metavar='T',
        help='[poisson] Density trim percentile (default: 5)')

    # shared poisson/bpa options
    parser.add_argument('--normal-k',       type=int,   default=30,     metavar='K',
        help='[poisson/bpa] Neighbours for normal estimation (default: 30)')

    args = parser.parse_args()

    merge_args = dict(
        method        = args.method,
        step          = args.step,
        edge_pct      = args.edge_pct,
        poisson_depth = args.poisson_depth,
        poisson_trim  = args.poisson_trim,
        normal_k      = args.normal_k,
    )

    run_pipeline(
        target        = args.target,
        output_dir    = args.output_dir,
        chunk_size    = args.chunk_size,
        grid_size     = args.grid_size,
        resume        = args.resume,
        do_merge_step = not args.no_merge,
        merge_name    = args.name,
        merge_args    = merge_args,
    )


if __name__ == '__main__':
    main()
