"""
jsf_inventory.py — Streaming JSF file scanner and reporter.

Usage:
    python jsf_inventory.py <file_or_directory> [--sample-pings N]

Scans JSF files WITHOUT loading them fully into memory and reports:
  - File size
  - Message count by type
  - Whether Type 3000 (bathymetric) data is present
  - Depth range and beam validity rate from the first N Type-3000 pings
  - Whether the file appears to be binned or raw sonar data
  - Continuation files detected (.001, .002, ...)
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np

import jsf_parser as jsp


# ---------------------------------------------------------------------------
# Per-file scan
# ---------------------------------------------------------------------------

def scan_file(path, sample_pings=100):
    """
    Stream-scan a single JSF file and return a stats dict.
    Never loads the whole file into memory.
    """
    stats = {
        'path':              path,
        'file_size':         os.path.getsize(path),
        'msg_counts':        defaultdict(int),
        'type80_sizes':      [],      # first 50 Type-80 data sizes
        'bathy_depths':      [],      # depth_m from first N valid beams across pings
        'bathy_valid_total': 0,
        'bathy_invalid_total': 0,
        'bathy_ping_count':  0,
        'has_nav':           False,
        'nav_samples':       [],      # first 10 (lat, lon) pairs
        'error':             None,
    }

    try:
        with open(path, 'rb') as f:
            for msg in jsp.iter_messages(f):
                mtype = msg.msg_type
                stats['msg_counts'][mtype] += 1

                # --- Type 80: sidescan sonar ---
                if mtype == 80 and len(stats['type80_sizes']) < 50:
                    stats['type80_sizes'].append(len(msg.data))

                # --- Type 2002: navigation ---
                elif mtype == 2002:
                    nav = jsp.parse_nav_2002(msg.data)
                    if nav:
                        stats['has_nav'] = True
                        if len(stats['nav_samples']) < 10:
                            stats['nav_samples'].append((nav[0], nav[1]))

                # --- Type 3000: bathymetry (sample only first N pings) ---
                elif mtype == 3000:
                    if stats['bathy_ping_count'] < sample_pings:
                        beams, _ = jsp.parse_bathy_3000(msg.data, msg.channel)
                        if len(beams) == 0:
                            stats['bathy_ping_count'] += 1
                            continue
                        valid_mask = np.isin(
                            beams['quality_flag'], list(jsp.VALID_FLAGS)
                        )
                        n_valid   = int(valid_mask.sum())
                        n_invalid = len(beams) - n_valid
                        stats['bathy_valid_total']   += n_valid
                        stats['bathy_invalid_total'] += n_invalid
                        if n_valid:
                            depths = beams['depth_mm'][valid_mask].astype(float) / 1000.0
                            stats['bathy_depths'].extend(depths.tolist())
                        stats['bathy_ping_count'] += 1

    except OSError as e:
        stats['error'] = str(e)

    return stats


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_file(stats):
    """Return 'binned', 'raw', 'no-imagery', or 'unknown'."""
    sizes = stats['type80_sizes']
    if not sizes:
        return 'no-imagery'
    avg = sum(sizes) / len(sizes)
    if avg < 12_000:
        return 'binned'
    return 'raw'


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_bytes(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} TB'


def print_file_report(stats, continuations):
    path = stats['path']
    basename = os.path.basename(path)

    print()
    print(f'=== {basename} ({fmt_bytes(stats["file_size"])}) ===')

    if stats['error']:
        print(f'  ERROR: {stats["error"]}')
        return

    # Message counts
    counts = stats['msg_counts']
    if not counts:
        print('  No messages found (empty or corrupt file).')
        return

    total = sum(counts.values())
    print(f'  Total messages: {total:,}')
    known = {
        80:   'sidescan sonar',
        2002: 'navigation (NMEA)',
        3000: 'bathymetric beams',
    }
    for mtype in sorted(counts):
        label = known.get(mtype, '')
        desc  = f'  ({label})' if label else ''
        print(f'    Type {mtype:>5}{desc}: {counts[mtype]:,}')

    # Navigation
    print(f'  Navigation (Type 2002): {"YES" if stats["has_nav"] else "NO"}')
    if stats['nav_samples']:
        s = stats['nav_samples']
        lat_min = min(p[0] for p in s)
        lat_max = max(p[0] for p in s)
        lon_min = min(p[1] for p in s)
        lon_max = max(p[1] for p in s)
        print(f'    Lat range (first {len(s)} fixes): {lat_min:.6f} – {lat_max:.6f}')
        print(f'    Lon range (first {len(s)} fixes): {lon_min:.6f} – {lon_max:.6f}')

    # Bathymetry
    has_bathy = 3000 in counts
    print(f'  Bathymetry (Type 3000): {"YES" if has_bathy else "NO"}')
    if has_bathy:
        n_pings = stats['bathy_ping_count']
        n_valid = stats['bathy_valid_total']
        n_inv   = stats['bathy_invalid_total']
        n_total = n_valid + n_inv
        if n_total > 0:
            pct_valid = 100.0 * n_valid / n_total
        else:
            pct_valid = 0.0

        print(f'    Sampled pings: {n_pings}')
        print(f'    Beam validity rate: {pct_valid:.1f}% valid '
              f'({n_valid:,} valid / {n_total:,} total)')

        depths = stats['bathy_depths']
        if depths:
            print(f'    Depth range (valid beams): '
                  f'{min(depths):.3f} – {max(depths):.3f} m')
            print(f'    Depth median: {sorted(depths)[len(depths)//2]:.3f} m')

        total_bathy = counts[3000]
        if n_pings > 0 and pct_valid > 0 and n_total > 0:
            avg_valid_per_ping = n_valid / n_pings
            est_valid = int(avg_valid_per_ping * total_bathy)
            print(f'    Estimated valid beams in full file: ~{est_valid:,}')

    # Classification
    mode = classify_file(stats)
    sizes = stats['type80_sizes']
    if sizes:
        avg_size = int(sum(sizes) / len(sizes))
        print(f'  Classification: {mode}  '
              f'(avg Type-80 msg = {avg_size:,} bytes, n={len(sizes)})')
    else:
        print(f'  Classification: {mode}')

    # Continuation files
    if continuations:
        print(f'  Continuation files ({len(continuations)}):')
        for cp in continuations:
            print(f'    {os.path.basename(cp)}  ({fmt_bytes(os.path.getsize(cp))})')
    else:
        print('  Continuation files: none')


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_stats):
    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    header = (
        f'{"File":<45} {"Size":>8} {"Type3000":>10} {"Valid%":>7} '
        f'{"Class":<10} {"Conts":>5}'
    )
    print(header)
    print('-' * 80)

    for stats, conts in all_stats:
        if stats['error']:
            name = os.path.basename(stats['path'])[:44]
            print(f'{name:<45} ERROR: {stats["error"]}')
            continue

        name     = os.path.basename(stats['path'])[:44]
        size_str = fmt_bytes(stats['file_size'])
        bathy_n  = stats['msg_counts'].get(3000, 0)

        n_valid  = stats['bathy_valid_total']
        n_total  = n_valid + stats['bathy_invalid_total']
        pct_str  = f'{100*n_valid/n_total:.0f}%' if n_total else 'n/a'

        mode     = classify_file(stats)
        n_conts  = len(conts)

        print(
            f'{name:<45} {size_str:>8} {bathy_n:>10,} {pct_str:>7} '
            f'{mode:<10} {n_conts:>5}'
        )

    print('=' * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Scan JSF sonar files and report message statistics.'
    )
    parser.add_argument(
        'target',
        help='Path to a single .jsf file or a directory containing .jsf files'
    )
    parser.add_argument(
        '--sample-pings', type=int, default=100, metavar='N',
        help='Number of Type-3000 pings to sample for depth/validity stats (default: 100)'
    )
    args = parser.parse_args()

    target = os.path.abspath(args.target)

    if os.path.isfile(target):
        files_to_scan = [target]
    elif os.path.isdir(target):
        files_to_scan = []
        for fname in sorted(os.listdir(target)):
            if fname.lower().endswith('.jsf'):
                files_to_scan.append(os.path.join(target, fname))
        if not files_to_scan:
            print(f'No .jsf files found in {target}')
            sys.exit(1)
    else:
        print(f'Error: {target} is not a file or directory.')
        sys.exit(1)

    all_stats = []
    for path in files_to_scan:
        print(f'\nScanning {os.path.basename(path)} ...', end=' ', flush=True)
        stats = scan_file(path, sample_pings=args.sample_pings)
        continuations = jsp.find_continuation_files(path)
        all_stats.append((stats, continuations))
        print('done.')
        print_file_report(stats, continuations)

    if len(files_to_scan) > 1:
        print_summary_table(all_stats)


if __name__ == '__main__':
    main()
