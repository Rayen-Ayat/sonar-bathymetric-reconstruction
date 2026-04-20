"""
jsf_parser.py — Shared streaming parser for EdgeTech JSF sonar files.

Provides:
  - iter_messages(file_obj)         : generator, one JsfMessage at a time
  - iter_logical_file(file_paths)   : chain multiple files transparently
  - parse_nav_2002(data)            : extract (lat, lon, heading) from Type 2002
  - parse_bathy_3000(data, channel) : extract beam array from Type 3000
  - georef_beams(...)               : vectorised georeferencing → POINT_DTYPE array
  - grid_median_clean(pts, size_m)  : 0.2 m grid median filter
  - find_continuation_files(path)   : find WP14.001_Stave.jsf siblings
  - NavState                        : mutable nav fix holder

Dtypes exported: BEAM_DTYPE, POINT_DTYPE, VALID_FLAGS
"""

import os
import re
import gc
import struct
from collections import namedtuple
from math import cos, radians

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JSF_MARKER = 0x1601
HEADER_SIZE = 16

BEAM_DTYPE = np.dtype([
    ('across_track_mm', np.uint16),
    ('depth_mm',        np.uint16),
    ('amplitude',       np.uint16),
    ('quality_flag',    np.uint16),
])

POINT_DTYPE = np.dtype([
    ('lon',   np.float64),
    ('lat',   np.float64),
    ('depth', np.float64),
])

VALID_FLAGS = frozenset({0xD400, 0xF400})

# Regex patterns for continuation file detection
# Base:  ...WP14_Stave.jsf  (no dot-number before _Stave)
# Cont:  ...WP14.001_Stave.jsf
_BASE_RE = re.compile(r'^(.+_WP\d+)(_[^.]+\.[Jj][Ss][Ff])$')
_CONT_RE = re.compile(r'^(.+_WP\d+)\.(\d+)(_[^.]+\.[Jj][Ss][Ff])$')

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

JsfMessage = namedtuple('JsfMessage', ['msg_type', 'channel', 'data'])


class NavState:
    """Mutable holder for the most-recent navigation fix."""
    __slots__ = ('lat', 'lon', 'heading', 'valid')

    def __init__(self):
        self.lat = self.lon = self.heading = 0.0
        self.valid = False

    def update(self, lat, lon, heading):
        self.lat = lat
        self.lon = lon
        self.heading = heading
        self.valid = True

    def __repr__(self):
        if not self.valid:
            return 'NavState(invalid)'
        return f'NavState(lat={self.lat:.6f}, lon={self.lon:.6f}, hdg={self.heading:.2f})'


# ---------------------------------------------------------------------------
# Core streaming iterator
# ---------------------------------------------------------------------------

def iter_messages(file_obj):
    """
    Generator: yield one JsfMessage at a time from an open binary file object.
    Never buffers the full file. Resyncs on bad markers.
    """
    f = file_obj
    # Read-ahead buffer for resync
    pending = b''

    while True:
        # Try to fill a full header
        raw = f.read(HEADER_SIZE - len(pending))
        if not raw:
            return  # EOF

        hdr = pending + raw
        pending = b''

        if len(hdr) < HEADER_SIZE:
            return  # truncated at EOF

        marker = struct.unpack_from('<H', hdr, 0)[0]

        if marker != JSF_MARKER:
            # Resync: scan forward byte by byte within what we have, then in file
            resync_buf = hdr
            found = False
            while True:
                idx = resync_buf.find(b'\x01\x16')
                if idx >= 0:
                    # Potential marker at resync_buf[idx]
                    # Need at least 16 bytes from that position
                    remaining = resync_buf[idx:]
                    if len(remaining) < HEADER_SIZE:
                        extra = f.read(HEADER_SIZE - len(remaining))
                        if not extra:
                            return
                        remaining = remaining + extra
                    # Verify it's really the marker (little-endian 0x1601)
                    if struct.unpack_from('<H', remaining, 0)[0] == JSF_MARKER:
                        pending = remaining
                        found = True
                        break
                    else:
                        # False positive, skip past it
                        resync_buf = remaining[2:]
                else:
                    chunk = f.read(4096)
                    if not chunk:
                        return
                    resync_buf = chunk

            if not found:
                return

            # Retry with the recovered header in pending
            continue

        # Parse header fields
        msg_type  = struct.unpack_from('<H', hdr, 4)[0]
        channel   = hdr[8]
        data_size = struct.unpack_from('<I', hdr, 12)[0]

        # Read payload
        if data_size == 0:
            yield JsfMessage(msg_type, channel, b'')
            continue

        data = f.read(data_size)
        if len(data) < data_size:
            return  # truncated at EOF

        yield JsfMessage(msg_type, channel, data)


def iter_logical_file(file_paths):
    """
    Generator: chain multiple JSF files as a single logical stream.
    Nav state and chunk counters in callers persist naturally across boundaries.
    """
    for path in file_paths:
        with open(path, 'rb') as f:
            yield from iter_messages(f)


# ---------------------------------------------------------------------------
# Type 2002 — Navigation
# ---------------------------------------------------------------------------

def parse_nav_2002(data):
    """
    Parse a Type 2002 navigation payload.
    Returns (lat, lon, heading) as floats, or None on any parse failure.

    Payload layout:
      Bytes 0-11  : 12-byte binary prefix (Unix epoch uint32 + other fields)
      Bytes 12+   : ASCII NMEA string
        $__ETC,HHMMSS.sss,DD,MM,YYYY,0,LAT,LON,HEADING,...
        fields (0-indexed): 6=lat, 7=lon, 8=heading
    """
    try:
        # Find the start of the NMEA sentence
        dollar = data.find(b'$')
        if dollar < 0:
            return None
        nmea = data[dollar:].decode('ascii', errors='replace')
        fields = nmea.split(',')
        if len(fields) < 9:
            return None
        lat     = float(fields[6])
        lon     = float(fields[7])
        heading = float(fields[8])
        # Basic sanity
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None
        if not (0 <= heading < 360):
            heading = heading % 360
        return (lat, lon, heading)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Type 3000 — Bathymetric beams
# ---------------------------------------------------------------------------

def parse_bathy_3000(data, channel):
    """
    Parse a Type 3000 bathymetric payload.
    Returns (beams_array, is_starboard).

    Payload layout:
      Bytes 0-79  : 80-byte sub-header (ignored)
      Bytes 80+   : 400 beams × 8 bytes each (BEAM_DTYPE)

    channel: header byte 8 — 0 or 512 = port, 1 or 513 = starboard
    """
    BATHY_SUBHEADER = 80
    EXPECTED_BEAM_BYTES = 400 * 8  # 3200

    is_starboard = bool(channel & 0x01)  # odd = starboard

    if len(data) < BATHY_SUBHEADER + EXPECTED_BEAM_BYTES:
        # Truncated — return empty array
        return np.empty(0, dtype=BEAM_DTYPE), is_starboard

    beam_data = data[BATHY_SUBHEADER: BATHY_SUBHEADER + EXPECTED_BEAM_BYTES]
    beams = np.frombuffer(beam_data, dtype=BEAM_DTYPE)  # shape (400,)
    return beams, is_starboard


# ---------------------------------------------------------------------------
# Georeferencing
# ---------------------------------------------------------------------------

def georef_beams(beams, lat, lon, heading_deg, is_starboard):
    """
    Convert valid beams to georeferenced (lon, lat, depth) points.

    Returns a structured array with dtype POINT_DTYPE.
    Returns empty array if no valid beams.
    """
    if len(beams) == 0:
        return np.empty(0, dtype=POINT_DTYPE)

    valid_mask = np.isin(beams['quality_flag'], list(VALID_FLAGS))
    valid_beams = beams[valid_mask]

    if len(valid_beams) == 0:
        return np.empty(0, dtype=POINT_DTYPE)

    heading_rad = np.radians(heading_deg)
    lat_rad     = radians(lat)
    sign        = 1.0 if is_starboard else -1.0
    perp_angle  = heading_rad + sign * (np.pi / 2.0)

    distance_m = valid_beams['across_track_mm'].astype(np.float64) / 1000.0
    depth_m    = valid_beams['depth_mm'].astype(np.float64) / 1000.0

    cos_perp = np.cos(perp_angle)
    sin_perp = np.sin(perp_angle)
    cos_lat  = cos(lat_rad)

    lat_offset = (distance_m * cos_perp) / 111320.0
    lon_offset = (distance_m * sin_perp) / (111320.0 * cos_lat)

    result = np.empty(len(valid_beams), dtype=POINT_DTYPE)
    result['lat']   = lat + lat_offset
    result['lon']   = lon + lon_offset
    result['depth'] = depth_m

    return result


# ---------------------------------------------------------------------------
# Grid median cleaning
# ---------------------------------------------------------------------------

def grid_median_clean(pts, grid_size_m=0.2):
    """
    Reduce a point cloud to one representative point per grid cell using
    the median value within each cell.

    pts        : structured array with dtype POINT_DTYPE
    grid_size_m: cell size in metres (default 0.2 m)

    Returns a cleaned structured array (same dtype).
    """
    if len(pts) == 0:
        return pts

    mean_lat = float(np.mean(pts['lat']))
    lat_step = grid_size_m / 111320.0
    lon_step = grid_size_m / (111320.0 * cos(radians(mean_lat)))

    lat_idx = np.floor(pts['lat'] / lat_step).astype(np.int64)
    lon_idx = np.floor(pts['lon'] / lon_step).astype(np.int64)

    lat_idx -= lat_idx.min()
    lon_idx -= lon_idx.min()
    max_col  = int(lon_idx.max()) + 1
    cell_key = lat_idx * max_col + lon_idx

    order   = np.argsort(cell_key, kind='stable')
    s_keys  = cell_key[order]
    s_pts   = pts[order]

    _, group_starts, group_sizes = np.unique(
        s_keys, return_index=True, return_counts=True
    )
    n_cells = len(group_starts)
    out = np.empty(n_cells, dtype=POINT_DTYPE)

    for i in range(n_cells):
        s = group_starts[i]
        e = s + group_sizes[i]
        g = s_pts[s:e]
        out[i]['lon']   = float(np.median(g['lon']))
        out[i]['lat']   = float(np.median(g['lat']))
        out[i]['depth'] = float(np.median(g['depth']))

    return out


# ---------------------------------------------------------------------------
# Continuation file discovery
# ---------------------------------------------------------------------------

def find_continuation_files(base_path):
    """
    Given a base JSF file path (e.g. .../WP14_Stave.jsf), return a sorted
    list of continuation file paths (e.g. [.../WP14.001_Stave.jsf]).

    Returns an empty list if none are found or if base_path doesn't match
    the expected naming pattern.
    """
    directory = os.path.dirname(os.path.abspath(base_path))
    basename  = os.path.basename(base_path)

    m = _BASE_RE.match(basename)
    if not m:
        return []

    stem   = m.group(1)   # e.g. "20190812_..._WP14"
    suffix = m.group(2)   # e.g. "_Stave.jsf"
    suffix_lower = suffix.lower()

    continuations = []
    try:
        entries = os.listdir(directory)
    except OSError:
        return []

    for fname in entries:
        cm = _CONT_RE.match(fname)
        if cm is None:
            continue
        if cm.group(1) != stem:
            continue
        if cm.group(3).lower() != suffix_lower:
            continue
        index = int(cm.group(2))
        continuations.append((index, os.path.join(directory, fname)))

    continuations.sort(key=lambda x: x[0])
    return [path for _, path in continuations]


def build_logical_files(input_path):
    """
    Given a file or directory path, return a list of:
        (logical_name, [base_file_path, *continuation_paths])

    For a single file: chains base + auto-discovered continuations.
    For a directory:   discovers all base .jsf files and chains each.
    """
    input_path = os.path.abspath(input_path)

    if os.path.isfile(input_path):
        conts = find_continuation_files(input_path)
        name  = os.path.splitext(os.path.basename(input_path))[0]
        return [(name, [input_path] + conts)]

    # Directory mode
    try:
        entries = sorted(os.listdir(input_path))
    except OSError as e:
        raise ValueError(f"Cannot list directory {input_path}: {e}")

    result = []
    for fname in entries:
        if not re.search(r'\.[Jj][Ss][Ff]$', fname):
            continue
        if _CONT_RE.match(fname):
            continue  # skip continuation files; they'll be chained to their base
        full = os.path.join(input_path, fname)
        if not os.path.isfile(full):
            continue
        conts = find_continuation_files(full)
        name  = os.path.splitext(fname)[0]
        result.append((name, [full] + conts))

    return result
