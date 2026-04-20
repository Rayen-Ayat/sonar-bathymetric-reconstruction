# 3D Surface Reconstruction from Sonar Bathymetric Data

GVSU VR 3D Reconstruction · Mohamed Rayen Ayat · Supervisor: Dr. Haoyu Li

## Overview

This project parses EdgeTech JSF binary sonar recordings collected by an
IVER3-3011 AUV and reconstructs 3D bathymetric surfaces for three Lake Superior
sites: Brunette Park, Hunters Point, and McLain State Park. It implements a
streaming parser, a resumable chunked pipeline, grid-median cleaning, and
benchmarks four surface reconstruction algorithms (scipy `griddata`, Delaunay
triangulation, Poisson, and Ball Pivoting Algorithm) across coverage,
triangle quality, and runtime.

## Key results

Cross-lake averages across the three Lake Superior surveys.

| Algorithm | Time (s) | Tri. quality | Coverage | Gap   | Median edge (m) |
|-----------|---------:|-------------:|---------:|------:|----------------:|
| griddata  |     42.3 |         0.83 |      95% |   32% |            1.03 |
| delaunay  |      4.9 |         0.73 |     100% |   74% |            0.40 |
| poisson   |      2.9 |         0.72 |     100% |   46% |            1.23 |
| bpa       |      8.7 |         0.74 |      99% |   73% |            0.72 |

## Repository structure

```
.
├── jsf_parser.py                 # streaming EdgeTech JSF binary parser
├── jsf_pipeline.py               # chunked processing pipeline (10k pings/chunk, resumable)
├── jsf_merge.py                  # chunk merge, 0.2 m grid-median clean, outlier removal
├── jsf_crossval.py               # ping-level 80/20 cross-validation (RMSE)
├── jsf_overlap_check.py          # swath-overlap QC on independent measurements
├── jsf_inventory.py              # dataset inventory utility
├── benchmark_reconstruction.py   # runs all four reconstruction algorithms
├── output_brunette/              # merged .npy point cloud (Brunette Park)
├── output_hunters/               # merged .npy point cloud (Hunters Point)
├── output_mclain/                # merged .npy point cloud (McLain State Park)
├── benchmark_out/                # OBJ meshes: 4 algorithms × 3 surveys = 12
├── jsf_pipeline_code_documentation.pdf
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
```

## Installation

### Option A — pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate sonar-recon
```

## Pipeline usage

The workflow is six stages. Replace `<site>` with `brunette`, `hunters`, or
`mclain` and point `<jsf_dir>` at the folder holding the raw JSF recordings.

**1. Verify parsing on a single JSF file**

```bash
python jsf_parser.py
```

**2. Parse and chunk the full survey (resumable)**

```bash
python jsf_pipeline.py --input <jsf_dir> --output output_<site>/ --resume
```

**3. Merge chunks into a single cleaned point cloud**

```bash
python jsf_merge.py --input output_<site>/ --output output_<site>/merged.npy
```

**4. Reconstruct surfaces with all four algorithms**

```bash
python benchmark_reconstruction.py --input output_<site>/merged.npy --output benchmark_out/
```

**5. (Optional) Cross-validation for unbiased RMSE**

```bash
python jsf_crossval.py
```

**6. (Optional) Swath overlap QC on independent measurements**

```bash
python jsf_overlap_check.py
```

## Outputs

- `output_brunette/`, `output_hunters/`, `output_mclain/` each hold the
  merged `.npy` point cloud for that site after cleaning and outlier removal.
- `benchmark_out/` holds 12 OBJ meshes organised by site and algorithm. The
  OBJ files open directly in MeshLab, CloudCompare, or Blender.

## Algorithms

**griddata** — scipy's `griddata` interpolates depth onto a regular 2D grid
from the scattered sonar returns. It gives the highest triangle quality and
tight median edge length, at the cost of longer runtime.

**delaunay** — a planar Delaunay triangulation on (x, y) lifts each vertex to
its measured depth. It is the fastest triangulated surface and reaches full
coverage, but leaves the highest fraction of long-edge gaps because it
triangulates across sparse regions without filtering.

**poisson** — Open3D's screened Poisson reconstruction fits a watertight
implicit surface to oriented points. It is the fastest overall and closes
gaps aggressively, producing smooth surfaces that trade local fidelity for
completeness.

**bpa** — the Ball Pivoting Algorithm (Open3D) rolls a ball of tuned radius
over the point cloud and emits a triangle wherever it rests on three points.
It preserves sharp features well and yields near-complete coverage, but is
sensitive to point density and leaves gaps similar in scale to Delaunay.

## Documentation

Full code-level documentation is in
[jsf_pipeline_code_documentation.pdf](jsf_pipeline_code_documentation.pdf).

## License

MIT — see [LICENSE](LICENSE).

## Contact

Mohamed Rayen Ayat — ayatmohamedrayen@gmail.com
