#!/bin/bash
# Reorganize ~/straylight into a topic-based, repo-ready layout.
#
# NOTHING IS DELETED. Every operation is a move (mv) or a directory create.
# The inverse of this script is undo_reorg.sh, line for line.
#
# Dry run:  MV="echo mv" MKDIR="echo mkdir -p" LN="echo ln -s" bash reorg.sh
# Real run: bash reorg.sh

set -euo pipefail

ROOT="/sdf/home/a/ataranto/straylight"
cd "$ROOT" || { echo "FATAL: cannot cd to $ROOT" >&2; exit 1; }
[ "$(pwd -P)" = "$ROOT" ] || { echo "FATAL: wrong CWD: $(pwd -P)" >&2; exit 1; }

MV="${MV:-mv -n}"
MKDIR="${MKDIR:-mkdir -p}"
LN="${LN:-ln -s}"

echo "=== creating directory tree ==="
$MKDIR common scattered_light zemax_geometry imaging telemetry survey_sim
$MKDIR jobs data/raw data/ephemeris data/visit_lists data/pointings
$MKDIR outputs/figures archive

echo "=== common/ (shared importable modules) ==="
$MV geom_utils.py                      common/
$MV mosaic_utils.py                    common/
$MV mosaic_utils_multiproc_new.py      common/
$MV create_dot_env.py                  common/

echo "=== scattered_light/ ==="
$MV _scattered_light_compact.ipynb     scattered_light/
$MV _scattered_light_optimized.ipynb   scattered_light/
$MV test_LVV-T956-a.ipynb              scattered_light/
$MV artifact_snippet.ipynb             scattered_light/
$MV stray_light_photometry.py          scattered_light/

echo "=== zemax_geometry/ ==="
$MV sky_geometry.ipynb                 zemax_geometry/
$MV sky_geometry_nsc.ipynb             zemax_geometry/
$MV offaxis_to_nsc_positions.py        zemax_geometry/
$MV statistic_geometry.ipynb           zemax_geometry/

echo "=== imaging/ ==="
$MV image_visualizer.ipynb             imaging/
$MV romulan.ipynb                      imaging/
$MV L3_chamfer_analysis.ipynb          imaging/
$MV bright_star_fov.ipynb              imaging/
$MV tiling_mosaic.py                   imaging/
$MV new-wcs-mosaic.py                  imaging/
$MV sum.py                             imaging/
$MV stack_images.py                    imaging/
$MV script/calibrate_and_stack.py      imaging/
$MV script/romulan.py                  imaging/

echo "=== telemetry/ ==="
$MV Dome_track.ipynb                   telemetry/
$MV filter-offset.ipynb                telemetry/
$MV runout_rotator.ipynb               telemetry/
$MV mpi_visit_scan                     telemetry/
$MV script/script_sky_geom.py          telemetry/
# Untitled1.ipynb is NOT scratch: it is the titled analysis
# "HV currents during chaos monkey 20251126". Renamed, not archived.
$MV Untitled1.ipynb                    telemetry/hvbias_chaos_monkey_20251126.ipynb

echo "=== survey_sim/ ==="
$MV survey_sim.ipynb                   survey_sim/
$MV Template_coord.ipynb               survey_sim/
# Untitled-2.ipynb is the canonical, better-documented member of this pair
# (newer, 18 markdown cells, different survey epoch than survey_sim.ipynb).
$MV Untitled-2.ipynb                   survey_sim/rubin_year_simulation.ipynb

echo "=== jobs/ ==="
$MV mosaic-sbatch.sh                   jobs/
$MV stdout                             jobs/stdout

echo "=== data/ ==="
$MV stray_light_20260506.csv           data/raw/
$MV stray_light_20260313.csv           data/raw/
$MV de440s.bsp                         data/ephemeris/
$MV brush_VISIT.txt                    data/visit_lists/
$MV brush_query-v2.txt                 data/visit_lists/
$MV brush_query-v3.txt                 data/visit_lists/
$MV brush_MAG-VIS.txt                  data/visit_lists/
$MV TEMPLATE_pointings_offsets.npy     data/pointings/
$MV TEST_pointings_offsets.npy         data/pointings/

echo "=== outputs/ ==="
$MV img_e_data                         outputs/img_e_data
$MV rubin_mosaic_tiles                 outputs/rubin_mosaic_tiles
$MV tape_mosaic.png                    outputs/figures/
$MV polar_plots.gif                    outputs/figures/

echo "=== archive/ (superseded - kept, not deleted) ==="
# Genuinely superseded: 83% subsumed by _scattered_light_compact.ipynb
$MV _scattered_light.ipynb             archive/
$MV _scattered_light_original.ipynb    archive/
# Newest mtime in its family but a SUBSET of both parents: an unfinished merge
$MV "_scattered_light_compact copy.ipynb"             archive/
$MV "_scattered_light_compact copy.ipynb.original_bak" archive/
# Its helper functions were extracted into common/geom_utils.py
$MV sky_geometry_original.ipynb        archive/
# True scratch
$MV Untitled.ipynb                     archive/
$MV Untitled2.ipynb                    archive/
$MV script/romulan.py~                 archive/
$MV __pycache__                        archive/__pycache__

echo "=== root compatibility symlinks ==="
# Net for anything run from the root (L3_baffle scripts, archived notebooks).
$LN data/raw/stray_light_20260506.csv  stray_light_20260506.csv
$LN data/raw/stray_light_20260313.csv  stray_light_20260313.csv
$LN data/ephemeris/de440s.bsp          de440s.bsp
$LN common                             src

echo
echo "=== done. remaining at root: ==="
find . -maxdepth 1 -type f -printf '  %f\n' | sort
echo
echo "file count: $(find . -type f | wc -l)  (baseline was 162)"
echo "To roll back: bash undo_reorg.sh"
