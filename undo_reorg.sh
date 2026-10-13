#!/bin/bash
# Exact inverse of reorg.sh. Restores the flat layout.
#
# This is the rollback path for the reorganization while the folder is not yet
# a git repo. It removes only the directories reorg.sh created (and only when
# they are empty) and the symlinks reorg.sh made. It never removes a real file.
#
# Dry run:  MV="echo mv" bash undo_reorg.sh
# Real run: bash undo_reorg.sh

set -euo pipefail

ROOT="/sdf/home/a/ataranto/straylight"
cd "$ROOT" || { echo "FATAL: cannot cd to $ROOT" >&2; exit 1; }
[ "$(pwd -P)" = "$ROOT" ] || { echo "FATAL: wrong CWD: $(pwd -P)" >&2; exit 1; }

MV="${MV:-mv -n}"

echo "=== removing compatibility symlinks ==="
for l in stray_light_20260506.csv stray_light_20260313.csv de440s.bsp src; do
    [ -L "$l" ] && rm -v "$l" || true
done

echo "=== archive/ -> root ==="
$MV archive/_scattered_light.ipynb                            .
$MV archive/_scattered_light_original.ipynb                   .
$MV "archive/_scattered_light_compact copy.ipynb"             .
$MV "archive/_scattered_light_compact copy.ipynb.original_bak" .
$MV archive/sky_geometry_original.ipynb                       .
$MV archive/Untitled.ipynb                                    .
$MV archive/Untitled2.ipynb                                   .
$MV archive/__pycache__                                       __pycache__
mkdir -p script
$MV "archive/romulan.py~"                                     script/

echo "=== outputs/ -> root ==="
$MV outputs/figures/polar_plots.gif      .
$MV outputs/figures/tape_mosaic.png      .
$MV outputs/rubin_mosaic_tiles           rubin_mosaic_tiles
$MV outputs/img_e_data                   img_e_data

echo "=== data/ -> root ==="
$MV data/pointings/TEST_pointings_offsets.npy      .
$MV data/pointings/TEMPLATE_pointings_offsets.npy  .
$MV data/visit_lists/brush_MAG-VIS.txt             .
$MV data/visit_lists/brush_query-v3.txt            .
$MV data/visit_lists/brush_query-v2.txt            .
$MV data/visit_lists/brush_VISIT.txt               .
$MV data/ephemeris/de440s.bsp                      .
$MV data/raw/stray_light_20260313.csv              .
$MV data/raw/stray_light_20260506.csv              .

echo "=== jobs/ -> root ==="
$MV jobs/stdout           stdout
$MV jobs/mosaic-sbatch.sh .

echo "=== survey_sim/ -> root ==="
$MV survey_sim/rubin_year_simulation.ipynb Untitled-2.ipynb
$MV survey_sim/Template_coord.ipynb        .
$MV survey_sim/survey_sim.ipynb            .

echo "=== telemetry/ -> root ==="
$MV telemetry/hvbias_chaos_monkey_20251126.ipynb Untitled1.ipynb
$MV telemetry/script_sky_geom.py                 script/
$MV telemetry/mpi_visit_scan                     .
$MV telemetry/runout_rotator.ipynb               .
$MV telemetry/filter-offset.ipynb                .
$MV telemetry/Dome_track.ipynb                   .

echo "=== imaging/ -> root ==="
$MV imaging/romulan.py               script/
$MV imaging/calibrate_and_stack.py   script/
$MV imaging/stack_images.py          .
$MV imaging/sum.py                   .
$MV imaging/new-wcs-mosaic.py        .
$MV imaging/tiling_mosaic.py         .
$MV imaging/bright_star_fov.ipynb    .
$MV imaging/L3_chamfer_analysis.ipynb .
$MV imaging/romulan.ipynb            .
$MV imaging/image_visualizer.ipynb   .

echo "=== zemax_geometry/ -> root ==="
$MV zemax_geometry/statistic_geometry.ipynb     .
$MV zemax_geometry/offaxis_to_nsc_positions.py  .
$MV zemax_geometry/sky_geometry_nsc.ipynb       .
$MV zemax_geometry/sky_geometry.ipynb           .

echo "=== scattered_light/ -> root ==="
$MV scattered_light/stray_light_photometry.py        .
$MV scattered_light/artifact_snippet.ipynb           .
$MV scattered_light/test_LVV-T956-a.ipynb            .
$MV scattered_light/_scattered_light_optimized.ipynb .
$MV scattered_light/_scattered_light_compact.ipynb   .

echo "=== common/ -> root ==="
$MV common/create_dot_env.py               .
$MV common/mosaic_utils_multiproc_new.py   .
$MV common/mosaic_utils.py                 .
$MV common/geom_utils.py                   .

echo "=== removing now-empty directories ==="
# rmdir only - refuses to remove a non-empty dir, so nothing can be lost here.
for d in common scattered_light zemax_geometry imaging telemetry survey_sim \
         jobs data/raw data/ephemeris data/visit_lists data/pointings data \
         outputs/figures outputs archive; do
    [ -d "$d" ] && rmdir "$d" 2>/dev/null && echo "  removed $d/" || true
done

echo
echo "=== done. file count: $(find . -type f | wc -l)  (baseline was 162) ==="
echo "Any directory NOT removed above still has files in it - inspect it."
