#!/bin/bash
#SBATCH --job-name=rubin_5yr
#SBATCH --time=0-12:00:00
#SBATCH --mem=32GB
#SBATCH --account=rubin:commissioning
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --output=/sdf/home/a/ataranto/straylight/stdout/%x-%j.out
#SBATCH --error=/sdf/home/a/ataranto/straylight/stdout/%x-%j.err
#SBATCH --mail-user=alessio.taranto@inaf.it
##SBATCH --mail-type=START,END,FAIL

# Directory and Environment configuration
export RUBIN_SIM_DATA_DIR="$HOME/rubin-user/rubin_sim_data"
PYTHON="/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.1.0/bin/python"

SCRIPT_DIR="$HOME/straylight/L3_baffle"
#cd "$SCRIPT_DIR"

mkdir -p /sdf/home/a/ataranto/straylight/stdout

echo "=========================================================="
echo "Starting 1-Year Rubin Simulation with Horseshoe Detailer"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Script: $SCRIPT_DIR/simulate_one_year_with_detailer.py"
echo "Python: $PYTHON"
echo "=========================================================="

$PYTHON "$SCRIPT_DIR/simulate_one_year_with_detailer.py" \
    --survey-length 1826.25 \
    --survey-start-mjd 61284.0 \
    --output-dir "$SCRIPT_DIR/rubin_5_year_output_n64" \
    --db-filename "rubin_one_year_horseshoe_avoided.db" \
    --csv-filename "rubin_visits_horseshoe_avoided.csv" \
    --parquet-filename "rubin_visits_horseshoe_avoided.parquet" \
    --shifted-report-filename "horseshoe_shifted_visits.csv" \
    --vmag-limit 3.5 \
    --inner-radius 1.995 \
    --outer-radius 2.025 \
    --offset-ra 0.05 \
    --offset-dec 0.05 \
    --nside 64 \
    

echo "=========================================================="
echo "Job completed at $(date)"
echo "=========================================================="

