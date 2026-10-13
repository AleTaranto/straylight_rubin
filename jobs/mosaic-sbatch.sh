#!/bin/bash
#
# Submit from the jobs/ directory:   cd ~/straylight/jobs && sbatch mosaic-sbatch.sh
# Slurm does not expand $HOME in #SBATCH lines, so the log paths below are
# relative to the submission directory - hence the requirement above.

#SBATCH --job-name=mosaic
#SBATCH --time=0-05:30:00
#SBATCH --mem=450GB
#SBATCH --account=rubin:commissioning
#SBATCH --partition=milano
#SBATCH --ntasks-per-node=60
#SBATCH --nodes=2
#SBATCH --qos=normal
#SBATCH --output=./stdout/%x-%j.out
#SBATCH --error=./stdout/%j.err
#SBATCH --mail-user=alessio.taranto@inaf.it
#SBATCH --mail-type=START,END,FAIL


# tiling_mosaic.py writes to ../outputs/rubin_mosaic_tiles, relative to imaging/,
# so the working directory must be imaging/.
cd $HOME/straylight/imaging/
setup_sitcom
echo "sumbitted job for mpi_skymap..."
mpirun -np 120 python $HOME/straylight/imaging/tiling_mosaic.py
echo "DONE"
