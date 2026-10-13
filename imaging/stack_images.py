import os
import numpy as np
from matplotlib.patches import Rectangle
import astropy.io.fits as fits
from scipy.ndimage import gaussian_filter
import time
import lsst.daf.butler as dafButler

from lsst.summit.utils.utils import getCameraFromInstrumentName
# from lsst.summit.utils.plotting import stretchDataMidTone
from lsst.utils.plotting.figures import make_figure

from astropy.coordinates import SkyCoord
from astropy import units as u

# --- project bootstrap: common/ holds the shared modules ---
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "common"))
# -----------------------------------------------------------
import mosaic_utils_multiproc_new as mu

# MPI
from mpi4py import MPI

def parse_custom_list(value):
    try:
        cleaned = value.strip("()[] ").replace(" ", "")
        parts = list(map(int, cleaned.split(",")))
        if len(parts) == 1:
            return [parts[0]]
        elif len(parts) == 2:
            start, end = parts
            return list(range(start, end + 1))
        elif len(parts) >= 3:
            return parts
        else:
            raise ValueError
    except Exception:
        raise ValueError(
            f"Formato non valido per il parametro. Usa un numero (es. 3), "
            "una coppia (es. 1,5), o una lista (es. 1,2,4,7)"
        )

# MPI init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set your parameters here (define on rank 0)
if rank == 0:
    st_time= time.time()
    visit_list = parse_custom_list("2025111900140,2025111900143")  # Example input
    where = None
    instrument = "LSSTCam"
    binning = 8
    img_type = "post_isr_image"
    collection = ["LSSTCam/runs/nightlyValidation"]
    stack_type = "sum"  # Options: "mean", "median", "sum"
    file_name = "NOFLAT_Yband_20deg_sum.fits"
    FITS_PATH = "/sdf/home/a/ataranto/rubin-user/20DEG-OFFAXIS/"
    background_path = os.path.join(FITS_PATH, "background_MASTER_20deg_0820.fits")
else:
    visit_list = None
    where = None
    instrument = None
    binning = None
    img_type = None
    collection = None
    stack_type = None
    file_name = None
    FITS_PATH = None
    background_path = None

# Broadcast configuration needed by all ranks (visit_list at minimum)
visit_list = comm.bcast(visit_list, root=0)
where = comm.bcast(where, root=0)
instrument = comm.bcast(instrument, root=0)
binning = comm.bcast(binning, root=0)
img_type = comm.bcast(img_type, root=0)
collection = comm.bcast(collection, root=0)
stack_type = comm.bcast(stack_type, root=0)
file_name = comm.bcast(file_name, root=0)
FITS_PATH = comm.bcast(FITS_PATH, root=0)
background_path = comm.bcast(background_path, root=0)

# Setup butler on each rank (if mu.get_stacked_image uses MPI internally it may expect per-rank butler)
butler = dafButler.Butler(
    "embargo",
    collections=collection,
    instrument=instrument
)
camera = getCameraFromInstrumentName(instrument)

# Get the (stack) mosaic - mu.get_stacked_image may use MPI internally
stack = mu.get_stacked_image(
    butler,
    butler,
    visit_list,
    img_type,
    instrument=instrument,
    where_clause=where,
    binning=binning,
    method=stack_type,
)

# Only rank 0 will have the final stacked image (common MPI pattern). If your mu.get_stacked_image
# returns the final array on every rank, the following rank check is still safe.
if rank == 0 and stack is not None:
    # Load the background image on rank 0 and subtract
    with fits.open(background_path) as bg_hdul:
        background = bg_hdul[0].data

    stack = stack - (len(visit_list) * background)

    hdu = fits.PrimaryHDU(stack)
    hdul = fits.HDUList([hdu])

    if file_name is None:
        out_name = os.path.join(FITS_PATH, f"{min(visit_list)}_{max(visit_list)}_stack.fits")
    else:
        out_name = os.path.join(FITS_PATH, file_name)

    hdul.writeto(out_name, overwrite=True)
    print(f"[rank 0] FITS file saved to: {out_name}")
    print(f"[rank 0] Total time taken: {time.time() - st_time:.2f} seconds")
# Ensure all ranks synchronize before exiting
comm.Barrier()
