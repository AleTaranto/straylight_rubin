import numpy as np
from matplotlib.patches import Rectangle
import astropy.io.fits as fits
from scipy.ndimage import gaussian_filter

import lsst.daf.butler as dafButler

from lsst.summit.utils.utils import getCameraFromInstrumentName
from lsst.summit.utils.plotting import stretchDataMidTone
from lsst.utils.plotting.figures import make_figure

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

# --- project bootstrap: common/ holds the shared modules ---
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "common"))
# -----------------------------------------------------------
import mosaic_utils_multiproc_new as mu

import os
import argparse
import time

def parse_custom_list(value: str):
    value = value.strip()

    # Caso 1: range con `..` (inclusivo)
    if ".." in value:
        try:
            start_str, end_str = value.split("..")
            start, end = int(start_str), int(end_str)
            if start > end:
                raise argparse.ArgumentTypeError(
                    f"Range non valido: {value}. Il primo valore deve essere <= del secondo."
                )
            return list(range(start, end + 1))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Formato range non valido: {value}. Usa ad esempio 1..5"
            )

    # Caso 2: lista di numeri separati da virgole
    if "," in value:
        try:
            return [int(x.strip()) for x in value.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Formato lista non valido: {value}. Usa ad esempio 1,3,7"
            )

    # Caso 3: singolo valore
    try:
        return [int(value)]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Valore non valido: {value}. Deve essere un numero intero."
        )

def parse_collections(value: str):
    if "," in value:
        return value.split(",")
    else:
        return [value]

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Python script to stack and calibrate LSSTCam images")
    parser.add_argument(
        "--visits",
        type=parse_custom_list,
        required=True,
        help="List of visit IDs to process. Can accept single value (e.g., 2025051900419), "
             "a range (e.g., 2025051900419..2025051900429), or a list (e.g., 2025051900419,2025051900420,2025051900421)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="embargo",
        help="Repo to use for the butler"
    )
    parser.add_argument(
        "--collections",
        type=parse_collections,
        default="LSSTCam/runs/nightlyValidation",
        help="A single o comma-separated list of collections where to find the images (default: LSSTCam/runs/nightlyValidation)"
    )
    parser.add_argument(
        "--where",
        type=str,
        default=None,
        help="Where clause to use in the query-datasets method of the butler instance"
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="LSSTCam",
        help="Instrument name (default: LSSTCam)"
    )
    parser.add_argument(
        "--binning",
        type=int,
        default=8,
        help="Binning factor (default: 8)"
    )
    parser.add_argument(
        "--img_type",
        type=str,
        default="post_isr_image",
        help="Type of image to process (default: post_isr_image)"
    )
    parser.add_argument(
        "--stack_type",
        type=str,
        default="median",
        help="Method to use for the stack [sum, mean, median] (default: median)"
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="None",
        help="Name of the output file, if None the filename will be 'firstvisit_lastvisit_stack.fits (default: None)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="None",
        help="The path for the output file, if None is the current workingdir (default: None)"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Wether or not do calibration process like dark subtraction and/or flat normalization"
    )
    parser.add_argument(
        "--master_flat",
        type=parse_custom_list,
        default=None,
        help="Visit(s) to use as master flat used for the calibration, argument required is --calibrate is set to True"
    )
    parser.add_argument(
        "--master_dark",
        type=parse_custom_list,
        default=None,
        help="Visit(s) to use as master dark used for the calibration, argument required is --calibrate is set to True"
    )
    args = parser.parse_args()

    if args.calibrate and (args.master_flat is None and args.master_dark is None):
        raise ValueError("Calibrate is set to True but neither master flat or master dark are provived.")

    # setup butler
    butler = dafButler.Butler(
        args.repo, 
        collections=args.collections, 
        instrument=args.instrument
        )
    butler_fl = dafButler.Butler(
        "embargo", 
        collections="u/ataranto/mono_flats", 
        instrument=args.instrument
        )
    camera = getCameraFromInstrumentName(args.instrument)

    ti = time.perf_counter()
    
    # get the (stack) mosaic
    stack = mu.get_stacked_image(
        butler, 
        butler_fl,
        args.visits,
        args.img_type, 
        instrument=args.instrument, 
        where_clause=args.where, 
        binning=args.binning,
        method=args.stack_type,
        calibrate=args.calibrate,
        master_dark_visits=args.master_dark,
        master_flat_visits=args.master_flat
    )
    print(f"Stack time: {time.perf_counter() - ti:.1f}s")
    
    FITS_PATH = args.output_path if args.output_path is not None else os.getcwd()
    if not os.path.exists(FITS_PATH):
        os.mkdir(FITS_PATH)

    # save fits file
    hdu = fits.PrimaryHDU(stack)
    hdul = fits.HDUList([hdu])

    if args.file_name is None:
        hdul.writeto(
            os.path.join(FITS_PATH, f"{min(args.visit_list)}_{max(args.visit_list)}_stack.fits"), 
            overwrite=True
            )
    else:
        hdul.writeto(
            os.path.join(FITS_PATH, args.file_name), 
            overwrite=True
            )
