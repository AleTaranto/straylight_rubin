#
#SCRIPT TO MAKE THE CROSS SECTION AND THE CONTOUR MAP OF THE AN IMAGES OR SOME SUMMED IMAGES
#
#
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import astropy.io.fits as fits
# from astropy.visualization import ZScaleInterval
from scipy.ndimage import gaussian_filter

import lsst.daf.butler as dafButler

from lsst.summit.utils.utils import getCameraFromInstrumentName
from lsst.summit.utils.plotting import stretchDataMidTone
from lsst.utils.plotting.figures import make_figure

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

import mosaic_utils_multiproc as mu

import os
import argparse


def plot_crossections(stack, pad_x=100, pad_y=600, width=150, offset_y=2100, mtf_stretch=False):
    pad_x = pad_x
    pad_y = pad_y
    width = width
    offset_y = offset_y
    
    bbox_x = [stack.shape[1] // 2 - width, stack.shape[1] // 2 + width]
    bbox_y = [stack.shape[0] // 2 - width, stack.shape[0] // 2 + width]
    
    cross_section_x = stack[bbox_x[0] : bbox_x[1], pad_x : -pad_x]
    cross_section_x = np.nanmean(cross_section_x, axis=0)
    
    cross_section_y = stack[pad_y : -pad_y, (bbox_y[0]) : (bbox_y[1])]
    cross_section_y = np.nanmean(cross_section_y, axis=1)
    
    cross_section_y2 = stack[pad_y : -pad_y, (bbox_y[0]) + offset_y : (bbox_y[1]) + offset_y]
    cross_section_y2 = np.nanmean(cross_section_y2, axis=1)
    
    layout = [(".", "axx_plot", "axy_plot", "."), (".", "img", "img", ".")]
    fig = make_figure(layout="constrained", figsize=(10,8))
    axs_dict = fig.subplot_mosaic(layout)
    
    axs_dict["axx_plot"].plot(cross_section_x, color="red")
    axs_dict["axy_plot"].plot(cross_section_y, color="blue")
    axs_dict["axy_plot"].plot(cross_section_y2, color="green")
    axs_dict["axx_plot"].sharey(axs_dict["axy_plot"])
    
    axs_dict["axx_plot"].set(
        xlabel="X-axis [pixels]", 
        ylabel="Intensity [ADU]", 
        title="Cross-section along X-axis"
        )
    axs_dict["axy_plot"].set(
        xlabel="Y-axis [pixels]", 
        title="Cross-section along Y-axis"
        )

    if mtf_stretch:
        # MTF stretch
        img = stretchDataMidTone(np.nan_to_num(stack, nan=np.nanmedian(stack)))
        img = np.where(np.isnan(stack), np.nan, img)
    else:
        img = stack
        
    axs_dict["img"].imshow(img, cmap="gray", origin="lower")
    axs_dict["img"].add_patch(
        Rectangle(
            [pad_x, (stack.shape[1] // 2) - width], 
            cross_section_x.size, 
            2*width, 
            edgecolor="red", 
            facecolor="none"
            )
        )
    axs_dict["img"].add_patch(
        Rectangle(
            [(stack.shape[0] // 2) - width, pad_y], 
            2*width, 
            cross_section_y.size, 
            edgecolor="blue", 
            facecolor="none"
        )
    )
    axs_dict["img"].add_patch(
        Rectangle(
            [(stack.shape[0] // 2) + offset_y - width, pad_y], 
            2*width, 
            cross_section_y2.size, 
            edgecolor="green", 
            facecolor="none")
        )
    _ = axs_dict["img"].set(
        xlabel="X-axis [pixels]", 
        ylabel="Y-axis [pixels]", 
        title="Image with Cross-sections"
    )

    return fig

def plot_mosaic_with_contours(stack, mtf_stretch=False, cmap="brg"):
    
    fig = make_figure(layout="constrained", figsize=(10, 10))
    ax = fig.add_subplot(111, aspect="equal")

    if mtf_stretch:
        # MTF stretch
        img = stretchDataMidTone(np.nan_to_num(stack, nan=np.nanmedian(stack)))
        img = np.where(np.isnan(stack), np.nan, img)
    else:
        img = stack
        
    # smoothing the data for contours 
    img_smooth = gaussian_filter(img, sigma=10.0)
    
    ax.imshow(
        img,
        origin="lower", 
        cmap=cmap
    )
    
    xGrid, yGrid = np.meshgrid(np.arange(img_smooth.shape[1]), np.arange(img_smooth.shape[0]))
    levels = np.linspace(np.nanmin(img_smooth), np.nanmax(img_smooth), 20)
    ax.contour(xGrid, yGrid, img_smooth, colors="white", linewidths=3.0, levels=levels)
    return fig

def parse_custom_list(value):
    try:
        # Rimuovi eventuali parentesi o spazi e separa per virgole
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
        raise argparse.ArgumentTypeError(
            f"Formato non valido per il parametro. Usa un numero (es. 3), "
            "una coppia (es. 1,5), o una lista (es. 1,2,4,7)"
        )

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Straylight photometry script")
    parser.add_argument(
    "--visit_list",
    type=parse_custom_list,
    required=True,
    help="List of visit IDs to process. Can accept single value (e.g., 2025051900419), "
         "a range (e.g., 2025051900419,2025051900429), or a list (e.g., 2025051900419,2025051900420,2025051900421)"
    )
    parser.add_argument(
        "--where",
        type=str,
        default=None,
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
    # parser.add_argument(
        # "--starless",
        # action="store_true",
        # help="Apply starless processing to the images"
    # )
    parser.add_argument(
        "--img_type",
        type=str,
        default="post_isr_image",
        help="Type of image to process (default: post_isr_image)"
    )
    args = parser.parse_args()

    # setup butler
    butler = dafButler.Butler(
        "embargo", 
        # collections=["LSSTCam/runs/nightlyValidation", "u/ataranto/lsst_outputs"], 
        collections=["u/ataranto/lsst_outputs"], 
        instrument=args.instrument
        )
    camera = getCameraFromInstrumentName(args.instrument)

    # get the (stack) mosaic
    # img_type = "post_isr_image"
    # instrument = "LSSTCam"
    # where_clause = None #"and detector in (90..98)"
    # binning = 8
    stack = mu.get_stacked_image(
        butler, 
        args.visit_list,
        args.img_type, 
        instrument=args.instrument, 
        where_clause=args.where, 
        binning=args.binning,
        # starless=args.starless,
    )

    # find distance form Antares
    target = SkyCoord.from_name("Antares")
    
    ra = butler.get("raw.visitInfo", exposure=args.visit_list[0], detector=94).boresightRaDec.getRa().asDegrees()
    dec = butler.get("raw.visitInfo", exposure=args.visit_list[0], detector=94).boresightRaDec.getDec().asDegrees()
    center = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    distance = target.separation(center).degree

    # get the filter name
    band = butler.get("raw.filter", exposure=args.visit_list[0], detector=94).bandLabel
    
    IMG_PATH = "/sdf/home/a/ataranto"
    FITS_PATH = "/sdf/home/a/ataranto"

    # save fits file
    hdu = fits.PrimaryHDU(stack)
    hdul = fits.HDUList([hdu])
    # if args.starless:
        # hdul.writeto(
            # os.path.join(FITS_PATH, f"{min(args.visit_list)}_{max(args.visit_list)}_band{band}_stacked_starless_dist2Ant_{distance:.2f}_deg.fits"), 
            # overwrite=True
            # )
    # else:
    hdul.writeto(
        os.path.join(FITS_PATH, f"{min(args.visit_list)}_{max(args.visit_list)}_band{band}_stacked.fits"), 
        overwrite=True
        )

    # cross sections
    #fig = plot_crossections(stack)
    #fig.savefig(
    #    os.path.join(
    #        IMG_PATH, 
    #        f"{min(args.visit_list)}_{max(args.visit_list)}_band{band}_cross_section_dist2Ant_{distance:.2f}_deg.png"
    #    ), 
    #    bbox_inches="tight", 
    #    dpi=300
        #)

    # img contours
   # fig = plot_mosaic_with_contours(stack, mtf_stretch=True)
    #fig.savefig(
    #    os.path.join(
    #        IMG_PATH, 
    #        f"{min(args.visit_list)}_{max(args.visit_list)}_band{band}_img_controus_dist2Ant_{distance:.2f}_deg.png"
    #    ), 
    #    bbox_inches="tight", 
    #    dpi=300
    #)
