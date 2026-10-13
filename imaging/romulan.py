import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
import numpy as np
import gc
import lsst.daf.butler as DafButler
import lsst.summit.utils.plotting as splot
import PIL
from astropy.coordinates import SkyCoord
import lsst.afw.display as afwDisplay
from lsst.pipe.tasks.visualizeVisit import (
    VisualizeBinExpConfig,
    VisualizeBinExpTask,
    VisualizeMosaicExpConfig,
    VisualizeMosaicExpTask,
)

from lsst.summit.utils.utils import getCameraFromInstrumentName
from photutils.segmentation import detect_sources, detect_threshold
from photutils.utils import circular_footprint
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel 
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

def bin_image(image, camera, binning=8):
    visualizeBinExpConfig = VisualizeBinExpConfig()
    visualizeBinExpConfig.binning = binning
    visualizeBinExpTask = VisualizeBinExpTask(config=visualizeBinExpConfig)
    return visualizeBinExpTask.run(inputExp=image, camera=camera).outputExp

def doStarless(img):
    kernel = make_2dgaussian_kernel(1, size=1)   
    threshold = detect_threshold(img, 3.)
    segment_map = detect_sources(img, threshold, npixels=4)
    footprint = circular_footprint(radius=1)
    dilateMask = segment_map.make_source_mask(footprint=footprint)
    dilateAndSmoothMask = convolve(dilateMask, kernel)

    starlessImg = np.where(dilateAndSmoothMask != 0, np.median(img), img)
    return starlessImg

#Collections=['LSSTComCam/runs/nightlyValidation']
butler = DafButler.Butler('embargo', collections=["LSSTCam/runs/nightlyValidation",'u/ataranto/lsst_outputs'], instrument='LSSTCam')
registry = butler.registry
camera = getCameraFromInstrumentName("LSSTCam")

plt.close('all')
def so_deg(val):
    return (float(val)*u.rad).to(u.deg).value


day_obs=20250515
frames=[]
al_hms='14h39m36.495s'
al_dms='-60d50m2.3s'
alpha = SkyCoord(ra=al_hms, dec=al_dms, unit=(u.hourangle, u.deg), frame='icrs')

print(alpha.ra.deg)
#alpha = SkyCoord(ra=ra_alp*u.deg, dec=dec_alp*u.deg, frame='icrs')
print(alpha.dec.deg)



Dref_uns = list(registry.queryDatasets('post_isr_image', where=f"visit={2025051500393} and detector=94"))  #corner dtectors 188, 161, 65, 20, 0, 27, 123, 168
Dref=sorted( Dref_uns, key = lambda x : x.dataId['detector'] )
dataImgs = [butler.get(dr) for dr in Dref]
wcs = dataImgs[0].wcs #.getFitsMetadata().items()
hold=wcs
for seq in range(440,445):
    plt.close('all')
    visit=day_obs*100000+seq
    print('visit   ',visit)
    Dref_uns = list(registry.queryDatasets('post_isr_image', where=f"visit={visit} and detector in (157,158,160)"))  #corner dtectors 188, 161, 65, 20, 0, 27, 123, 168
    Dref=sorted( Dref_uns, key = lambda x : x.dataId['detector'] )
    # print(Dref)
    exposures = [butler.get(dr) for dr in Dref]
    wcs = dataImgs[0].wcs #.getFitsMetadata().items()
    center_ra, center_dec = wcs.getSkyOrigin()
    center = SkyCoord(ra=so_deg(center_ra)*u.deg, dec=so_deg(center_dec)*u.deg, frame='icrs')
    # print(center.ra.deg)
#alpha = SkyCoord(ra=ra_alp*u.deg, dec=dec_alp*u.deg, frame='icrs')
    # print(center.dec.deg)
    print('alpha cen off axis:   ', alpha.separation(center).deg, 'deg ///   ',alpha.separation(center).deg*3600.0, ' arcsec ' )
    print('-----------------')
    binning =8
    #uncomment for star removal
    
    
    exposures_binned = [bin_image(exposure, camera, binning) for exposure in exposures]
    
    for i, exp in enumerate(exposures_binned):
            starlessImg = doStarless(exp.image.array)
            exp.image.array = starlessImg
            exposures_binned[i] = exp
    
    visualizeMosaicExpConfig = VisualizeMosaicExpConfig()
    visualizeMosaicExpConfig.binning = binning
    visualizeMosaicExpTask = VisualizeMosaicExpTask(config=visualizeMosaicExpConfig)
    mosaic = visualizeMosaicExpTask.run(inputExps=exposures_binned, camera=camera).outputData
    backend='matplotlib'
    visit_cmap = "gray"
    scale = "asinh"
    
    
    fig = plt.figure(figsize=(16,16))
    # disp = afwDisplay.Display(1, backend, cmap=visit_cmap)
    # norm=LogNorm(vmin=2200)
    norm = ImageNormalize(stretch=LogStretch())
    plt.imshow(mosaic.array,origin='lower',norm=norm)
    plt.colorbar()
    plt.xlim(6850*8/binning,)
    plt.ylim(5250*8/binning,6380*8/binning)

    plt.show()
    break
    # print("Delta coord:  RA    ", ra_cen-so_deg(wcs.getSkyOrigin()[0]) ,' deg    DEC   ', dec_cen-so_deg(wcs.getSkyOrigin()[1]),"  deg")

    # print(f"Delta coord:  RA    {3600*(float(np.abs(hold.getSkyOrigin()[0]-wcs.getSkyOrigin()[0]))*u.rad).to(u.deg).value:.2f}  arcsec  \
                          # DEC   {3600*(float(np.abs(hold.getSkyOrigin()[1]-wcs.getSkyOrigin()[1]))*u.rad).to(u.deg).value:.2f}  arcsec")
    hold=wcs
    # print(wcs)
 #   fig=splot.plot(dataImgs[0], stretch='asinh',percentile=98,cmap='plasma')
#     plt.title(f"visit {visit}")
    
#     plt.show()
#     fig.savefig(f"angel_wings{visit}.png", dpi=250, bbox_inches='tight')
#     frames.append(PIL.Image.open(f"angel_wings{visit}.png"))


# frames[0].save('angel_animation.gif', save_all=True, append_images=frames, duration=500, loop=0)
