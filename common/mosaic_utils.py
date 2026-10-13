import numpy as np

import lsst.daf.butler as dafButler
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

def bin_image(image, camera, binning=8):
    visualizeBinExpConfig = VisualizeBinExpConfig()
    visualizeBinExpConfig.binning = binning
    visualizeBinExpTask = VisualizeBinExpTask(config=visualizeBinExpConfig)
    return visualizeBinExpTask.run(inputExp=image, camera=camera).outputExp

def get_stacked_image(
        butler, 
        visit_list, 
        img_type,    #post isr or calexp or raw
        instrument="LSSTCam", 
        where_clause=None, 
        binning=8,
        starless=False, 
        ):
    
    camera = getCameraFromInstrumentName(instrument)
    binning = binning

    visualizeMosaicExpConfig = VisualizeMosaicExpConfig()
    visualizeMosaicExpConfig.binning = binning
    visualizeMosaicExpTask = VisualizeMosaicExpTask(config=visualizeMosaicExpConfig)

    stack = None

    for visit in visit_list:
        if where_clause is None:
            dataset_refs = butler.query_datasets(img_type, where=f"visit={visit}")

        else:
            dataset_refs = butler.query_datasets(img_type, where=f"visit={visit} " + where_clause)

        exposures_binned = {
            camera.get(dr.dataId["detector"]).getName(): bin_image(butler.get(dr), camera, binning)
            for dr in dataset_refs
            }

        if starless:
            for detName in exposures_binned:
                
                exposures_binned[detName].image.array = doStarless(exposures_binned[detName].image.array)
        
        mosaic = visualizeMosaicExpTask.run(inputExps=exposures_binned.values(), camera=camera).outputData.array
        
        if stack is None:
            stack = mosaic
        else:
            stack = stack + mosaic

    return stack

def doStarless(img):
    kernel = make_2dgaussian_kernel(1, size=1)   
    threshold = detect_threshold(img, 2.)
    segment_map = detect_sources(img, threshold, npixels=1)
    footprint = circular_footprint(radius=1)
    dilateMask = segment_map.make_source_mask(footprint=footprint)
    dilateAndSmoothMask = convolve(dilateMask, kernel)

    starlessImg = np.where(dilateAndSmoothMask != 0, np.median(img), img)
    return starlessImg