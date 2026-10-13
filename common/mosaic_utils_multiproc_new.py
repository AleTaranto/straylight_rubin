import numpy as np
import os
import shutil
import pathlib

from lsst.pipe.tasks.visualizeVisit import (
    VisualizeBinExpConfig,
    VisualizeBinExpTask,
    VisualizeMosaicExpConfig,
    VisualizeMosaicExpTask,
)
from lsst.summit.utils.utils import getCameraFromInstrumentName
import multiprocessing as mp
from mpi4py import MPI

def bin_image(image, instrument, binning):
    camera = getCameraFromInstrumentName(instrument)
    visualizeBinExpConfig = VisualizeBinExpConfig()
    visualizeBinExpConfig.binning = binning
    visualizeBinExpTask = VisualizeBinExpTask(config=visualizeBinExpConfig)
    return visualizeBinExpTask.run(inputExp=image, camera=camera).outputExp

def get_binned_visit(butler, drefs, instrument, binning=8):

    nproc = len(drefs) if len(drefs) < mp.cpu_count() - 1 else mp.cpu_count() - 1
    
    camera = getCameraFromInstrumentName(instrument)
    visualizeMosaicExpConfig = VisualizeMosaicExpConfig()
    visualizeMosaicExpConfig.binning = binning
    visualizeMosaicExpTask = VisualizeMosaicExpTask(config=visualizeMosaicExpConfig)
    
    with mp.Pool(nproc) as pool:
        # Use multiprocessing to bin images in parallel
        exposures = pool.imap_unordered(butler.get, drefs)
        exposures = list(exposures)
    
    with mp.Pool(nproc) as pool:
        # Bin the images in parallel
        exp_chunk = [(exp, instrument, binning) for exp in exposures]
        exposures_binned = pool.starmap(bin_image, exp_chunk)
        exposures_binned = list(exposures_binned)
    
    return visualizeMosaicExpTask.run(inputExps=exposures_binned, camera=camera)
    
def plain_stack(butler, instrument, binning, visits_drefs, method="median"):
    
    stack=None
    for drefs in visits_drefs:
        
        mosaic = get_binned_visit(butler, drefs, instrument, binning=binning).outputData.array
        
        # first iter just do nothing
        if stack is None:
            stack = mosaic
            continue
        
        # then
        if method == "median":
            stack = np.nanmedian(np.stack((stack, mosaic)), axis=0)
        elif method == "mean":
            stack = np.nanmean(np.stack((stack, mosaic)), axis=0)
        elif method == "sum":
            stack = np.nansum(np.stack((stack, mosaic)), axis=0)

    return stack

def do_dark_subtraction(light, dark):
    return light - dark

def do_flat_division(light, flat):
    print("Doing flat division")
    return light / flat

def do_calibration(light, flat=None, dark=None):
    
    if flat is not None:
        light = do_flat_division(light, flat)

    if dark is not None:
        light = do_dark_subtraction(light, dark)

    return light

def clean_lscratch(clean=True):
    if not clean:
        return
    username = os.environ.get("USER", "unknown")
    scratch_dir = pathlib.Path(f"/lscratch/{username}")
    if not scratch_dir.exists():
        return
    try:
        for item in scratch_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    except Exception as e:
        print(f"Warning: could not fully clean {scratch_dir}: {e}")

def get_stacked_image(
    butler,
    butler_fl,
    visit_list,
    img_type,
    instrument="LSSTCam",
    where_clause=None,
    binning=8,
    method="median",
    calibrate=False,
    master_dark_visits=None,
    master_flat_visits=None,
    clean_scratch=False,
):

    import gc  # local import to avoid changing module-level imports

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    visualizeMosaicExpConfig = VisualizeMosaicExpConfig()
    visualizeMosaicExpConfig.binning = binning
    visualizeMosaicExpTask = VisualizeMosaicExpTask(config=visualizeMosaicExpConfig)

    visit_name = "exposure" if img_type in ["raw", "post_isr_image"] else "visit"
    camera = getCameraFromInstrumentName(instrument)

    # Build masters only on rank 0 and broadcast to all ranks
    master_dark = None
    master_flat = None
    if calibrate and rank == 0:
        # dark
        if isinstance(master_dark_visits, list) and len(master_dark_visits) == 1:
            if where_clause is None:
                dark_drefs = butler.query_datasets("post_isr_image", where=f"exposure={master_dark_visits[0]}")
            else:
                dark_drefs = butler.query_datasets("post_isr_image", where=f"exposure={master_dark_visits[0]} " + "and " + where_clause)
            master_dark = get_binned_visit(butler, dark_drefs, instrument, binning).outputData.array
            # drop heavy intermediate
            del dark_drefs
            gc.collect()
        elif isinstance(master_dark_visits, list) and len(master_dark_visits) > 1:
            if where_clause is None:
                dark_visits_drefs = [butler.query_datasets("post_isr_image", where=f"exposure={visit}") for visit in master_dark_visits]
            else:
                dark_visits_drefs = [butler.query_datasets("post_isr_image", where=f"exposure={visit} " + "and " + where_clause) for visit in master_dark_visits]
            master_dark = plain_stack(butler, instrument, binning, dark_visits_drefs)
            del dark_visits_drefs
            gc.collect()

        # flat
        if isinstance(master_flat_visits, list) and len(master_flat_visits) == 1:
            if where_clause is None:
                flat_drefs = butler_fl.query_datasets("flat", where=f"exposure={master_flat_visits[0]}")
            else:
                flat_drefs = butler_fl.query_datasets("flat", where=f"exposure={master_flat_visits[0]} " + "and " + where_clause)
            master_flat = get_binned_visit(butler_fl, flat_drefs, instrument, binning).outputData.array
            del flat_drefs
            gc.collect()
        elif isinstance(master_flat_visits, list) and len(master_flat_visits) > 1:
            if where_clause is None:
                flat_visits_drefs = [butler_fl.query_datasets("flat", where=f"exposure={visit}") for visit in master_flat_visits]
            else:
                flat_visits_drefs = [butler_fl.query_datasets("flat", where=f"exposure={visit} " + "and " + where_clause) for visit in master_flat_visits]
            master_flat = plain_stack(butler_fl, instrument, binning, flat_visits_drefs)
            del flat_visits_drefs
            gc.collect()

    # broadcast masters to all ranks (None if not computed)
    if calibrate:
        master_dark = comm.bcast(master_dark, root=0)
        master_flat = comm.bcast(master_flat, root=0)

    # split visits among MPI ranks (round-robin)
    assigned_visits = [v for i, v in enumerate(visit_list) if (i % size) == rank]

    local_mosaics = []
    for visit in assigned_visits:
        clean_lscratch(clean=clean_scratch)
        try:
            if where_clause is None:
                drefs = butler.query_datasets(img_type, where=f"{visit_name}={visit}")
            else:
                drefs = butler.query_datasets(img_type, where=f"{visit_name}={visit} " + "and " + where_clause)
        except Exception as e:
            print(f"[rank {rank}] Error querying datasets for visit {visit}: {e}")
            continue

        try:
            mosaic = get_binned_visit(butler, drefs, instrument, binning=binning).outputData.array
        except Exception as e:
            print(f"[rank {rank}] Error getting binned visit for visit {visit}: {e}")
            # free query variable before continue
            try:
                del drefs
            except Exception:
                pass
            gc.collect()
            continue

        # drop drefs ASAP to free memory
        try:
            del drefs
        except Exception:
            pass
        gc.collect()

        # calibrate if requested
        if calibrate and master_dark is not None:
            mosaic = do_calibration(mosaic, dark=master_dark)
        if calibrate and master_flat is not None:
            mosaic = do_calibration(mosaic, flat=master_flat)

        local_mosaics.append(mosaic)
        clean_lscratch(clean=clean_scratch)

    # free masters on all ranks once local processing done to reduce memory footprint
    if calibrate:
        try:
            del master_dark
        except Exception:
            pass
        try:
            del master_flat
        except Exception:
            pass
        gc.collect()

    # gather lists of mosaics to root (pickle-based gather)
    all_local = comm.gather(local_mosaics, root=0)

    # local copies no longer needed
    try:
        del local_mosaics
    except Exception:
        pass
    gc.collect()

    final_stack = None
    if rank == 0:
        # flatten list of lists
        all_mosaics = []
        for lst in all_local:
            if lst:
                all_mosaics.extend(lst)

        # free the gathered nested lists to free memory before stacking
        try:
            del all_local
        except Exception:
            pass
        gc.collect()

        if len(all_mosaics) == 0:
            return None

        # build array and reduce along axis 0
        arr = np.stack(all_mosaics, axis=0)  # shape (N, H, W)

        # free the all_mosaics list (arrays are now in arr)
        try:
            del all_mosaics
        except Exception:
            pass
        gc.collect()

        if method == "median":
            final_stack = np.nanmedian(arr, axis=0)
        elif method == "mean":
            final_stack = np.nanmean(arr, axis=0)
        elif method == "sum":
            final_stack = np.nansum(arr, axis=0)
        else:
            # fallback to median if unknown method
            final_stack = np.nanmedian(arr, axis=0)

        # free big temporary 'arr' after reduction
        try:
            del arr
        except Exception:
            pass
        gc.collect()

    # return stacked image on rank 0, None on other ranks
    return final_stack
