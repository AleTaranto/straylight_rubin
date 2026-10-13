# mpi_mosaic_minram.py (difference_image_predetection version)

import numpy as np
import matplotlib.pyplot as plt
import time
import gc
from mpi4py import MPI

from lsst.daf.butler import Butler
import lsst.afw.cameraGeom as cameraGeom
import lsst.geom as geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.afw.geom import makeSkyWcs
import lsst.summit.utils.plotting as splot

import resource
from memory_profiler import memory_usage


# -------------------------
# Minimal metadata-only loader
# -------------------------
def load_metadata(butler, ref):
    exp = butler.get(ref)
    return exp.getWcs(), exp.getBBox()


# -------------------------
# Build mosaic WCS from corners
# -------------------------
def make_mosaic_wcs(exps_meta, pixscale_arcsec=0.2):
    gc.collect()
    pixscale_deg = pixscale_arcsec / 3600.0
    sky_pts = []

    for wcs, bbox in exps_meta:
        corners = [
            (bbox.getMinX(), bbox.getMinY()),
            (bbox.getMinX(), bbox.getMaxY()),
            (bbox.getMaxX(), bbox.getMaxY()),
            (bbox.getMaxX(), bbox.getMinY()),
        ]
        for x, y in corners:
            sky_pts.append(wcs.pixelToSky(geom.Point2D(x, y)))

    ras = np.array([p.getRa().asDegrees() for p in sky_pts])
    decs = np.array([p.getDec().asDegrees() for p in sky_pts])

    ra0 = ras.mean()
    dec0 = decs.mean()
    center = geom.SpherePoint(ra0 * geom.degrees, dec0 * geom.degrees)

    xs = (ras - ra0) / pixscale_deg
    ys = (decs - dec0) / pixscale_deg

    xmin, xmax = int(xs.min()) - 50, int(xs.max()) + 50
    ymin, ymax = int(ys.min()) - 50, int(ys.max()) + 50

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    crpix = geom.Point2D(-xmin, -ymin)
    cd = np.array([[pixscale_deg, 0.0], [0.0, pixscale_deg]])

    mosaic_wcs = makeSkyWcs(crpix, center, cd)
    mosaic_bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(width, height))

    return mosaic_wcs, mosaic_bbox


# -------------------------
# Warp one exposure
# -------------------------
def warp_into_mosaic_minimal(butler, ref, mosaic_wcs, mosaic_bbox,
                              warping_kernel="bilinear", num_threads=1,
                              src_img=None, src_wcs=None):

    if src_img is None or src_wcs is None:
        exp = butler.get(ref)
        src_img = exp.getImage()
        src_wcs = exp.getWcs()

    dest_img = afwImage.ImageF(mosaic_bbox)
    dest_img.set(float("nan"))

    ctrl = afwMath.WarpingControl(warping_kernel)

    afwMath.warpImage(dest_img, mosaic_wcs, src_img, src_wcs, ctrl)

    arr = dest_img.getArray().astype("float32", copy=False)

    del dest_img
    gc.collect()

    return arr


# -------------------------
# MPI mosaic driver
# -------------------------
def mpi_make_and_plot_mosaic_minram(repo, collections, query,
                                    pixscale=0.2,
                                    combine="mean",
                                    warping_kernel="bilinear",
                                    num_threads=1,
                                    show_plot=True,
                                    outfile="../outputs/figures/mosaic_diff.png"):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    butler = Butler(repo, collections=collections)
    registry = butler.registry

    # ---- Query DIFFERENCE images ----
    if rank == 0:
        refs = list(registry.queryDatasets(
            "post_isr_image",
            where=query
        ))
        print(f"[rank 0] found {len(refs)} detector datasets")
    else:
        refs = None

    refs = comm.bcast(refs, root=0)
    if len(refs) == 0:
        raise RuntimeError("No valid image found")

    refs_per_rank = np.array_split(np.array(refs, dtype=object), size)
    local_refs = list(refs_per_rank[rank])

    # ---- Build visit-level WCS on rank 0 ----
    if rank == 0:
        visit_ids = sorted(set(r.dataId.get("visit") for r in refs))
        visit_meta = []
        dets=[1,23,28,68,124,158,169,187]
        # find detector 94 for this visit
        rep = None
        for r in refs:
            if r.dataId.get("detector") in dets:
                rep = r
                
                exp = butler.get(rep)
                visit_meta.append((exp.getWcs(), exp.getBBox()))
                del exp
        if len(visit_meta) == 0:
            raise RuntimeError("No valid detector 94 exposures found to build mosaic WCS")
    
        mosaic_wcs, mosaic_bbox = make_mosaic_wcs(visit_meta, pixscale_arcsec=pixscale)
    
        h, w = mosaic_bbox.getHeight(), mosaic_bbox.getWidth()
        print(f"[rank 0] mosaic size = {h} x {w}")
    
        del visit_meta
        gc.collect()
    else:
        mosaic_wcs = None
        mosaic_bbox = None
        h = w = None

    mosaic_wcs = comm.bcast(mosaic_wcs, root=0)
    mosaic_bbox = comm.bcast(mosaic_bbox, root=0)
    h = comm.bcast(h, root=0)
    w = comm.bcast(w, root=0)

    local_sum = np.zeros((h, w), dtype=np.float32)
    local_weight = np.zeros((h, w), dtype=np.float32)

    # ---- Loop local refs ----
    for i, ref in enumerate(local_refs):
        exp = butler.get(ref)
        src_img = exp.getImage()
        src_wcs = exp.getWcs()

        arr = warp_into_mosaic_minimal(
            butler, ref, mosaic_wcs, mosaic_bbox,
            warping_kernel=warping_kernel,
            num_threads=num_threads,
            src_img=src_img, src_wcs=src_wcs
        )

        # np.nan_to_num(arr, copy=False, nan=0.0)

        # local_sum += arr
        # local_weight += (arr != 0.0).astype(np.float32)
        valid = np.isfinite(arr)

        local_sum[valid] += arr[valid]
        local_weight[valid] += 1.0

        del src_img, src_wcs, arr
        gc.collect()
        elapsed = time.time() - t0
        if (i + 1) % 20 == 0:
            print(f"[rank {rank}] processed {i+1}/{len(local_refs)} in {elapsed} s")

    # ---- Reduce ----
    if rank == 0:
        global_sum = np.zeros_like(local_sum)
        global_weight = np.zeros_like(local_weight)
    else:
        global_sum = None
        global_weight = None

    comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)
    comm.Reduce(local_weight, global_weight, op=MPI.SUM, root=0)

    # ---- Combine + Plot ----
    if rank == 0:
        mask = global_weight > 0
        global_sum[mask] /= global_weight[mask]
        global_sum[~mask] = np.nan
   

        elapsed = time.time() - t0
        print(f"[rank 0] Combined in {elapsed:.1f} s")

        if show_plot:
            fig = plt.figure(figsize=(25, 8))

            # Use official Rubin CCS stretch
            _ = splot.plot(
                global_sum,
                stretch="ccs",
                cmap="inferno",
                figure=fig
            )

            plt.title("Image Mosaic")
            plt.savefig(outfile, dpi=300)
            plt.close(fig)
        return global_sum, mosaic_wcs, mosaic_bbox
\
    return None, None, None


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    import logging

    logging.getLogger("lsst.summit.utils.utils").setLevel(logging.ERROR)
    logging.getLogger("lsst.summit.utils.plotting").setLevel(logging.ERROR)
    t0 = time.time()

    repo = "main"
    collections=['u/elhoward/StrayLight/diffim']
    # collections = ["u/ataranto/lsst_outputs"]
    query = """
        exposure in (2025121200069..2025121200092)
        AND detector in (1..19,21..26,28..64,66..121,124..160,162..167,169..187)
    """

    mosaic, wcs, bbox = mpi_make_and_plot_mosaic_minram(
        repo,
        collections,
        query,
        pixscale=1.6,
        warping_kernel="bilinear",
        num_threads=1,
        show_plot=True,
        outfile="../outputs/figures/tape_mosaic.png"
    )

    # elapsed = time.time() - t0
    # print(f"Combined in {elapsed:.1f} s")

