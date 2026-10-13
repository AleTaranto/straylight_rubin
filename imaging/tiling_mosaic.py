import numpy as np
import matplotlib.pyplot as plt
import time
import gc
import os
from mpi4py import MPI

from lsst.daf.butler import Butler
import lsst.geom as geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.afw.geom import makeSkyWcs
import lsst.summit.utils.plotting as splot

# -------------------------
# Helper: Tiling Logic
# -------------------------
def get_tiles(full_bbox, tile_size=4000):
    tiles = []
    w_total = full_bbox.getWidth()
    h_total = full_bbox.getHeight()
    for y in range(0, h_total, tile_size):
        for x in range(0, w_total, tile_size):
            w = min(tile_size, w_total - x)
            h = min(tile_size, h_total - y)
            tiles.append(geom.Box2I(geom.Point2I(x, y), geom.Extent2I(w, h)))
    return tiles

# -------------------------
# Mosaic WCS Builder
# -------------------------
def make_mosaic_wcs(exps_meta, pixscale_arcsec=0.2):
    pixscale_deg = pixscale_arcsec / 3600.0
    sky_pts = []
    # Usiamo TUTTI i metadati passati per trovare i limiti, non solo i centri
    for wcs, bbox in exps_meta:
        for corner in bbox.getCorners():
            sky_pts.append(wcs.pixelToSky(geom.Point2D(corner)))

    ras = np.array([p.getRa().asDegrees() for p in sky_pts])
    decs = np.array([p.getDec().asDegrees() for p in sky_pts])
    
    # Definiamo i limiti estremi con un margine di sicurezza di 500 pixel
    margin_px = 7500
    ra_min, ra_max = np.min(ras), np.max(ras)
    dec_min, dec_max = np.min(decs), np.max(decs)
    
    # Centro del mosaico
    ra0, dec0 = (ra_min + ra_max)/2, (dec_min + dec_max)/2
    
    # Calcoliamo l'estensione necessaria in pixel
    width_px = int((ra_max - ra_min) / pixscale_deg) + (margin_px * 2)
    height_px = int((dec_max - dec_min) / pixscale_deg) + (margin_px * 2)
    
    # CRPIX deve essere impostato in modo che ra_min/dec_min 
    # finiscano dentro il box positivo (es. a coordinata 500, 500)
    # Quindi il centro ra0/dec0 sarà a metà della larghezza/altezza
    crpix = geom.Point2D(width_px / 2, height_px / 2)
    
    mosaic_wcs = makeSkyWcs(crpix, geom.SpherePoint(ra0*geom.degrees, dec0*geom.degrees), 
                            np.array([[pixscale_deg, 0.0], [0.0, pixscale_deg]]))
    
    mosaic_bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(width_px, height_px))
    return mosaic_wcs, mosaic_bbox

# -------------------------
# Main MPI PNG Tiler
# -------------------------
def mpi_png_tiler_rubin(repo, collections, query, pixscale=1.2, tile_size=4000, out_dir="../outputs/rubin_mosaic_tiles"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    t0 = time.time()

    if rank == 0 and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    butler = Butler(repo, collections=collections)
    
    if rank == 0:
        refs = list(butler.registry.queryDatasets("post_isr_image", where=query))
        print(f"[Rank 0] Found {len(refs)} datasets")
        visit_meta = []
        for r in refs[::max(1, len(refs)//20)]:
            exp = butler.get(r)
            visit_meta.append((exp.getWcs(), exp.getBBox()))
            del exp
        mosaic_wcs, mosaic_bbox = make_mosaic_wcs(visit_meta, pixscale_arcsec=pixscale)
        all_tiles = get_tiles(mosaic_bbox, tile_size=tile_size)
    else:
        refs = mosaic_wcs = mosaic_bbox = all_tiles = None

    refs = comm.bcast(refs, root=0)
    mosaic_wcs = comm.bcast(mosaic_wcs, root=0)
    mosaic_bbox = comm.bcast(mosaic_bbox, root=0)
    all_tiles = comm.bcast(all_tiles, root=0)

    # --- Pre-calcolo Bounding Boxes locali ---
    local_refs = np.array_split(refs, size)[rank]
    local_bboxes = []
    
    for r in local_refs:
        try:
            temp_exp = butler.get(r)
            w, b = temp_exp.getWcs(), temp_exp.getBBox()
            # Trasformiamo il bbox dell'esposizione nel sistema di pixel del mosaico
            pixels = [mosaic_wcs.skyToPixel(w.pixelToSky(geom.Point2D(c))) for c in b.getCorners()]
            # Creiamo un Box2D che racchiude l'immagine warppata
            env = geom.Box2D()
            for p in pixels:
                env.include(p)
            local_bboxes.append(env)
            del temp_exp
        except Exception as e:
            local_bboxes.append(None)
        gc.collect()

    # Loop sulle Tile
    for idx, t_bbox_i in enumerate(all_tiles):
        # Convertiamo il Box2I della tile in Box2D per il confronto
        t_bbox_d = geom.Box2D(t_bbox_i)
        
        l_sum = np.zeros((t_bbox_i.getHeight(), t_bbox_i.getWidth()), dtype=np.float32)
        l_wt = np.zeros_like(l_sum)

        for i, ref in enumerate(local_refs):
            # Se il Bounding Box dell'immagine non interseca quello della Tile, saltiamo
            if local_bboxes[i] is None or not t_bbox_d.overlaps(local_bboxes[i]): 
                continue
            
            try:
                exp = butler.get(ref)
                dest = afwImage.ImageF(t_bbox_i)
                dest.set(np.nan)
                afwMath.warpImage(dest, mosaic_wcs, exp.getImage(), exp.getWcs(), afwMath.WarpingControl("bilinear"))
                
                arr = dest.getArray()
                valid = np.isfinite(arr)
                l_sum[valid] += arr[valid]
                l_wt[valid] += 1.0
                del exp, dest
            except Exception:
                continue

        g_sum = np.zeros_like(l_sum) if rank == 0 else None
        g_wt = np.zeros_like(l_wt) if rank == 0 else None
        comm.Reduce(l_sum, g_sum, op=MPI.SUM, root=0)
        comm.Reduce(l_wt, g_wt, op=MPI.SUM, root=0)

        if rank == 0:
            mask = g_wt > 0
            if np.any(mask):
                g_sum[mask] /= g_wt[mask]
                g_sum[~mask] = np.nan
                
                tile_wcs = mosaic_wcs.copyAtShiftedPixelOrigin(
                    geom.Extent2D(t_bbox_i.getMinX(), t_bbox_i.getMinY())
                )

                fig = plt.figure(figsize=(25, 12))
                _ = splot.plot(
                    g_sum,
                    stretch="ccs",
                    cmap="inferno",
                    figure=fig,
                    title=f"Tile {idx} | Origin: {t_bbox_i.getMinX()},{t_bbox_i.getMinY()}"
                )

                outfile = os.path.join(out_dir, f"tile_{idx:04d}.png")
                plt.savefig(outfile, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"[Rank 0] Tile {idx} saved.")
            
            del g_sum, g_wt
        gc.collect()

if __name__ == "__main__":
    repo = "main"
    collections = ['u/elhoward/StrayLight/diffim']
    query = """
        exposure in (2025121200069..2025121200092, 2025121900041..2025121900080)
        AND detector in (1..19,21..26,28..64,66..121,124..160,162..167,169..187)
    """

    mpi_png_tiler_rubin(repo, collections, query, pixscale=1.0, tile_size=8000)