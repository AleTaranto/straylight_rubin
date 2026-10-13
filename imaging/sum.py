# Clean, unified pixelwise mosaic: combine = "sum" | "mean" | "median"

from mpi4py import MPI
import numpy as np
import gc
import time
import matplotlib.pyplot as plt
from lsst.daf.butler import Butler


def mpi_sum_images_pixelwise(repo, collections, query,
                             combine="mean",
                             num_threads=1,
                             show_plot=True,
                             outfile="../outputs/figures/mosaic_pixel_sum.png"):
    """
    Build a pixel-aligned mosaic without warping.
    combine = "sum" (raw sum), "mean", or "median".
    All images must share the same pixel grid.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    t0 = time.time()

    butler = Butler(repo, collections=collections)
    registry = butler.registry

    # --- Query exposures ---
    if rank == 0:
        refs = list(registry.queryDatasets("post_isr_image", where=query))
        print(f"[rank 0] found {len(refs)} refs")
    else:
        refs = None

    refs = comm.bcast(refs, root=0)
    if refs is None or len(refs) == 0:
        if rank == 0:
            raise RuntimeError("No exposures found for query")
        return None

    # --- Split references across ranks ---
    refs_per_rank = np.array_split(np.array(refs, dtype=object), size)
    local_refs = list(refs_per_rank[rank])

    # --- Determine image shape ---
    if rank == 0:
        exp0 = butler.get(refs[0])
        h = exp0.getBBox().getHeight()
        w = exp0.getBBox().getWidth()
        del exp0
    else:
        h = w = None

    h = comm.bcast(h, root=0)
    w = comm.bcast(w, root=0)

    # --- Allocate local accumulators based on combine ---
    if combine == "median":
        local_stack = []
    elif combine == "sum":
        local_sum = np.zeros((h, w), dtype=np.float32)
    elif combine == "mean":
        local_sum = np.zeros((h, w), dtype=np.float32)
        local_weight = np.zeros((h, w), dtype=np.float32)
    else:
        raise ValueError(f"Unknown combine mode: {combine}")

    # --- Process local exposures ---
    for i, ref in enumerate(local_refs):
        img = butler.get(ref).getImage().getArray().astype(np.float32, copy=False)

        # --- Crop to common dimensions ---
        ih, iw = img.shape
        h0, w0 = local_sum.shape if combine in ["sum","mean"] else (h, w)
        h_crop = min(h0, ih)
        w_crop = min(w0, iw)
        img = img[:h_crop, :w_crop]
        if combine in ["sum", "mean"]:
            local_sum = local_sum[:h_crop, :w_crop]
            if combine == "mean":
                local_weight = local_weight[:h_crop, :w_crop]().astype(np.float32, copy=False)

        if combine == "median":
            # --- Crop for median mode ---
            ih, iw = img.shape
            h_crop = min(h, ih)
            w_crop = min(w, iw)
            img = img[:h_crop, :w_crop]
            local_stack.append(img)
        elif combine == "sum":
            local_sum += img
        elif combine == "mean":
            local_sum += img
            local_weight += 1.0

        del img
        gc.collect()

        if (i + 1) % 20 == 0 or (i + 1) == len(local_refs):
            print(f"[rank {rank}] processed {i+1}/{len(local_refs)} local refs")

    # --- MPI Reduction / Gather ---
    if combine == "median":
        # Gather lists to root
        if rank == 0:
            all_stacks = comm.gather(local_stack, root=0)
        else:
            comm.gather(local_stack, root=0)

    elif combine == "sum":
        if rank == 0:
            global_sum = np.zeros((h, w), dtype=np.float32)
        else:
            global_sum = None
        comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)

    elif combine == "mean":
        if rank == 0:
            global_sum = np.zeros((h, w), dtype=np.float32)
            global_weight = np.zeros((h, w), dtype=np.float32)
        else:
            global_sum = None
            global_weight = None
        comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)
        comm.Reduce(local_weight, global_weight, op=MPI.SUM, root=0)

    # --- Final assembly at root ---
    if rank == 0:
        if combine == "median":
            merged = []
            for stk in all_stacks:
                merged.extend(stk)
            stacked = np.stack(merged, axis=0)
            global_sum = np.nanmedian(stacked, axis=0).astype(np.float32)

        elif combine == "sum":
            # global_sum already contains raw sum
            pass

        elif combine == "mean":
            mask = global_weight > 0
            global_sum[mask] /= global_weight[mask]
            global_sum[~mask] = np.nan

        # Plot
        if show_plot:
            fig = plt.figure(figsize=(20, 20))
            plt.imshow(global_sum, origin="lower", cmap="gray",
                       vmin=np.nanpercentile(global_sum, 5),
                       vmax=np.nanpercentile(global_sum, 99))
            plt.colorbar()
            plt.title(f"Pixel Mosaic ({combine})")
            plt.savefig(outfile, dpi=300)
            plt.close(fig)

        elapsed = time.time() - t0
        print(f"[rank 0] Mosaic done in {elapsed:.1f} s")
        return global_sum

    return None



if __name__ == "__main__":
    repo = "embargo"
    collections = ["LSSTCam/runs/nightlyValidation"]

    query = """
        visit in (2025111900135..2025111900136)
        AND detector in (0..189)
    """

    # Run with MPI:
    #   mpiexec -n 4 python example_run.py

    # ---- Mean mosaic (similar to previous behavior) ----
    mosaic_mean = mpi_sum_images_pixelwise(
        repo=repo,
        collections=collections,
        query=query,
        combine="sum",               # <--- mean combine
        outfile="../outputs/figures/mosaic_mean.png",
    )

