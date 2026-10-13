import sys
import warnings
import time
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.filterwarnings(
    "ignore",
    category=AstropyDeprecationWarning
)
import pathlib
# geom_utils now lives in common/; anchored to this file so it survives a move.
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent / "common"))
import geom_utils as gu
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(
        f"Starting MPI job with {size} ranks...",
        flush=True
    )
    
max_dist = 2.025
min_dist = max_dist - 0.03
limit = 7
input_file = (
    "./rubin_one_year_output/"
    "rubin_visits_ra_dec.csv"
)
output_file = (
    "./rubin_one_year_output/"
    "one_year_occurrance_horseshoe.txt"
)
# Numero di visite tra due aggiornamenti della progress bar
PROGRESS_EVERY = 10

# MPI tag utilizzato per i messaggi di progresso
PROGRESS_TAG = 100

# Numero massimo di tentativi per una query VizieR
MAX_ATTEMPTS = 3

df = pd.read_csv(input_file)
global_n = len(df)
my_visit = df.iloc[rank::size]

local_n = len(my_visit)


# ============================================================
# Progress bar
# ============================================================
if rank == 0:

    pbar = tqdm(
        total=global_n,
        desc="Overall progress",
        unit="visit",
        file=sys.stderr,
        disable=False,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    # Forza la visualizzazione immediata
    pbar.refresh()

    # Ultimo numero di visite ricevuto da ciascun rank
    last_count = [0] * size


def poll_progress():
    """
    Chiamata esclusivamente da rank 0.

    Riceve tutti i messaggi di progresso attualmente
    disponibili senza bloccarsi.
    """

    while comm.Iprobe(
        source=MPI.ANY_SOURCE,
        tag=PROGRESS_TAG
    ):
        status = MPI.Status()
        sender_count = comm.recv(
            source=MPI.ANY_SOURCE,
            tag=PROGRESS_TAG,
            status=status
        )
        sender_rank = status.Get_source()
        delta = sender_count - last_count[sender_rank]
        if delta > 0:
            pbar.update(delta)
            last_count[sender_rank] = sender_count


# ============================================================
# Local results
local_occurrences = []
pending_requests = []
# Main loop
# ============================================================

local_count = 0


for i, (global_i, ra, dec, st_time) in enumerate(
    my_visit[
        [
            "ra_deg",
            "dec_deg",
            "datetime_utc"
        ]
    ].itertuples(
        index=True,
        name=None
    )
):
    center = SkyCoord(
        ra=ra,
        dec=dec,
        unit=(u.deg, u.deg),
        frame="icrs"
    )

    brightest_stars = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            brightest_stars = gu.find_bright_stars(
                center,
                max_dist,
                sorting="Vmag",
                mag_filter_dict={"Vmag": "<3"},
                limit=limit
            )
            break
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                print(
                    f"[Rank {rank}] "
                    f"VizieR failed for index {global_i}: "
                    f"{e}",
                    flush=True
                )
                brightest_stars = None
            else:
                time.sleep(2 ** attempt)

    if (
        brightest_stars is not None
        and any(
            star > min_dist
            for star in brightest_stars["Radial_Distance"]
        )
    ):
        local_occurrences.append(
            int(global_i)
        )
    # --------------------------------------------------------
    # Update local progress
    local_count += 1
    
    # Communicate progress
    if local_count % PROGRESS_EVERY == 0:
        if rank == 0:
            # Rank 0 aggiorna direttamente il proprio contatore.
            delta = (
                local_count
                - last_count[0]
            )
            if delta > 0:
                pbar.update(delta)
                last_count[0] = local_count
            # Riceve eventuali messaggi già disponibili
            # dagli altri rank.
            poll_progress()
        else:
            # Gli altri rank inviano il contatore cumulativo.
            request = comm.isend(
                local_count,
                dest=0,
                tag=PROGRESS_TAG
            )
            pending_requests.append(request)
            if len(pending_requests) >= 32:
                pending_requests[:] = [
                    req
                    for req in pending_requests
                    if not req.Test()
                ]
# ============================================================
# Final progress message
# ============================================================
if rank != 0:
    # Invia anche l'eventuale resto che non era multiplo
    # di PROGRESS_EVERY.
    request = comm.isend(
        local_count,
        dest=0,
        tag=PROGRESS_TAG
    )
    pending_requests.append(request)
    for request in pending_requests:
        request.Wait()
else:
    # Aggiorniamo la parte finale di rank 0.
    delta = (
        local_count
        - last_count[0]
    )
    if delta > 0:
        pbar.update(delta)
        last_count[0] = local_count
    # --------------------------------------------------------
    # Wait for all other ranks
    # --------------------------------------------------------
    while sum(last_count) < global_n:
        status = MPI.Status()
        sender_count = comm.recv(
            source=MPI.ANY_SOURCE,
            tag=PROGRESS_TAG,
            status=status
        )
        sender_rank = status.Get_source()
        delta = (
            sender_count
            - last_count[sender_rank]
        )
        if delta > 0:
            pbar.update(delta)
            last_count[sender_rank] = sender_count
    pbar.close()
# ============================================================
# Gather results
# ============================================================

comm.Barrier()
all_occurrences = comm.gather(
    local_occurrences,
    root=0
)
# ============================================================
# Write output
# ============================================================
if rank == 0:
    occurrences = sorted(
        idx
        for rank_occurrences in all_occurrences
        for idx in rank_occurrences
    )
    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for idx in occurrences:

            f.write(
                f"{idx},\n"
            )


    print(
        f"Done. "
        f"{len(occurrences)} occurrences written to "
        f"{output_file}",
        flush=True
    )