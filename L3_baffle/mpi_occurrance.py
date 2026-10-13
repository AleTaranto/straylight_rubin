import sys, pathlib
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


max_dist = 2.05 
min_dist = max_dist-0.065 
limit=7

df = pd.read_csv("./rubin_one_year_output/rubin_visits_ra_dec.csv")
my_visit = df[rank::comm.Get_size()]
ra_arr = my_visit['ra_deg']
dec_arr = my_visit['dec_deg']
st_time_arr = my_visit['datetime_utc']
count = 0

with open(f"./rubin_one_year_output/one_year_occurrance_horseshoe.txt", "a", encoding="utf-8") as f:
    for i, (ra, dec, st_time) in tqdm(enumerate(zip(ra_arr, dec_arr, st_time_arr)),total=len(ra_arr), position=rank):
        center = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
        brightest_stars = gu.find_bright_stars(center, max_dist, sorting='Vmag', mag_filter_dict={"Vmag": "<3"}, limit=limit)
        # if brightest_stars is not None:  print(brightest_stars["Radial_Distance"][0])
        if brightest_stars is not None and any(star > min_dist for star in brightest_stars["Radial_Distance"]):            
            f.write(f" {i+rank*int(len(df)/comm.Get_size())}, \n ")
        #     count += 1
        # if count>1 and count % 25 == 0: print(f"partial occurrences: {count}")
