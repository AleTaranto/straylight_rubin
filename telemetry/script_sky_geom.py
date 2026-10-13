import lsst.daf.butler as dafButler
from lsst.summit.utils.utils import getCameraFromInstrumentName

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord 
from astropy.time import Time, TimeDelta
import sys, pathlib

# --- project bootstrap: common/ holds the shared modules ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "common"))
# -----------------------------------------------------------
import geom_utils as gu

# import lsst_efd_client

butler = dafButler.Butler("embargo", collections=["LSSTCam/runs/nightlyValidation","LSSTCam/raw/all"], instrument="LSSTCam")
# butler = dafButler.Butler("main", collections=["u/ataranto/general_outputs"], instrument="LSSTCam")
camera = getCameraFromInstrumentName("LSSTCam")
# Get the EFD client
# client = lsst_efd_client.EfdClient("usdf_efd")

visit_list = []
for arg in sys.argv[1:]:
    try:
        visit = int(arg)
        if len(str(visit)) == 13:
            visit_list.append(visit)
    except ValueError:
        continue

if not visit_list:  # If no valid visits provided, use default
    raise ValueError("No valid visit numbers provided. Please provide visit numbers as command-line arguments.")

max_dist = 8* u.deg
for i,visit in enumerate(visit_list):
    print(f"Processing visit {visit}...")
        # visit info
    t = butler.get("raw.visitInfo", exposure=visit, detector=94).date.toAstropy()
    ra = butler.get("raw.visitInfo", exposure=visit, detector=94).boresightRaDec.getRa().asDegrees()
    dec = butler.get("raw.visitInfo", exposure=visit, detector=94).boresightRaDec.getDec().asDegrees()
    info=butler.get('raw.visitInfo', detector=94, exposure=visit)  #change exposure to visit for raw or post_isr
    st_time=str(info.date).split('"')[1]
    time_start = Time(st_time, scale="utc", format="isot")
    time_end   = time_start+TimeDelta(30, format='sec')
#     rot_angle=(await client.select_time_series(
#         "lsst.sal.MTRotator.rotation",
#         fields=['actualPosition'],
#         start=time_start,
#         end=time_end
# ))        
#     rot=0.5*(float(rot_angle.max().iloc[0])+float(rot_angle.min().iloc[0]))

    center = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    # bright stars
    try:
        brightest_stars = gu.find_bright_stars(center, max_dist, sorting='Vmag' ,mag_filter_dict={"Vmag": "<8"},limit=10)  #decide the ordering of the star, default is decrescent mag
    except Exception as e:
        print(f"Error finding bright stars for visit {visit}: {e}")
        brightest_stars = None
        
    if brightest_stars is not None:
        for i, star in enumerate(brightest_stars):
            starCoord = SkyCoord(ra=star['_RA.icrs'], dec=star['_DE.icrs'], unit=(u.deg, u.deg), frame='icrs')
            staraz = gu.convert_radec_to_altaz("rubin", t, starCoord)
            starDistance = star['Radial_Distance']
            starVMag = star['Vmag']
            star_ha = gu.get_pos_ang(starCoord, center, "rubin", st_time)    
            # print(f"\t{i+1} - HIP {int(star['HIP'])}, Vmag: {starVMag:.2f}, at {starDistance:.2f}°, Clocking angle: {star_ha:.2f}°")

            print(f"\t{i+1} - HIP {int(star['HIP'])}, Vmag: {starVMag:.2f}, at {starDistance:.2f}°,angle x-y: {starDistance*np.sin(star_ha):.2f}° / {starDistance*np.cos(star_ha):.2f}°")
            
        # retrieve moon position
        sat = gu.get_obj_position(t.mjd,'SATURN BARYCENTER')
        satDistance = center.separation(sat).deg
        sat_ha = gu.get_pos_ang(sat, center, "rubin", t)
        jup = gu.get_obj_position(t.mjd,'JUPITER BARYCENTER')
        jupDistance = center.separation(jup).deg
        jup_ha = gu.get_pos_ang(jup, center, "rubin", t)
        moon = gu.get_obj_position(t.mjd,'moon')
        moonDistance = center.separation(moon).deg
        moon_ha = gu.get_pos_ang(moon, center, "rubin", t)
        print(" ")
        print(f"MOON    distance: {moonDistance:.2f}°, clocking angle: {moon_ha:.2f}°")
        print(f"SATURN  distance: {satDistance :.2f}°, clocking angle: {sat_ha :.2f}°")
        print(f"JUPITER distance: {jupDistance :.2f}°, clocking angle: {jup_ha :.2f}°") 
       