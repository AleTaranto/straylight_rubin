from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings

warnings.filterwarnings(
    "ignore",
    category=AstropyDeprecationWarning,
)
from astroquery.vizier import Vizier
import numpy as np
import pathlib
from astropy.time import Time
from skyfield.api import load, Loader

# Anchored to this module's location, not the CWD: geom_utils is imported from
# several topic directories, so a relative path would resolve differently in each.
# Note skyfield's load() does NOT accept absolute paths (it joins the filename
# onto its loader directory), hence a Loader pointed at data/ephemeris/ and the
# bare filename below. A missing kernel would otherwise be silently re-downloaded
# from JPL (32 MB), so check first and fail loudly.
BSP_NAME = "de440s.bsp"
BSP_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "ephemeris"
BSP_PATH = BSP_DIR / BSP_NAME
_eph_load = Loader(str(BSP_DIR), verbose=False)
from astropy.coordinates import EarthLocation, SkyCoord, AltAz 
from astropy import units as u
import io
from time import sleep, time
from contextlib import redirect_stdout, redirect_stderr


def get_obj_position(mjd,obj):
    
    # Input time in MJD
    time_astropy = Time(mjd, format='mjd')
    ts = load.timescale()
    time = ts.tdb(jd=time_astropy.jd)

    # Load ephemeris
    if not BSP_PATH.exists():
        raise FileNotFoundError(f"ephemeris kernel missing: {BSP_PATH}")

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        eph = _eph_load(BSP_NAME)

    # Observer (optional: for topocentric correction)
    earth = eph['earth']
    object = eph[obj]

    # Get position of the Moon from Earth's center (geocentric)
    moon_position = earth.at(time).observe(object).apparent()
    ra_moon, dec_moon, _ = moon_position.radec()

    ra_deg = ra_moon.hours * 15  # Convert hours to degrees
    dec_deg = dec_moon.degrees  # Already in degrees

    # Moon's position as SkyCoord
    moon_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')

    return moon_coord

def convert_radec_to_altaz(loc, t, radecCoord):
    if not isinstance(t, Time):
        t = Time(t)
    
    loc = EarthLocation.of_site(loc)
    altaz = AltAz(location=loc, obstime=t)
    radecCoord = SkyCoord(radecCoord)
    return radecCoord.transform_to(altaz)
    
def find_bright_stars(center, max_dist, sorting='Vmag', mag_filter_dict = {"Vmag": "<4"},limit=5):

    vizier = Vizier(columns=["HIP", "Vmag", "_RA.icrs", "_DE.icrs"], column_filters=mag_filter_dict)    
    vizier.ROW_LIMIT = -1
    vizier.TIMEOUT = 300

    # Query sul catalogo Hipparcos piu leggero tanto mi interessano solo le stelle piu brillanti che ci sono anche qua
    for attempt in range(3):
        try:
            result = vizier.query_region(
                center,
                radius=max_dist * u.deg,
                catalog="I/239/hip_main",
                cache=False
            )
            break

        except Exception as e:
            if attempt == 2:
                raise

            sleep(2 ** attempt)

    if len(result) == 0:
        return None

    stars = result[0][['HIP', 'Vmag', '_RA.icrs', '_DE.icrs']]
    starCoord = SkyCoord(ra=stars['_RA.icrs'], dec=stars['_DE.icrs'], unit=(u.deg, u.deg), frame='icrs')

    radial_distance= starCoord.separation(center).deg
    stars['Radial_Distance'] = radial_distance
    # stars=stars[stars['Radial_Distance']>15]
    stars.sort(keys=sorting)

    # if len(stars) > 3:
    #     # print(f"Found {len(result[0])} stars in the specified region. Limiting to 5 brightest stars.")
    #     stars = stars[:limit]
    
    return stars[:limit+1] #[stars['Radial_Distance']>19]
    
def get_pos_ang(target, center, loc, tobs):
    # Convert to AltAz
    target_altaz = convert_radec_to_altaz(loc, tobs, target)
    center_altaz = convert_radec_to_altaz(loc, tobs, center)

    # Compute angular separation (great-circle distance)
    sep = target_altaz.separation(center_altaz)

    # Compute position angle from center to target, measured East of North (on the sphere)
    pa = center_altaz.position_angle(target_altaz)

    # Return both separation and position angle (in degrees)
    # For hour angle-like offset, you may want to return pa.to(u.deg).value
    return pa
    # return pa.to(u.deg).value
    