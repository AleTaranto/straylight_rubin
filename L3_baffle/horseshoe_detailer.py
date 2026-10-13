import os
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree

from rubin_scheduler.scheduler.detailers import BaseDetailer
from rubin_scheduler.scheduler.utils import ObservationArray, wrap_ra_dec
from rubin_scheduler.utils import DEFAULT_NSIDE, gnomonic_project_tosky


class HorseshoeAvoidanceDetailer(BaseDetailer):
    """A Rubin Scheduler Detailer to avoid stray light reflections from the L3
    horseshoe baffle.

    When a bright star falls within a specific off-axis angular annulus
    ([1.995, 2.025] degrees by default) from the field of view center,
    reflections off the L3 lens baffle create a characteristic stray light
    artifact known as the "Horseshoe".

    This Detailer intercepts candidate observations during scheduling, checks if
    any bright star from a given catalog (e.g. Yale Bright Star Catalog, YBSC v5)
    falls into the forbidden annulus using an ultra-fast 3D Cartesian KD-tree.

    The magnitude check is evaluated in the specific filter/band of observation:
    - If observing in band R, it checks whether the star has R_mag <= 3.0.
    - If observing in band I, it checks whether the star has I_mag <= 3.0.
    - And similarly for u, g, z, y.

    If a star exceeds the brightness threshold in the observation's reference
    filter, the detailer applies a slight nudging offset (default +0.05 deg in
    RA and Dec) to move the pointing safely out of the artifact zone.

    Parameters
    ----------
    catalog_path : `str` or `Path`, optional
        Path to the FITS catalog containing stars with coordinates and
        magnitudes. Defaults to `rubin_one_year_output/ybsc_v5.fits`.
    mag_limit : `float` or `dict`, optional
        Limiting magnitude threshold. If a float is passed (default 3.0), the
        same limit is applied across all bands. If a dict is passed (e.g.
        `{'r': 3.0, 'i': 3.0, 'default': 3.0}`), per-band limits are applied.
    use_observation_band : `bool`, optional
        If True (default), evaluates star brightness using the star's magnitude
        in the reference filter of the observation (u, g, r, i, z, y).
        If False, uses the standard V-band magnitude.
    inner_radius_deg : `float`, optional
        Inner radius of the forbidden annulus in degrees. Default 1.995 deg.
    outer_radius_deg : `float`, optional
        Outer radius of the forbidden annulus in degrees. Default 2.025 deg.
    offset_ra_deg : `float`, optional
        Offset applied to RA in degrees when a star falls in the annulus.
        Default 0.05 deg.
    offset_dec_deg : `float`, optional
        Offset applied to Dec in degrees when a star falls in the annulus.
        Default 0.05 deg.
    offset_mode : `str`, optional
        Method used to apply the angular offset:
        - "gnomonic": Tangent plane displacement via `gnomonic_project_tosky`
          (consistent with `DitherDetailer`). Default.
        - "direct": Direct spherical coordinate addition + coordinate wrapping.
    annotate_note : `bool`, optional
        If True, appends `note_tag` to `observation_array['scheduler_note']`
        or `observation_array['note']` when an observation is shifted.
        Default True.
    note_tag : `str`, optional
        Substring tag added to the observation note. Default "hs_avoid".
    nside : `int`, optional
        HEALPix resolution parameter for the BaseDetailer inheritance.
        Controls spatial resolution for scheduler computations (default: DEFAULT_NSIDE).
        Must be a power of 2 (e.g., 32, 64, 128, 256).
        This parameter is passed from external scripts like simulate_one_year_with_detailer.py
        to support simulations at different resolution levels.
    vmag_limit : `float` or None, optional
        Deprecated alias for `mag_limit` kept for backward compatibility.
    """

    def __init__(
        self,
        catalog_path=None,
        mag_limit=3.0,
        use_observation_band=True,
        inner_radius_deg=1.995,
        outer_radius_deg=2.025,
        offset_ra_deg=0.05,
        offset_dec_deg=0.05,
        offset_mode="gnomonic",
        annotate_note=True,
        note_tag="hs_avoid",
        nside=DEFAULT_NSIDE,
        vmag_limit=None,
    ):
        super().__init__(nside=nside)
        self.nside = int(nside)

        if self.nside < 1 or (self.nside & (self.nside - 1)) != 0:
            raise ValueError(
                f"nside must be a positive power of 2, got {self.nside}. "
                f"Valid values: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ..."
            )

        if catalog_path is None:
            catalog_path = (
                Path(__file__).resolve().parent
                / "rubin_one_year_output"
                / "ybsc_v5.fits"
            )
        self.catalog_path = Path(catalog_path)

        # Handle backward compatibility with vmag_limit
        if vmag_limit is not None:
            self.mag_limit = vmag_limit
        else:
            self.mag_limit = mag_limit

        self.use_observation_band = bool(use_observation_band)
        self.inner_radius_deg = float(inner_radius_deg)
        self.outer_radius_deg = float(outer_radius_deg)
        self.offset_ra_deg = float(offset_ra_deg)
        self.offset_dec_deg = float(offset_dec_deg)
        self.offset_mode = offset_mode.lower()
        self.annotate_note = bool(annotate_note)
        self.note_tag = str(note_tag)

        # Precompute offsets in radians
        self.offset_ra_rad = float(np.radians(self.offset_ra_deg))
        self.offset_dec_rad = float(np.radians(self.offset_dec_deg))

        # Chord distances on the 3D unit sphere for the annulus:
        # chord distance d = 2 * sin(theta / 2)
        self.d_min = 2.0 * np.sin(np.radians(self.inner_radius_deg) / 2.0)
        self.d_max = 2.0 * np.sin(np.radians(self.outer_radius_deg) / 2.0)

        # Load star catalog, calculate filter band magnitudes, and build KD-tree
        self._load_catalog()

        # Cumulative execution counters and history
        self.n_evaluated = 0
        self.n_shifted = 0
        self.shifted_history = []
        self.n_shifted_by_band = {b: 0 for b in ["u", "g", "r", "i", "z", "y", "other"]}

    def get_mag_limit(self, band):
        """Retrieve limiting magnitude for a given filter band."""
        if isinstance(self.mag_limit, dict):
            b_key = str(band).lower()
            return float(self.mag_limit.get(b_key, self.mag_limit.get("default", 3.0)))
        return float(self.mag_limit)

    def _load_catalog(self):
        """Load star catalog, compute multi-band magnitudes, and build KD-tree."""
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Star catalog FITS file not found: {self.catalog_path}"
            )

        with fits.open(self.catalog_path) as hdul:
            data = hdul[1].data

            # Identify RA and Dec columns
            ra_candidates = ["coord_ra", "RA_DEG", "ra", "RA"]
            dec_candidates = ["coord_dec", "DEC_DEG", "dec", "DEC", "de"]

            ra_col = next((c for c in ra_candidates if c in data.names), None)
            dec_col = next((c for c in dec_candidates if c in data.names), None)

            if ra_col is None or dec_col is None:
                raise KeyError(
                    f"Could not find RA/Dec columns in {self.catalog_path}. "
                    f"Available columns: {data.names}"
                )

            mask = np.isfinite(data[ra_col]) & np.isfinite(data[dec_col])

            ra_deg = np.asarray(data[ra_col][mask], dtype=float)
            dec_deg = np.asarray(data[dec_col][mask], dtype=float)

            # Extract base Johnson magnitudes and colors
            has_vmag = "Vmag" in data.names
            has_bmv = "B-V" in data.names
            has_ub = "U-B" in data.names

            vmag = np.asarray(data["Vmag"][mask], dtype=float) if has_vmag else np.full(len(ra_deg), np.nan)
            bmv = np.asarray(data["B-V"][mask], dtype=float) if has_bmv else np.full(len(ra_deg), np.nan)
            ub = np.asarray(data["U-B"][mask], dtype=float) if has_ub else np.full(len(ra_deg), np.nan)

        self.star_mags = {"V": vmag}

        # ------------------------------------------------------------------
        # Compute band magnitudes (DES / LSST system transformations)
        # ------------------------------------------------------------------
        # g-band
        gmag = np.copy(vmag)
        m1 = (bmv > -0.2) & (bmv <= 0.4)
        m2 = (bmv > 0.4) & (bmv <= 2.2)
        gmag[m1] = vmag[m1] + 0.552 * bmv[m1] - 0.099
        gmag[m2] = vmag[m2] + 0.493 * bmv[m2] - 0.067
        self.star_mags["g"] = gmag

        # r-band
        rmag = np.copy(vmag)
        mr = (bmv > -0.2) & (bmv <= 2.2)
        rmag[mr] = vmag[mr] - 0.543 * bmv[mr] + 0.128
        self.star_mags["r"] = rmag

        # i-band
        imag = np.copy(vmag)
        mi = (bmv > -0.2) & (bmv <= 2.2)
        imag[mi] = vmag[mi] - 1.04 * bmv[mi] + 0.312
        self.star_mags["i"] = imag

        # z-band
        zmag = np.copy(vmag)
        mz = (bmv > -0.2) & (bmv <= 2.2)
        zmag[mz] = vmag[mz] - 1.302 * bmv[mz] + 0.417
        self.star_mags["z"] = zmag

        # y-band
        ymag = np.copy(vmag)
        my = (bmv > -0.2) & (bmv <= 2.2)
        ymag[my] = vmag[my] - 1.416 * bmv[my] + 0.504
        self.star_mags["y"] = ymag

        # u-band (u ≈ U_Johnson - 0.7 where U = V + B-V + U-B, fallback to Vmag)
        umag = np.copy(vmag)
        mu = np.isfinite(bmv) & np.isfinite(ub)
        umag[mu] = vmag[mu] + bmv[mu] + ub[mu] - 0.7
        self.star_mags["u"] = umag

        # Convert to 3D Cartesian unit vectors for fast spatial indexing
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)

        self.xyz_stars = np.column_stack([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ])
        self.tree = cKDTree(self.xyz_stars)

    def check_pointings(self, ra_rad, dec_rad, bands=None):
        """Check whether pointings have stars falling in the forbidden annulus
        exceeding the limiting magnitude in the observation filter.

        Parameters
        ----------
        ra_rad : `float` or array-like
            Right Ascension in radians.
        dec_rad : `float` or array-like
            Declination in radians.
        bands : `str` or array-like, optional
            Filter band names for each pointing (e.g. 'r', 'i', 'z'). If None,
            defaults to 'V'.

        Returns
        -------
        compromised : `np.ndarray` of `bool`
            Boolean mask indicating which pointings contain a star in the
            forbidden annulus exceeding the magnitude threshold in the reference filter.
        hit_info : `list` of `list` of `dict`
            List of detected stars per pointing, with star index, magnitude in
            the observation band, V-band magnitude, and angular separation.
        """
        ra_rad = np.atleast_1d(ra_rad)
        dec_rad = np.atleast_1d(dec_rad)
        n_pts = len(ra_rad)

        if bands is None:
            band_list = ["v"] * n_pts
        elif isinstance(bands, str):
            band_list = [bands.lower()] * n_pts
        else:
            band_list = [str(b).lower() for b in bands]

        xyz_obs = np.column_stack([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ])

        candidates = self.tree.query_ball_point(xyz_obs, r=self.d_max)

        compromised = np.zeros(n_pts, dtype=bool)
        hit_info = []

        for i in range(n_pts):
            cand_indices = candidates[i]
            cand_info = []
            if len(cand_indices) > 0:
                dists = np.linalg.norm(
                    self.xyz_stars[cand_indices] - xyz_obs[i], axis=1
                )
                annulus_mask = dists >= self.d_min
                if np.any(annulus_mask):
                    annulus_stars = np.array(cand_indices)[annulus_mask]
                    annulus_dists = dists[annulus_mask]

                    obs_b = band_list[i]
                    if self.use_observation_band:
                        mag_ref_array = self.star_mags.get(obs_b, self.star_mags["V"])
                        mag_thresh = self.get_mag_limit(obs_b)
                    else:
                        mag_ref_array = self.star_mags["V"]
                        mag_thresh = self.get_mag_limit("V")

                    bright_in_band = mag_ref_array[annulus_stars] <= mag_thresh
                    if np.any(bright_in_band):
                        compromised[i] = True
                        matched_stars = annulus_stars[bright_in_band]
                        matched_dists = annulus_dists[bright_in_band]

                        for s_idx, d in zip(matched_stars, matched_dists):
                            sep_deg = 2.0 * np.degrees(
                                np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
                            )
                            cand_info.append({
                                "star_idx": int(s_idx),
                                "band": obs_b,
                                "star_mag_band": float(mag_ref_array[s_idx]),
                                "star_vmag": float(self.star_mags["V"][s_idx]),
                                "mag_threshold": float(mag_thresh),
                                "sep_deg": float(sep_deg),
                            })
            hit_info.append(cand_info)

        return compromised, hit_info

    def __call__(self, observation_array, conditions=None):
        """Process and detail candidate observations.

        Parameters
        ----------
        observation_array : `ObservationArray`
            The candidate observations to detail.
        conditions : `rubin_scheduler.scheduler.conditions` object, optional
            Current environmental conditions from the scheduler.

        Returns
        -------
        observation_array : `ObservationArray`
            The updated observation array, with coordinates shifted for any
            compromised pointings.
        """
        if len(observation_array) == 0:
            return observation_array

        self.n_evaluated += len(observation_array)
        ra = observation_array["RA"]
        dec = observation_array["dec"]

        bands = (
            observation_array["band"]
            if "band" in observation_array.dtype.names
            else None
        )

        compromised, hit_info = self.check_pointings(ra, dec, bands=bands)

        if not np.any(compromised):
            return observation_array

        comp_indices = np.where(compromised)[0]
        self.n_shifted += len(comp_indices)

        # Create a copy to avoid modifying the original survey state
        observation_array = observation_array.copy()

        for idx in comp_indices:
            orig_ra = float(ra[idx])
            orig_dec = float(dec[idx])
            obs_band = str(observation_array["band"][idx]).lower() if "band" in observation_array.dtype.names else "v"

            if obs_band in self.n_shifted_by_band:
                self.n_shifted_by_band[obs_band] += 1
            else:
                self.n_shifted_by_band["other"] += 1

            if self.offset_mode == "gnomonic":
                new_ra, new_dec = gnomonic_project_tosky(
                    self.offset_ra_rad,
                    self.offset_dec_rad,
                    orig_ra,
                    orig_dec,
                )
            else:
                new_ra = orig_ra + self.offset_ra_rad
                new_dec = orig_dec + self.offset_dec_rad

            wrapped_ra, wrapped_dec = wrap_ra_dec(
                np.array([new_ra]), np.array([new_dec])
            )
            observation_array["RA"][idx] = wrapped_ra[0]
            observation_array["dec"][idx] = wrapped_dec[0]

            # Annotate scheduler notes to trace which visits were modified
            if self.annotate_note:
                for col in ["scheduler_note", "note"]:
                    if col in observation_array.dtype.names:
                        current = str(observation_array[col][idx])
                        if self.note_tag not in current:
                            max_len = 59
                            separator = ";" if current else ""
                            combined = f"{current}{separator}{self.note_tag}"[
                                :max_len
                            ]
                            observation_array[col][idx] = combined

            # Record shift history
            self.shifted_history.append({
                "orig_ra_deg": float(np.degrees(orig_ra)),
                "orig_dec_deg": float(np.degrees(orig_dec)),
                "new_ra_deg": float(np.degrees(wrapped_ra[0])),
                "new_dec_deg": float(np.degrees(wrapped_dec[0])),
                "offset_ra_deg": self.offset_ra_deg,
                "offset_dec_deg": self.offset_dec_deg,
                "band": obs_band,
                "stars": hit_info[idx],
                "night": getattr(conditions, "night", None)
                if conditions
                else None,
                "mjd": float(observation_array["mjd"][idx])
                if "mjd" in observation_array.dtype.names
                else None,
            })

        return observation_array

    def summary(self):
        """Print a summary of the detailer's actions."""
        pct = (
            (self.n_shifted / self.n_evaluated * 100.0)
            if self.n_evaluated > 0
            else 0.0
        )
        print("=" * 65)
        print("HorseshoeAvoidanceDetailer Summary:")
        print(f"  HEALPix NSIDE:      {self.nside}")
        print(f"  Catalog:            {self.catalog_path.name}")
        print(f"  Stars indexed:      {len(self.xyz_stars)}")
        print(f"  Evaluation Mode:    {'Per-Observation Filter' if self.use_observation_band else 'V-band Only'}")
        if isinstance(self.mag_limit, dict):
            print(f"  Magnitude Limits:   {self.mag_limit}")
        else:
            print(f"  Magnitude Limit:    Mag <= {self.mag_limit} in reference filter")
        print(f"  Annulus:            [{self.inner_radius_deg:.3f}, {self.outer_radius_deg:.3f}] deg")
        print(f"  Offsets applied:    ΔRA = +{self.offset_ra_deg} deg, ΔDec = +{self.offset_dec_deg} deg")
        print(f"  Evaluated visits:   {self.n_evaluated}")
        print(f"  Shifted visits:     {self.n_shifted} ({pct:.3f}%)")
        print("  Shifted by band:")
        for b in ["u", "g", "r", "i", "z", "y"]:
            cnt = self.n_shifted_by_band.get(b, 0)
            print(f"    - Band {b}: {cnt:4d} visits")
        print("=" * 65)

    def get_shifted_dataframe(self):
        """Return history of shifted visits as a pandas DataFrame."""
        if not self.shifted_history:
            return pd.DataFrame()
        return pd.DataFrame(self.shifted_history)


def add_horseshoe_detailer_to_scheduler(
    scheduler,
    detailer=None,
    survey_types=None,
    position="prepend_tracking",
    **detailer_kwargs,
):
    """Add HorseshoeAvoidanceDetailer to all matching surveys in a CoreScheduler.

    Parameters
    ----------
    scheduler : `rubin_scheduler.scheduler.CoreScheduler`
        Scheduler instance whose surveys will be updated.
    detailer : `HorseshoeAvoidanceDetailer`, optional
        Existing detailer instance. If None, a new instance is created with
        `detailer_kwargs`.
    survey_types : `tuple` of `type`, optional
        If provided, only add the detailer to surveys matching these types
        (e.g., `(BlobSurvey, GreedySurvey)`). Defaults to None (adds to all
        surveys containing a `detailers` list).
    position : `str` or `int`, optional
        Where to insert the detailer in each survey's `detailers` list:
        - "prepend": Insert at index 0.
        - "prepend_tracking" (default): Insert immediately before
          `TrackingInfoDetailer` if present, ensuring tracking information
          reflects the nudged RA/Dec. Otherwise appends.
        - "append": Append at the end of the list.
        - `int`: Specific index.
    **detailer_kwargs
        Keyword arguments passed to `HorseshoeAvoidanceDetailer` if `detailer`
        is None.

    Returns
    -------
    detailer : `HorseshoeAvoidanceDetailer`
        The detailer instance attached to the scheduler.
    count : `int`
        Number of surveys updated.
    """
    if detailer is None:
        detailer = HorseshoeAvoidanceDetailer(**detailer_kwargs)

    count = 0
    for tier in scheduler.survey_lists:
        for survey in tier:
            if hasattr(survey, "detailers") and isinstance(survey.detailers, list):
                if survey_types is None or isinstance(survey, survey_types):
                    # Prevent adding duplicate instances
                    if detailer in survey.detailers:
                        continue

                    if position == "prepend":
                        survey.detailers.insert(0, detailer)
                    elif position == "prepend_tracking":
                        tracking_idx = next(
                            (
                                i
                                for i, d in enumerate(survey.detailers)
                                if type(d).__name__ == "TrackingInfoDetailer"
                            ),
                            None,
                        )
                        if tracking_idx is not None:
                            survey.detailers.insert(tracking_idx, detailer)
                        else:
                            survey.detailers.append(detailer)
                    elif position == "append":
                        survey.detailers.append(detailer)
                    elif isinstance(position, int):
                        survey.detailers.insert(position, detailer)
                    else:
                        survey.detailers.append(detailer)

                    count += 1

    return detailer, count


if __name__ == "__main__":
    print("Testing HorseshoeAvoidanceDetailer with per-filter magnitude triggering...")
    detailer = HorseshoeAvoidanceDetailer(mag_limit=3.0, use_observation_band=True)

    # Star 162 in YBSC has Vmag = 3.27, but Rmag = 2.70 (red star!)
    hdul = fits.open(detailer.catalog_path)
    data = hdul[1].data
    mask = np.isfinite(data["coord_ra"]) & np.isfinite(data["coord_dec"])
    ra_star = data["coord_ra"][mask][162]
    dec_star = data["coord_dec"][mask][162]
    hdul.close()

    # Pointing 2.01 deg away from Star 162 in Dec (in the forbidden annulus)
    ra_obs = np.radians(ra_star)
    dec_obs = np.radians(dec_star + 2.01)

    # 1. Observation in band 'r' (star Rmag = 2.70 <= 3.0 -> SHOULD TRIGGER)
    obs_r = ObservationArray(n=1)
    obs_r["RA"] = [ra_obs]
    obs_r["dec"] = [dec_obs]
    obs_r["band"] = ["r"]
    obs_r["scheduler_note"] = ["visit_r"]

    obs_r = detailer(obs_r)
    shifted_r = (obs_r["RA"][0] != ra_obs) or (obs_r["dec"][0] != dec_obs)
    print(f"Observation in Band R (star Rmag = 2.70 <= 3.0): Triggered shift? -> {shifted_r}")

    # 2. Observation in band 'g' (star Gmag ≈ 3.8 > 3.0 -> SHOULD NOT TRIGGER)
    obs_g = ObservationArray(n=1)
    obs_g["RA"] = [ra_obs]
    obs_g["dec"] = [dec_obs]
    obs_g["band"] = ["g"]
    obs_g["scheduler_note"] = ["visit_g"]

    obs_g = detailer(obs_g)
    shifted_g = (obs_g["RA"][0] != ra_obs) or (obs_g["dec"][0] != dec_obs)
    print(f"Observation in Band G (star Gmag ≈ 3.8 > 3.0):  Triggered shift? -> {shifted_g}")

    detailer.summary()
