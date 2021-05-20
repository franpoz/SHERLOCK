import os
from argparse import ArgumentParser

import astroplan
import numpy
from astropy.coordinates import SkyCoord, get_moon
from astropy.time import Time
import astropy.units as u
from astroplan import EclipsingSystem, moon_illumination, Constraint
import pandas as pd
import numpy as np
from astroplan import (Observer, FixedTarget, AtNightConstraint, AltitudeConstraint)
import datetime as dt


class MoonIlluminationSeparationConstraint(Constraint):
    """
    Constrain the distance between the Earth's moon and some targets.
    """

    def __init__(self, min_dist, max_dist):
        """
        Parameters
        ----------
        min_dist : `~astropy.units.Quantity`
            Minimum moon distance when moon illumination is 0
        max_dist : `~astropy.units.Quantity`
            Maximum moon distance when moon illumination is 1
        """
        self.min_dist = min_dist
        self.max_dist = max_dist

    def compute_constraint(self, times, observer, targets):
        # removed the location argument here, which causes small <1 deg
        # innacuracies, but it is needed until astropy PR #5897 is released
        # which should be astropy 1.3.2
        moon = get_moon(times)
        # note to future editors - the order matters here
        # moon.separation(targets) is NOT the same as targets.separation(moon)
        # the former calculates the separation in the frame of the moon coord
        # which is GCRS, and that is what we want.
        moon_separation = moon.separation(targets)
        illumination = moon_illumination(times)
        min_dist = self.min_dist.value + (self.max_dist.value - self.min_dist.value) * illumination
        mask = min_dist <= moon_separation.degree
        return mask


if __name__ == '__main__':
    ap = ArgumentParser(description='Planning of observations from Sherlock objects of interest')
    ap.add_argument('--object_dir',
                    help="If the object directory is not your current one you need to provide the ABSOLUTE path",
                    required=False)
    ap.add_argument('--candidate', help="Candidate number from promising list", required=True)
    ap.add_argument("--observatories", help="Csv file containing the observatories coordinates", required=False)
    ap.add_argument('--lat', help="Observer latitude", required=False)
    ap.add_argument('--lon', help="Observer longitude", required=False)
    ap.add_argument('--alt', help="Observer altitude", required=False)
    ap.add_argument('--tz', help="Time zone (in hours offset from UTC)", required=False)
    ap.add_argument('--min_altitude', help="Minimum altitude of the target to be observable", default=23,
                    required=False)
    ap.add_argument('--moon_min_dist', help="Minimum required moon distance for moon minimum illumination.", default=30,
                    required=False)
    ap.add_argument('--moon_max_dist', help="Minimum required moon distance for moon maximum illumination.", default=55,
                    required=False)
    ap.add_argument('--max_days', help="Maximum number of days for the plan to take.", default=365, required=False)
    args = ap.parse_args()
    if args.observatories is None and (args.lat is None or args.lon is None or args.alt is None):
        raise ValueError("You either need to set the 'observatories' property or the lat, lon and alt.")
    object_dir = os.getcwd() if args.object_dir is None else args.object_dir
    ns_derived_file = object_dir + "/results/ns_derived_table.csv"
    ns_file = object_dir + "/results/ns_table.csv"
    if not os.path.exists(ns_derived_file) or not os.path.exists(ns_file):
        raise ValueError("Bayesian fit posteriors files {" + ns_file + ", " + ns_derived_file + "} not found")
    fit_derived_results = pd.read_csv(object_dir + "/results/ns_derived_table.csv")
    fit_results = pd.read_csv(object_dir + "/results/ns_table.csv")
    period_row = fit_results[fit_results["#name"].str.contains("_period")]
    period = period_row["median"].item()
    period_low_err = float(period_row["lower_error"].item())
    period_up_err = float(period_row["upper_error"].item())
    epoch_row = fit_results[fit_results["#name"].str.contains("_epoch")]
    epoch = epoch_row["median"].item()
    epoch_low_err = float(epoch_row["lower_error"].item())
    epoch_up_err = float(epoch_row["upper_error"].item())
    duration_row = fit_derived_results[fit_derived_results["#property"].str.contains("Total transit duration")]
    duration = duration_row["value"].item()
    duration_low_err = float(duration_row["lower_error"].item())
    duration_up_err = float(duration_row["upper_error"].item())
    name = "SOI_" + str(args.candidate)
    star_df = pd.read_csv(object_dir + "/params_star.csv")
    ra = star_df.iloc[0]["ra"]
    dec = star_df.iloc[0]["dec"]
    coords = str(ra) + " " + str(dec)
    if args.observatories is not None:
        observatories_df = pd.read_csv(args.observatories)
    else:
        observatories_df = pd.DataFrame(columns=['name', 'tz', 'lat', 'long', 'alt'])
        observatories_df = observatories_df.append("Obs-1", args.tz, args.lat, args.lon, args.alt)
    # TODO probably convert epoch to proper JD
    epoch_bjd = epoch + 2457000.0
    primary_eclipse_time = Time(epoch_bjd, format='jd')
    target = FixedTarget(SkyCoord(coords, unit=(u.deg, u.deg)))
    n_transits = args.max_days // period
    obs_time = Time.now()
    system = EclipsingSystem(primary_eclipse_time=primary_eclipse_time, orbital_period=u.Quantity(period, unit="d"),
                             duration=u.Quantity(duration, unit="h"), name=name)
    observables_df = pd.DataFrame(columns=['observatory', 'ingress', 'egress', 'midtime', 'ingress_local',
                                           'egress_local', 'midtime_local',
                                           "midtime_up_err_h",
                                           "midtime_low_err_h", 'twilight_evening',
                                           'twilight_morning', 'twilight_evening_local',
                                           'twilight_morning_local', 'observable', 'moon_phase', 'moon_dist'])
    for index, observatory_row in observatories_df.iterrows():
        observer_timezone = dt.timezone(dt.timedelta(hours=observatory_row["tz"]))
        observer_site = Observer(latitude=observatory_row["lat"], longitude=observatory_row["lon"],
                                 elevation=u.Quantity(observatory_row["alt"], unit="m"),
                                 timezone=observer_timezone)
        midtransit_times = system.next_primary_eclipse_time(obs_time, n_eclipses=n_transits)
        ingress_egress_times = system.next_primary_ingress_egress_time(obs_time, n_eclipses=n_transits)
        constraints = [AtNightConstraint.twilight_civil(), AltitudeConstraint(min=args.min_altitude * u.deg),
                       MoonIlluminationSeparationConstraint(min_dist=args.moon_min_dist * u.deg,
                                                            max_dist=args.moon_max_dist * u.deg)]
        midtime_observable = astroplan.is_event_observable(constraints, observer_site, target, times=midtransit_times)
        entire_observable = astroplan.is_event_observable(constraints, observer_site, target, times_ingress_egress=ingress_egress_times)
        visible_midtransit_times = midtransit_times[numpy.where(midtime_observable[0])]
        visible_ingress_egress_times = ingress_egress_times[numpy.where(entire_observable[0])]
        moon_for_visible_midtransit_times = get_moon(visible_midtransit_times)
        moon_dist_visible_midtransit_times = moon_for_visible_midtransit_times.separation(
            SkyCoord(star_df.iloc[0]["ra"], star_df.iloc[0]["dec"], unit="deg"))
        i = 0
        for midtransit_time in visible_midtransit_times:
            ingress_egress_for_midtransit = next((iet for iet in ingress_egress_times
                                                  if iet[0] < midtransit_time and iet[1] > midtransit_time))
            visible_ingress_egress_for_midtransit = next((iet for iet in visible_ingress_egress_times
                                                          if iet[0] < midtransit_time and iet[1] > midtransit_time), None)
            moon_dist = round(moon_dist_visible_midtransit_times[i].degree)
            observable = 1 if visible_ingress_egress_for_midtransit is not None else 0.5
            twilight_evening = observer_site.twilight_evening_nautical(midtransit_time)
            twilight_morning = observer_site.twilight_morning_nautical(midtransit_time)
            moon_phase = np.round(astroplan.moon_illumination(midtransit_time), 2)
            # TODO get is_event_observable for several parts of the transit (ideally each 5 mins) to get the proper observable percent. Also with baseline
            transits_since_epoch = round((midtransit_time.jd - epoch_bjd) / period)
            midtransit_time_low_err = np.round((transits_since_epoch * period_low_err + epoch_low_err) * 24, 2)
            midtransit_time_up_err = np.round((transits_since_epoch * period_up_err + epoch_up_err) * 24, 2)
            observables_df = observables_df.append({"observatory": observatory_row["name"], "ingress": ingress_egress_for_midtransit[0].isot,
                                   "egress": ingress_egress_for_midtransit[1].isot, "midtime": midtransit_time,
                                   "ingress_local": ingress_egress_for_midtransit[0].to_datetime(timezone=observer_timezone),
                                   "egress_local": ingress_egress_for_midtransit[1].to_datetime(timezone=observer_timezone),
                                   "midtime_local": midtransit_time.to_datetime(timezone=observer_timezone),
                                   "midtime_up_err_h": midtransit_time_up_err,
                                   "midtime_low_err_h": midtransit_time_low_err,
                                   "twilight_evening": twilight_evening.isot,
                                   "twilight_morning": twilight_morning.isot,
                                   "twilight_evening_local": twilight_evening.to_datetime(timezone=observer_timezone),
                                   "twilight_morning_local": twilight_morning.to_datetime(timezone=observer_timezone),
                                   "observable": observable, "moon_phase": moon_phase, "moon_dist": moon_dist}, ignore_index=True)
            i = i + 1
    observables_df = observables_df.sort_values(["midtime", "observatory"], ascending=True)
    observables_df.to_csv(object_dir + "/observation_plan.csv", index=False)
    print("Observation plan created in directory: " + object_dir)
