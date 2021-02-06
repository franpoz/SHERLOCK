from argparse import ArgumentParser

import astroplan
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astroplan import EclipsingSystem
import pandas as pd
from os import path
from astroplan import (Observer, FixedTarget, PrimaryEclipseConstraint, AtNightConstraint, AltitudeConstraint,
                       LocalTimeConstraint)
import datetime as dt

if __name__ == '__main__':
    ap = ArgumentParser(description='Planning of observations from Sherlock objects of interest')
    ap.add_argument('--object_dir',
                    help="If the object directory is not your current one you need to provide the ABSOLUTE path",
                    required=True)
    ap.add_argument('--candidate', help="Candidate number from promising list", required=True)
    ap.add_argument('--lat', help="Observer latitude", required=True)
    ap.add_argument('--lon',
                    help="Observer longitude",
                    required=True)
    args = ap.parse_args()
    fit_derived_results = pd.read_csv(args.object_dir + "/ns_derived_table.csv")
    fit_results = pd.read_csv(args.object_dir + "/ns_table.csv")
    period_row = fit_results[fit_results["#name"].str.contains("_period")]
    period = period_row["median"].item()
    period_low_err = period_row["lower_error"].item()
    period_up_err = period_row["upper_error"].item()
    epoch_row = fit_results[fit_results["#name"].str.contains("_epoch")]
    epoch = epoch_row["median"].item()
    epoch_low_err = epoch_row["lower_error"].item()
    epoch_up_err = epoch_row["upper_error"].item()
    duration_row = fit_derived_results[fit_derived_results["#property"].str.contains("Full-transit duration")]
    duration = duration_row["value"].item()
    duration_low_err = duration_row["lower_error"].item()
    duration_up_err = duration_row["upper_error"].item()
    name = "SOI_" + str(args.candidate)
    # TODO probably convert epoch to proper JD
    primary_eclipse_time = Time(epoch, format='jd')
    system = EclipsingSystem(primary_eclipse_time=primary_eclipse_time, orbital_period=u.Quantity(period, unit="d"),
                             duration=u.Quantity(duration, unit="h"), name=name)
    observer_site = Observer(latitude=args.lat, longitude=args.lon, timezone='UTC')
    # TODO select proper coordinates: load from params_star.csv
    # TODO select proper observer place (ground, space)
    coords = "1:12:43.2 +1:12:43"
    target = FixedTarget(SkyCoord(coords, unit=(u.deg, u.deg)))
    # TODO select proper time
    n_transits = 100  # This is the roughly number of transits per year
    obs_time = Time('2017-01-01 12:00')
    # TODO bulk to file
    midtransit_times = system.next_primary_eclipse_time(obs_time, n_eclipses=n_transits)
    ingress_egress_times = system.next_primary_ingress_egress_time(obs_time, n_eclipses=n_transits)
    # TODO select local times somehow
    min_local_time = dt.time(19, 0)  # 19:00 local time at APO (7pm)
    max_local_time = dt.time(0, 0)  # 00:00 local time at APO (midnight)
    constraints = [AtNightConstraint.twilight_civil(), AltitudeConstraint(min=30*u.deg),
                   LocalTimeConstraint(min=min_local_time, max=max_local_time)]
    midtime_observable = astroplan.is_event_observable(constraints, observer_site, target, times=midtransit_times)
    entire_observable = astroplan.is_event_observable(constraints, observer_site, target, times=midtransit_times,
                                  times_ingress_egress=ingress_egress_times)
