from argparse import ArgumentParser

import astroplan
import numpy
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astroplan import EclipsingSystem, MoonSeparationConstraint
import pandas as pd
import numpy as np
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
    ap.add_argument("--observatories", help="Csv file containing the observatories coordinates", required=False)
    ap.add_argument('--lat', help="Observer latitude", required=False)
    ap.add_argument('--lon', help="Observer longitude", required=False)
    ap.add_argument('--alt', help="Observer altitude", required=False)
    args = ap.parse_args()
    if args.observatories is None and (args.lat is None or args.lon is None or args.alt is None):
        raise ValueError("You either need to set the 'observatories' property or the lat, lon and alt.")
    fit_derived_results = pd.read_csv(args.object_dir + "/results/ns_derived_table.csv")
    fit_results = pd.read_csv(args.object_dir + "/results/ns_table.csv")
    period_row = fit_results[fit_results["#name"].str.contains("_period")]
    period = period_row["median"].item()
    period_low_err = period_row["lower_error"].item()
    period_up_err = period_row["upper_error"].item()
    epoch_row = fit_results[fit_results["#name"].str.contains("_epoch")]
    epoch = epoch_row["median"].item()
    epoch_low_err = epoch_row["lower_error"].item()
    epoch_up_err = epoch_row["upper_error"].item()
    duration_row = fit_derived_results[fit_derived_results["#property"].str.contains("Total transit duration")]
    duration = duration_row["value"].item()
    duration_low_err = duration_row["lower_error"].item()
    duration_up_err = duration_row["upper_error"].item()
    name = "SOI_" + str(args.candidate)
    star_df = pd.read_csv(args.object_dir + "/params_star.csv")
    coords = str(star_df.iloc[0]["ra"]) + " " + str(star_df.iloc[0]["dec"])
    if args.observatories is not None:
        observatories_df = pd.read_csv(args.observatories)
    else:
        observatories_df = pd.DataFrame(columns=['name', 'lat', 'long', 'alt'])
        observatories_df = observatories_df.append("Obs-1", args.lat, args.lon, args.alt)
    # TODO probably convert epoch to proper JD
    epoch_bjd = epoch + 2457000.0
    primary_eclipse_time = Time(epoch_bjd, format='jd')
    target = FixedTarget(SkyCoord(coords, unit=(u.deg, u.deg)))
    n_transits = 365 // period  # This is the roughly number of transits per year
    obs_time = Time.now()
    system = EclipsingSystem(primary_eclipse_time=primary_eclipse_time, orbital_period=u.Quantity(period, unit="d"),
                             duration=u.Quantity(duration, unit="h"), name=name)
    observables_df = pd.DataFrame(columns=['name', 'ingress', 'egress', 'midtime', 'observable', 'moon_phase'])
    for index, observatory_row in observatories_df.iterrows():
        observer_site = Observer(latitude=observatory_row["lat"], longitude=observatory_row["lon"],
                                 elevation=u.Quantity(observatory_row["alt"], unit="m"), timezone='UTC')
        midtransit_times = system.next_primary_eclipse_time(obs_time, n_eclipses=n_transits)
        ingress_egress_times = system.next_primary_ingress_egress_time(obs_time, n_eclipses=n_transits)
        constraints = [AtNightConstraint.twilight_civil(), AltitudeConstraint(min=23 * u.deg),
                       MoonSeparationConstraint(min=45 * u.deg)]
        midtime_observable = astroplan.is_event_observable(constraints, observer_site, target, times=midtransit_times)
        entire_observable = astroplan.is_event_observable(constraints, observer_site, target, times_ingress_egress=ingress_egress_times)
        visible_midtransit_times = midtransit_times[numpy.where(midtime_observable[0])]
        visible_ingress_egress_times = ingress_egress_times[numpy.where(entire_observable[0])]
        for midtransit_time in visible_midtransit_times:
            ingress_egress_for_midtransit = next((iet for iet in ingress_egress_times
                                                  if iet[0] < midtransit_time and iet[1] > midtransit_time))
            visible_ingress_egress_for_midtransit = next((iet for iet in visible_ingress_egress_times
                                                          if iet[0] < midtransit_time and iet[1] > midtransit_time), None)
            observable = 1 if visible_ingress_egress_for_midtransit is not None else 0.5
            #get twilight times
            moon_phase = astroplan.moon_illumination(midtransit_time)
            # TODO get moon distance to target
            # TODO get is_event_observable for several parts of the transit (ideally each 5 mins) to get the proper observable percent. Also with baseline
            # TODO store transit midtime uncertainty
            observables_df = observables_df.append({"name": observatory_row["name"], "ingress": ingress_egress_for_midtransit[0].isot,
                                   "egress": ingress_egress_for_midtransit[1].isot, "midtime": midtransit_time,
                                   "observable": observable, "moon_phase": np.round(moon_phase, 2)}, ignore_index=True)
    observables_df.to_csv(args.object_dir + "/observation_plan.csv", index=False)
    print("Observation plan created in directory: " + args.object_dir)
