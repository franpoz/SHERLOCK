import os
import shutil
from argparse import ArgumentParser

import astroplan
import matplotlib
import numpy
from astroplan.plots import plot_airmass
from astropy.coordinates import SkyCoord, get_moon
import astropy.units as u
from astroplan import EclipsingSystem, moon_illumination, Constraint
import pandas as pd
import numpy as np
from astroplan import (Observer, FixedTarget, AtNightConstraint, AltitudeConstraint)
import datetime as dt
import lightkurve
import matplotlib.pyplot as plt

from astropy.time import Time
from lcbuilder.lcbuilder_class import LcBuilder

from sherlockpipe.observation_report import ObservationReport


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


def get_twin(ax):
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return other_ax
    return None


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
    ap.add_argument('--transit_fraction', help="Minimum transit fraction to be observable.", type=float, default=0.5,
                    required=False)
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
    depth_row = fit_derived_results[fit_derived_results["#property"].str.contains("depth \(dil.\)")]
    depth = float(depth_row["value"].item()) * 1000
    depth_low_err = float(depth_row["lower_error"].item()) * 1000
    depth_up_err = float(depth_row["upper_error"].item()) * 1000
    star_df = pd.read_csv(object_dir + "/params_star.csv")
    object_id = star_df.iloc[0]["obj_id"]
    name = object_id + "_SOI_" + str(args.candidate)
    ra = star_df.iloc[0]["ra"]
    dec = star_df.iloc[0]["dec"]
    coords = str(ra) + " " + str(dec)
    if args.observatories is not None:
        observatories_df = pd.read_csv(args.observatories)
    else:
        observatories_df = pd.DataFrame(columns=['name', 'tz', 'lat', 'long', 'alt'])
        observatories_df = observatories_df.append("Obs-1", args.tz, args.lat, args.lon, args.alt)
    # TODO probably convert epoch to proper JD
    mission, mission_prefix, id_int = LcBuilder().parse_object_info(object_id)
    if mission == "TESS":
        primary_eclipse_time = Time(epoch, format='btjd', scale="tdb")
    elif mission == "Kepler" or mission == "K2":
        primary_eclipse_time = Time(epoch, format='bkjd', scale="tdb")
    else:
        primary_eclipse_time = Time(epoch, format='jd')
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
    plan_dir = object_dir + "/plan"
    images_dir = plan_dir + "/images"
    if os.path.exists(plan_dir):
        shutil.rmtree(plan_dir, ignore_errors=True)
    os.mkdir(plan_dir)
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir, ignore_errors=True)
    os.mkdir(images_dir)
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
        ingress_observables = astroplan.is_event_observable(constraints, observer_site, target, times=ingress_egress_times[:, 0])[0]
        midtime_observables = astroplan.is_event_observable(constraints, observer_site, target, times=midtransit_times)[0]
        egress_observables = astroplan.is_event_observable(constraints, observer_site, target, times=ingress_egress_times[:, 1])[0]
        entire_observables = astroplan.is_event_observable(constraints, observer_site, target, times_ingress_egress=ingress_egress_times)[0]
        moon_for_midtransit_times = get_moon(midtransit_times)
        moon_dist_midtransit_times = moon_for_midtransit_times.separation(
            SkyCoord(star_df.iloc[0]["ra"], star_df.iloc[0]["dec"], unit="deg"))
        moon_phase_midtransit_times = np.round(astroplan.moon_illumination(midtransit_times), 2)
        i = 0
        for midtransit_time in midtransit_times:
            ingress = ingress_egress_times[i][0]
            egress = ingress_egress_times[i][1]
            ingress_observable = ingress_observables[i]
            egress_observable = egress_observables[i]
            midtime_observable = midtime_observables[i]
            entire_observable = entire_observables[i]
            moon_dist = round(moon_dist_midtransit_times[i].degree)
            observable = 0
            if ingress_observable and egress_observable:
                observable = 1
            elif (ingress_observable and midtime_observable) or (midtime_observable and egress_observable):
                observable = 0.5
            elif ingress_observable or egress_observable:
                observable = 0.25
            if observable < args.transit_fraction:
                i = i + 1
                continue
            twilight_evening = observer_site.twilight_evening_nautical(midtransit_time)
            twilight_morning = observer_site.twilight_morning_nautical(midtransit_time)
            moon_phase = moon_phase_midtransit_times[i]
            # TODO get is_event_observable for several parts of the transit (ideally each 5 mins) to get the proper observable percent. Also with baseline
            transits_since_epoch = round((midtransit_time - primary_eclipse_time).jd / period)
            midtransit_time_low_err = np.round((((transits_since_epoch * period_low_err) ** 2 + epoch_low_err ** 2) ** (1 / 2)) * 24, 2)
            midtransit_time_up_err = np.round((((transits_since_epoch * period_up_err) ** 2 + epoch_up_err ** 2) ** (1 / 2)) * 24, 2)
            observables_df = observables_df.append({"observatory": observatory_row["name"], "ingress": ingress.isot,
                                   "egress": egress.isot, "midtime": midtransit_time.isot,
                                   "ingress_local": ingress.to_datetime(timezone=observer_timezone),
                                   "egress_local": egress.to_datetime(timezone=observer_timezone),
                                   "midtime_local": midtransit_time.to_datetime(timezone=observer_timezone),
                                   "midtime_up_err_h": midtransit_time_up_err,
                                   "midtime_low_err_h": midtransit_time_low_err,
                                   "twilight_evening": twilight_evening.isot,
                                   "twilight_morning": twilight_morning.isot,
                                   "twilight_evening_local": twilight_evening.to_datetime(timezone=observer_timezone),
                                   "twilight_morning_local": twilight_morning.to_datetime(timezone=observer_timezone),
                                   "observable": observable, "moon_phase": moon_phase, "moon_dist": moon_dist},
                                                   ignore_index=True)
            delta_t = (egress - ingress) * 3
            init_plot_time = ingress - duration * u.hour
            end_plot_time = egress + duration * u.hour
            plot_time = init_plot_time + delta_t * np.linspace(0, 1, 50)
            plt.tick_params(labelsize=6)
            airmass_ax = plot_airmass(target, observer_site, plot_time, brightness_shading=True, altitude_yaxis=True)
            airmass_ax.axvspan(twilight_morning.plot_date, end_plot_time.plot_date, color='white')
            airmass_ax.axvspan(init_plot_time.plot_date, twilight_evening.plot_date, color='white')
            airmass_ax.axvspan(twilight_evening.plot_date, twilight_morning.plot_date, color='gray')
            airmass_ax.axhspan(1. / np.cos(np.radians(90 - args.min_altitude)), 5.0, color='green')
            airmass_ax.get_figure().gca().set_title("")
            airmass_ax.get_figure().gca().set_xlabel("")
            airmass_ax.get_figure().gca().set_ylabel("")
            airmass_ax.set_xlabel("")
            airmass_ax.set_ylabel("")
            xticks = []
            xticks_labels = []
            if ingress > init_plot_time and ingress < end_plot_time:
                xticks.append(ingress.plot_date)
                hour_min_sec_arr = ingress.isot.split("T")[1].split(":")
                xticks_labels.append("T1_" + hour_min_sec_arr[0] + ":" + hour_min_sec_arr[1])
                plt.axvline(x=ingress.plot_date, color="orange")
            if midtransit_time > init_plot_time and midtransit_time < end_plot_time:
                xticks.append(midtransit_time.plot_date)
                hour_min_sec_arr = midtransit_time.isot.split("T")[1].split(":")
                xticks_labels.append("T0_" + hour_min_sec_arr[0] + ":" + hour_min_sec_arr[1])
                plt.axvline(x=midtransit_time.plot_date, color="black")
            if egress > init_plot_time and egress < end_plot_time:
                xticks.append(egress.plot_date)
                hour_min_sec_arr = egress.isot.split("T")[1].split(":")
                xticks_labels.append("T4_" + hour_min_sec_arr[0] + ":" + hour_min_sec_arr[1])
                plt.axvline(x=egress.plot_date, color="orange")
            airmass_ax.xaxis.set_tick_params(labelsize=5)
            airmass_ax.set_xticks([])
            airmass_ax.set_xticklabels([])
            degrees_ax = get_twin(airmass_ax)
            degrees_ax.yaxis.set_tick_params(labelsize=6)
            degrees_ax.set_yticks([1., 1.55572383, 2.])
            degrees_ax.set_yticklabels([90, 50, 30])
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(1.25, 0.75)
            plt.savefig(plan_dir + "/images/" + observatory_row["name"] + "_" + str(midtransit_time.isot)[:-4] + ".png",
                        bbox_inches='tight')
            plt.close()
            i = i + 1
    observables_df = observables_df.sort_values(["midtime", "observatory"], ascending=True)
    observables_df.to_csv(plan_dir + "/observation_plan.csv", index=False)
    print("Observation plan created in directory: " + object_dir)
    report = ObservationReport(observatories_df, observables_df, object_id, plan_dir,
                               epoch, epoch_low_err, epoch_up_err, period, period_low_err, period_up_err, duration,
                               duration_low_err, duration_up_err, depth, depth_low_err, depth_up_err,
                               args.transit_fraction, args.moon_min_dist, args.moon_max_dist, args.min_altitude,
                               args.max_days)
    report.create_report()
    shutil.rmtree(images_dir, ignore_errors=True)
