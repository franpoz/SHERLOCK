import glob
import io
import logging
import os
import shutil
import zipfile

from scipy.stats import binned_statistic
from uncertainties import ufloat
import batman
import matplotlib.pyplot as plt
import requests
import wotan
from astropy.timeseries import BoxLeastSquares
from exoml.santo.SANTO import SANTO
import numpy as np
import pandas as pd
from lcbuilder.helper import LcbuilderHelper
from matplotlib.gridspec import GridSpec
from watson.watson import Watson, SingleTransitProcessInput
import astropy.units as u
from sherlockpipe.constants import MOMENTUM_DUMP_QUALITY_FLAG
from sherlockpipe.loading.common import load_from_yaml
from sherlockpipe.loading.tool_with_candidate import ToolWithCandidate
from sherlockpipe.single_transits.report import MoriartyReport


class MoriartySearch(ToolWithCandidate):
    watson = None

    def __init__(self, object_dir, object_id, is_candidate_from_search, candidates_df, transits_mask=[],
                 batch_size=256, threshold=0.5,
                 cache_dir=os.path.expanduser('~') + "/") -> None:
        super().__init__(is_candidate_from_search, candidates_df)
        self.cache_dir = cache_dir
        self.object_dir = os.getcwd() if object_dir is None else object_dir
        self.object_id = object_id
        self.transits_mask = transits_mask
        self.batch_size = batch_size
        self.threshold = threshold

    def run(self, cpus, **kwargs):
        model_dir = f"{self.cache_dir}/.sherlockpipe/moriarty/0.0.1"
        if not os.path.exists(model_dir) or len(os.listdir(f"{model_dir}/SANTO_model")) != 9:
            r = requests.get(
                "https://www.dropbox.com/scl/fi/vt6t215uqdlofle0as1h6/SANTO_model_1.0.0.zip?rlkey=qvsjd9e04jilpu0g175bbngza&st=6xdemxfu&dl=1")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(f'{model_dir}/')
        model_dir = f"{model_dir}/SANTO_model"
        santo = SANTO()
        predictions, spectra, target_stats_df = santo.predict(self.object_dir, ['/lc.csv'],
                                                                f"{model_dir}/SANTO_model_chk.h5",
                                                                batch_size=self.batch_size, plot=False,
                                                                plot_positives=False,
                                                                tagged_data=False,
                                                                smooth=False, sherlock=True,
                                                                threshold=self.threshold)
        moriarty_dir = f"{self.object_dir}/moriarty" if len(self.search_candidates_df) == 0 else \
            f"{self.object_dir}/moriarty_{','.join(self.search_candidates_df['run'].astype(int).astype(str))}"
        if os.path.exists(moriarty_dir):
            shutil.rmtree(moriarty_dir)
        os.mkdir(moriarty_dir)
        lc_file = f"{self.object_dir}/lc.csv"
        lc_data_file = f"{self.object_dir}/lc_data.csv"
        apertures_file = f"{self.object_dir}/apertures.yaml"
        star_df = pd.read_csv(f"{self.object_dir}/params_star.csv")
        lc_data = pd.read_csv(lc_data_file)
        lc_df = pd.read_csv(lc_file)
        apertures = load_from_yaml(apertures_file)
        apertures = apertures['sectors']
        time = lc_df['time'].values
        flux = lc_df['flux'].values
        cadence = np.median(np.diff(time)) * 24 * 3600
        lc_data_time = lc_data['time'].values
        predictions = predictions['/lc.csv']
        #TODO mask predictions, time and flux
        spectra = spectra['/lc.csv']
        np.savetxt(f"{moriarty_dir}/predictions.csv", predictions, delimiter=',')
        time_predictions = time[:len(predictions)]
        time_filled, flux_filled, cadence_days = santo.fill_data_gaps(time_predictions, predictions)
        power_x, spectra = santo.compute_autocorrelation(flux_filled, cadence_s=cadence)
        for transit_mask in self.transits_mask:
            period = transit_mask['P']
            duration = transit_mask['D']
            t0 = transit_mask['T0']
            time_predictions, predictions, _ = LcbuilderHelper.mask_transits(time_predictions, predictions, period, duration * 2 / 24 / 60, t0)
            time, flux, _ = LcbuilderHelper.mask_transits(time, flux, period, duration * 2 / 24 / 60, t0)
        self.plot_lc_preds_spectrum_broken_x(time, flux, predictions, power_x, spectra,
                                        1e-7, moriarty_dir, self.object_id)
        is_positive = predictions >= 0.5
        diff = np.diff(is_positive.astype(int))
        positive_starts = np.where(diff == 1)[0] + 1
        positive_ends = np.where(diff == -1)[0] + 1
        if is_positive[0]:
            positive_starts = np.insert(positive_starts, 0, 0)
        if is_positive[-1]:
            positive_ends = np.append(positive_ends, len(predictions))
        N = predictions.shape[0]
        std = np.nanstd(flux[0:N])
        stats_df = pd.DataFrame(columns=['target_file', 'type', 'depth', 'std', 'snr', 't0', 'duration_points', 'max_score'])
        for start, end in zip(positive_starts, positive_ends):
            region_preds = predictions[start:end]
            t0 = np.nanmedian(time[start:end][len(region_preds) // 2])
            depth = np.nanmedian(flux[start:end][len(region_preds) // 2])
            snr = (1 - depth) / std * np.sqrt(len(region_preds))
            target_stats = {'target_file': ['lc.csv'], 'type': ['p'], 'depth': [depth], 'std': [std],
                            'snr': [snr], 't0': [t0], 'duration_points': [end - start],
                            'max_score': [np.max(region_preds)]}
            stats_df = pd.concat([stats_df, pd.DataFrame.from_dict(target_stats)], ignore_index=True)
        fit_df = pd.DataFrame(columns=['target_file', 'type', 'depth', 'depth_err', 'snr', 't0', 'duration(h)',
                                       'duration_err(h)', 'max_score', 'rp', 'rp_err', 'period', 'period_err'])
        for index, stats_row in stats_df.iterrows():
            t0_index = np.abs(time - stats_row['t0']).argmin()
            duration_points = stats_row['duration_points']
            half_duration_points = duration_points // 2
            half_window_points = duration_points * 5
            min_index = int(np.max([t0_index - half_window_points]))
            max_index = int(np.min([t0_index + half_window_points, len(time) - 1]))
            no_gaps = np.all(np.diff(np.sort(time[min_index:max_index])) <= 1200 / (3600 * 24))
            momentum_dumps_lc_data_mask = np.bitwise_and(lc_data["quality"].to_numpy(), MOMENTUM_DUMP_QUALITY_FLAG) >= 1
            time_with_dumps = lc_data_time[momentum_dumps_lc_data_mask]
            if not no_gaps or len(time[min_index:max_index]) == 0:
                logging.info(f"Ignoring epoch {stats_row['t0']} with gaps")
                continue
            elif any(time[min_index] <= v <= time[max_index] for v in time_with_dumps):
                logging.info(f"Ignoring epoch {stats_row['t0']} with momentum dumps")
                continue
            elif self.in_transit_mask(stats_row['t0'], duration_points):
                logging.info(f"Ignoring epoch {stats_row['t0']} with in-transit flag")
                continue
            #TODO fit transit
            duration_days = duration_points * cadence / 3600 / 24
            detrended_flux = wotan.flatten(time[min_index:max_index], flux[min_index:max_index], method='biweight',
                                           window_length=duration_days * 16)
            bls = BoxLeastSquares(time[min_index:max_index], detrended_flux)
            bls_result = bls.power([100], np.linspace(duration_days / 2, duration_days * 4, 10))
            bls_dur_points = bls_result['duration'][0] * 3600 * 24 / 120
            subset_time = time[min_index:max_index]
            bls_snr = (bls_result['depth'][0] / np.nanstd(detrended_flux) *
                       np.sqrt(len(np.argwhere((bls_result['transit_time'][0] - bls_result['duration'][0] / 2 < subset_time) &
                                               (subset_time > bls_result['transit_time'][0] + bls_result['duration'][0] / 2))))
                       )
            star_rad_uncertain = ufloat(star_df.loc[0, 'radius'],
                                        max(star_df.loc[0, 'R_star_lerr'], star_df.loc[0, 'R_star_uerr']))
            depth_uncertain = ufloat(bls_result['depth'][0], bls_result['depth_err'][0] * bls_result['depth'][0])
            rp_uncertain = ((star_rad_uncertain ** 2) * depth_uncertain) ** (0.5)
            rp = LcbuilderHelper.convert_from_to(rp_uncertain.n, u.R_sun, u.R_earth)
            periods, periods_summary = self.sample_periods_single_transit(
                star_df.loc[0, 'mass'],
                star_df.loc[0, 'radius'],
                bls_result['duration'][0],
                T14_err_days=bls_result['duration'][0] / 2,
                R_p_earth=rp)
            target_fit = {'target_file': ['lc.csv'], 'type': ['p'], 'depth': [bls_result['depth'][0]],
                          'depth_err': [bls_result['depth_err'][0] * bls_result['depth'][0]],
                          'snr': [bls_snr], 't0': [bls_result['transit_time'][0]], 'duration(h)': [bls_result['duration'][0] * 24],
                          'duration_err(h)': [bls_result['duration'][0] * 24 / 2],
                          'max_score': [stats_row['max_score']],
                          'rp': [rp],
                          'rp_err': [LcbuilderHelper.convert_from_to(rp_uncertain.s, u.R_sun, u.R_earth)],
                          'period': periods_summary['median_days'],
                          'period_min': [periods_summary['p16_days']],
                          'period_max': [periods_summary['p84_days']]
            }
            fit_df = pd.concat([fit_df, pd.DataFrame.from_dict(target_fit)], ignore_index=True)
            min_large_index = int(np.max([0, t0_index - half_duration_points * 100]))
            max_large_index = int(np.min([t0_index + half_duration_points * 100, len(time) - 1]))
            self.plot_light_curve_broken_x(time[min_large_index:max_large_index], flux[min_large_index:max_large_index],
                                      half_duration_points, target_fit['t0'][0],
                                      cadence, moriarty_dir, self.object_id, target_fit,
                                      f"{moriarty_dir}/{self.object_id}_t0_{stats_row['t0']}_all.png",
                                      gap_threshold=0.5,
                                      max_panels=6,  # límite de paneles
                                      min_points_per_panel=200)
            fig, axs = plt.subplots(1, 1, figsize=(14, 7), constrained_layout=True)
            fig.suptitle(f"{stats_row['target_file']} T0={round(bls_result['transit_time'][0], 3)} Duration(h)={round(bls_result['duration'][0] * 24, 2)} "
                         f"S/N={round(bls_snr, 2)} Max Score={round(stats_row['max_score'], 2)}")
            window_time = time[min_index:max_index]
            window_flux = flux[min_index:max_index]
            center_point_window = len(window_time) // 2
            axs.scatter(window_time[0:center_point_window - half_duration_points],
                        window_flux[0:center_point_window - half_duration_points], color='blue')
            axs.scatter(window_time[center_point_window + half_duration_points:],
                        window_flux[center_point_window + half_duration_points:], color='blue')
            axs.scatter(
                window_time[center_point_window - half_duration_points:center_point_window + half_duration_points],
                window_flux[center_point_window - half_duration_points:center_point_window + half_duration_points],
                color='purple')
            y_binned, edges, _ = binned_statistic(window_time, window_flux, statistic='mean', bins=40)
            x_binned = 0.5 * (edges[1:] + edges[:-1])
            bin_stds, _, _ = binned_statistic(window_time, window_flux, statistic='std', bins=40)
            bin_width = (edges[1] - edges[0])
            bin_centers = edges[1:] - bin_width / 2
            axs.errorbar(x_binned, y_binned, yerr=bin_stds / 2, xerr=bin_width / 2, marker='o', markersize=6,
                         color='darkorange', alpha=1, linestyle='none')
            focus_it_fit_indexes = np.argwhere((bls_result['transit_time'][0] + bls_result['duration'][0] / 2 > window_time) &
                                               (bls_result['transit_time'][0] - bls_result['duration'][0] / 2 < window_time))
            focus_at_fit_indexes = np.argwhere((bls_result['transit_time'][0] + bls_result['duration'][0] / 2 < window_time))
            focus_bt_fit_indexes = np.argwhere((bls_result['transit_time'][0] - bls_result['duration'][0] / 2 > window_time))
            concat_times = np.concatenate((window_time[focus_bt_fit_indexes], window_time[focus_it_fit_indexes], window_time[focus_at_fit_indexes]))
            axs.plot(concat_times,
                     np.concatenate((np.ones(len(window_flux[focus_bt_fit_indexes])),
                                     np.full(len(window_flux[focus_it_fit_indexes]), 1 - bls_result['depth'][0]),
                                     np.ones(len(window_flux[focus_at_fit_indexes])))), color='red')
            #axs.plot(time[min_index:max_index], model_for_target[min_index:max_index], color='red')
            axs.set_xlabel("Time (TBJD)", size=30)
            axs.set_ylabel("Flux norm.", size=30)
            plt.savefig(f"{moriarty_dir}/{self.object_id}_t0_{stats_row['t0']}_focus.png", bbox_inches='tight')
            plt.close()
            # Watson.plot_single_transit(
            #     SingleTransitProcessInput(self.object_dir, self.object_id, 0, f"{self.object_dir}/lc.csv",
            #                               f"{self.object_dir}/lc_data.csv", f"{self.object_dir}/tpfs",
            #                               apertures, stats_row['t0'], 1 - stats_row['depth'],
            #                               duration_points * cadence / 60, 10, 0.05, 10, None))
            # single_transit_file_name = "single_transit_" + str(0) + "_T0_" + str(stats_row['t0']) + ".png"
            # shutil.move(self.object_dir + '/' + single_transit_file_name,
            #             f"{moriarty_dir}/{self.object_id}_t0_{stats_row['t0']}_st.png")
            Watson.plot_single_transit(
                SingleTransitProcessInput(self.object_dir, self.object_id, 0, f"{self.object_dir}/lc.csv",
                                          f"{self.object_dir}/lc_data.csv", f"{self.object_dir}/tpfs",
                                          apertures, bls_result['transit_time'][0], bls_result['depth'][0],
                                          bls_dur_points * cadence / 60, 10, 0.05, 10, None))
            single_transit_file_name = "single_transit_" + str(0) + "_T0_" + str(bls_result['transit_time'][0]) + ".png"
            shutil.move(self.object_dir + '/' + single_transit_file_name,
                        f"{moriarty_dir}/{self.object_id}_t0_{stats_row['t0']}_st.png")
        fit_df.to_csv(f"{moriarty_dir}/{self.object_id}_fit.csv", index=False)
        MoriartyReport(moriarty_dir, self.object_id,
              star_df['ra'].iloc[0],
              star_df['dec'].iloc[0],
              star_df['v'].iloc[0],
              star_df['j'].iloc[0],
              star_df['h'].iloc[0],
              star_df['k'].iloc[0], self.search_candidates_df).create_report()
        for f in glob.glob(f"{moriarty_dir}/*.png"):
            os.remove(f)

    def plot_light_curve_broken_x(self, time, flux, half_duration_points, t0,
                              cadence, moriarty_dir, object_id, target_fit, out_path,
                              gap_threshold=0.5,
                              max_panels=6,
                              min_points_per_panel=200,
                              color_oot="blue",
                              color_it="red",
                              marker_size=6,
                              draw_t0_line=True):
        """
        Broken-X light curve. In-transit = exactly half_duration_points samples on each side
        of the sample whose time is nearest to t0 (index-based definition).
        """

        time = np.asarray(time)
        flux = np.asarray(flux)
        if time.ndim != 1 or flux.ndim != 1 or time.size != flux.size:
            raise ValueError("`time` and `flux` must be 1D arrays of the same length.")
        n = len(time)
        if n == 0:
            raise ValueError("Empty time/flux arrays.")

        # --- Find index closest to t0 robustly (works if t0 is outside range too) ---
        # time assumed sorted ascending
        pos = np.searchsorted(time, t0)
        if pos <= 0:
            center_idx = 0
        elif pos >= n:
            center_idx = n - 1
        else:
            # choose the closer of pos-1 and pos
            center_idx = pos - 1 if (t0 - time[pos - 1]) <= (time[pos] - t0) else pos

        # Inclusive index bounds for in-transit region
        it_lo = max(center_idx - half_duration_points, 0)
        it_hi = min(center_idx + half_duration_points, n - 1)
        it_mask_global = np.zeros(n, dtype=bool)
        it_mask_global[it_lo:it_hi + 1] = True  # inclusive

        # --- Split by large gaps ---
        dt = np.diff(time)
        cuts = np.where(dt > gap_threshold)[0]  # gap between i and i+1 if i in cuts
        starts = np.r_[0, cuts + 1]
        ends   = np.r_[cuts, n - 1]
        segs = [(s, e, e - s + 1) for s, e in zip(starts, ends)]

        # Filter tiny segments; if all removed, fall back to all
        segs_f = [seg for seg in segs if seg[2] >= min_points_per_panel]
        segs = segs_f if segs_f else segs

        # Keep largest panels, then order chronologically
        segs.sort(key=lambda x: x[2], reverse=True)
        segs = segs[:max_panels]
        segs.sort(key=lambda x: x[0])

        # --- Plot ---
        fig = plt.figure(figsize=(25, 8), constrained_layout=True)
        gs = GridSpec(1, len(segs), figure=fig, wspace=0.05)
        fig.suptitle(
            f"{object_id} "
            f"T0={round(target_fit['t0'][0], 3)} "
            f"Duration(h)={round(target_fit['duration(h)'][0], 2)} "
            f"S/N={round(target_fit['snr'][0], 2)} "
            f"Max Score={round(target_fit['max_score'][0], 2)}"
        )

        ymins, ymaxs, axes = [], [], []
        for k, (s, e, _) in enumerate(segs):
            ax = fig.add_subplot(gs[0, k]); axes.append(ax)

            seg_idx = np.arange(s, e + 1)
            it_mask  = it_mask_global[seg_idx]
            oot_mask = ~it_mask

            if np.any(oot_mask):
                ax.scatter(time[seg_idx][oot_mask], flux[seg_idx][oot_mask],
                           s=marker_size, color=color_oot)
            if np.any(it_mask):
                ax.scatter(time[seg_idx][it_mask],  flux[seg_idx][it_mask],
                           s=marker_size, color=color_it, zorder=3)

            if draw_t0_line and (time[s] <= t0 <= time[e]):
                ax.axvline(t0, linestyle='--', alpha=0.4)

            ax.set_xlim(time[s], time[e])

            y_seg = flux[seg_idx]
            ymins.append(np.nanpercentile(y_seg, 1))
            ymaxs.append(np.nanpercentile(y_seg, 99))

            if k != 0:
                ax.set_yticklabels([])
            ax.tick_params(axis='x', labelrotation=0, labelsize=9)
            ax.set_xlabel("TBJD" if k == len(segs) - 1 else "")
        # Shared Y across panels
        ylo, yhi = np.nanmin(ymins), np.nanmax(ymaxs)
        for ax in axes:
            ax.set_ylim(ylo, yhi)
        axes[0].set_ylabel("Flux norm.")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        return out_path

    def plot_lc_preds_spectrum_broken_x(self, time, flux, predictions, power_x, spectra,
                                        zero_epsilon, lcs_dir, target_file,
                                        gap_threshold=0.5,
                                        max_panels=6,
                                        min_points_per_panel=200,
                                        marker_size=1,
                                        dpi=300):
        """
        Figura 3x1 con 'broken X' en las dos primeras filas (curva de luz y predicciones)
        y el espectro completo en la tercera fila.

        - No se colapsa el eje tiempo: se divide en segmentos (columnas) por huecos > gap_threshold.
        - Se muestran hasta 'max_panels' segmentos (los mayores), en orden cronológico.
        - La tercera fila (espectro) ocupa todo el ancho (no se divide).

        Parámetros clave:
          time, flux, predictions: arrays 1D (se usan hasta N puntos = len(predictions)).
          power_x, spectra: arrays para el espectro (no dependen de time).
          zero_epsilon: umbral para calcular límites Y robustos del panel de flux.
        """

        # ---------- helpers ----------
        def add_break_marks(ax_left, ax_right, size=0.02, lw=0.8):
            # Dos rayitas diagonales en los bordes contiguos para indicar el corte
            d = size
            # borde derecho del eje izquierdo
            kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, lw=lw)
            ax_left.plot((1 - d, 1 + d), (-d, d), **kwargs)
            ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            # borde izquierdo del eje derecho
            kwargs = dict(transform=ax_right.transAxes, color='k', clip_on=False, lw=lw)
            ax_right.plot((-d, d), (-d, d), **kwargs)
            ax_right.plot((-d, d), (1 - d, 1 + d), **kwargs)

        # ---------- saneamiento básico ----------
        N = np.asarray(predictions).shape[0]
        time = np.asarray(time)[:N]
        flux = np.asarray(flux)[:N]
        predictions = np.asarray(predictions)[:N]

        if time.ndim != 1 or flux.ndim != 1 or predictions.ndim != 1:
            raise ValueError("time, flux y predictions deben ser 1D.")
        if not (len(time) == len(flux) == len(predictions)):
            raise ValueError("time, flux y predictions deben tener la misma longitud (N).")
        n = len(time)
        if n == 0:
            raise ValueError("Arrays vacíos.")

        # ordenar por tiempo por si acaso
        order = np.argsort(time)
        time = time[order]
        flux = flux[order]
        predictions = predictions[order]

        # ---------- segmentación por huecos grandes ----------
        dt = np.diff(time)
        cuts = np.where(dt > gap_threshold)[0]  # gap entre i -> i+1 si i está en cuts
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, n - 1]
        segments = [(s, e, e - s + 1) for s, e in zip(starts, ends)]

        # filtrar segmentos muy pequeños; si se eliminan todos, usar todos
        segs = [seg for seg in segments if seg[2] >= min_points_per_panel] or segments
        # elegir los mayores y luego ordenar cronológicamente
        segs.sort(key=lambda x: x[2], reverse=True)
        segs = segs[:max_panels]
        segs.sort(key=lambda x: x[0])
        n_cols = len(segs)

        # ---------- anchos proporcionales a la duración temporal ----------
        durations = np.array([float(time[e] - time[s]) for (s, e, _) in segs], dtype=float)
        durations = np.maximum(durations, 1e-6)  # evitar ratios ~0

        # ---------- figura y gridspec (control manual de espacios) ----------
        fig = plt.figure(figsize=(16, 8), constrained_layout=False)
        gs = GridSpec(
            3, n_cols, figure=fig,
            height_ratios=[1.0, 1.0, 1.2],
            width_ratios=durations,
            wspace=0.005
        )
        # márgenes ajustados (casi sin espacio horizontal)
        fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.08, wspace=0.005, hspace=0.08)

        # ---------- límites Y robustos ----------
        # Fila 0 (flux)
        keys_gt_zero = np.argwhere(flux > zero_epsilon).flatten()
        if keys_gt_zero.size > 0:
            y0_min = np.nanmin(flux[keys_gt_zero])
            y0_max = np.nanmax(flux[keys_gt_zero])
        else:
            y0_min = np.nanmin(flux)
            y0_max = np.nanmax(flux)
        if not np.isfinite(y0_min) or not np.isfinite(y0_max) or y0_min == y0_max:
            y0_min, y0_max = np.nanmin(flux), np.nanmax(flux)

        # Fila 1 (predictions)
        y1_min = np.nanpercentile(predictions, 1)
        y1_max = np.nanpercentile(predictions, 99)
        if not np.isfinite(y1_min) or not np.isfinite(y1_max) or y1_min == y1_max:
            y1_min, y1_max = np.nanmin(predictions), np.nanmax(predictions)

        # ---------- FILA 0: curva de luz ----------
        axes_lc = []
        for k, (s, e, _) in enumerate(segs):
            ax = fig.add_subplot(gs[0, k])
            axes_lc.append(ax)
            ax.scatter(time[s:e + 1], flux[s:e + 1], s=marker_size)
            ax.set_xlim(time[s], time[e])
            ax.margins(x=0)  # sin margen extra en X
            ax.tick_params(axis='both', labelsize=8)
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(axis='y', labelleft=True, labelright=True)

            # Anotar tamaño real del hueco a la derecha del panel, si existe siguiente
            if k < n_cols - 1:
                _, e_prev, _ = segs[k]
                s_next, _, _ = segs[k + 1]
                gap = time[s_next] - time[e_prev]

        if axes_lc:
            axes_lc[0].set_ylabel("Flux")

        # ---------- FILA 1: predicciones ----------
        axes_pred = []
        for k, (s, e, _) in enumerate(segs):
            ax = fig.add_subplot(gs[1, k])
            axes_pred.append(ax)
            ax.plot(time[s:e + 1], predictions[s:e + 1])
            ax.set_xlim(time[s], time[e])
            ax.margins(x=0)
            ax.tick_params(axis='both', labelsize=8)
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(axis='y', labelleft=True, labelright=True)
            if k == 0:
                ax.set_ylabel("Pred.")
            ax.set_xlabel("TBJD")

        # Repetir anotación de gaps (suave) en fila 1
        for k in range(n_cols - 1):
            _, e_prev, _ = segs[k]
            s_next, _, _ = segs[k + 1]
            gap = time[s_next] - time[e_prev]
        # ---------- Break marks entre columnas (filas 0 y 1) ----------
        for k in range(n_cols - 1):
            add_break_marks(axes_lc[k], axes_lc[k + 1])
            add_break_marks(axes_pred[k], axes_pred[k + 1])

        # ---------- FILA 2: espectro a todo el ancho ----------
        ax_spec = fig.add_subplot(gs[2, :])
        ax_spec.plot(power_x, spectra)
        ax_spec.set_xscale('log')
        ax_spec.set_xlabel("Frecuencia (log)")
        ax_spec.set_ylabel("Power")
        fig.suptitle(target_file)

        # ---------- guardar ----------
        out_path = f"{lcs_dir}/{target_file}.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
        plt.clf();
        plt.close()
        return out_path

    def in_transit_mask(self, t0, duration_mins):
        in_transit = False
        duration_days = duration_mins / 60 / 24
        for transit_mask in self.transits_mask:
            transit_t0 = transit_mask['T0']
            transit_duration_min = transit_mask['D']
            transit_period = transit_mask['P']
            epoch_shift = np.abs(self.nearest_epoch_offset(t0, transit_t0, transit_period))
            if epoch_shift < transit_duration_min / 60 / 24 + duration_days: # ensuring the durations don't overlap
                in_transit = True
                break
        return in_transit

    def nearest_epoch_offset(self, ts, t0, P, signed=True):
        """
        Devuelve la diferencia temporal entre ts y el t0 periódico más cercano.
        ts, t0, P en las mismas unidades (p.ej., días BJD_TDB).
        Acepta escalares o arrays (vectorizado).
        """
        n = np.rint((np.asarray(ts) - t0) / P).astype(int)
        dt = np.asarray(ts) - (t0 + n * P)
        return dt if signed else np.abs(dt)

    def sample_periods_single_transit(
            self,
            M_star_sun,
            R_star_sun,
            T14_days,
            T14_err_days=None,
            R_p_earth=1.0,
            n_samples=100000,
            b_max=0.9,
            geometric_b_prior=True,
            random_state=None,
    ):
        """
        Estima una distribución de períodos plausibles para un tránsito único,
        asumiendo órbita circular y aproximación geométrica simple.

        Parámetros
        ----------
        M_star_sun : float
            Masa estelar en masas solares.
        R_star_sun : float
            Radio estelar en radios solares.
        T14_days : float
            Duración del tránsito (1º a 4º contacto) en días.
        T14_err_days : float or None
            Incertidumbre (1 sigma) en la duración, en días. Si None, se toma fija.
        R_p_earth : float
            Radio planetario (valor más probable) en radios terrestres.
        n_samples : int
            Número de muestras Monte Carlo.
        b_max : float
            Máximo valor de parámetro de impacto que permitimos (en unidades de R*).
        geometric_b_prior : bool
            Si True, usa prior geométrico p(b) ∝ b en [0, b_max].
            Si False, prior uniforme en b.
        random_state : int or None
            Semilla para reproducibilidad.

        Devuelve
        --------
        periods_days : np.ndarray
            Array de períodos en días (filtrado a valores físicos).
        summary : dict
            Diccionario con estadísticos (median, p16, p84, etc.).
        """
        G = 6.67430e-11  # m^3 kg^-1 s^-2
        M_SUN = 1.98847e30  # kg
        R_SUN = 6.957e8  # m
        R_EARTH = 6.371e6
        rng = np.random.default_rng(random_state)

        # Pasar a unidades SI
        M_star = M_star_sun * M_SUN
        R_star = R_star_sun * R_SUN
        R_p = R_p_earth * R_EARTH

        # k = Rp/R*
        k = R_p / R_star

        # Muestras de T14
        if T14_err_days is not None and T14_err_days > 0:
            T14_samples_days = rng.normal(T14_days, T14_err_days, size=n_samples)
        else:
            T14_samples_days = np.full(n_samples, T14_days)

        # Filtramos duraciones no físicas (negativas o cero)
        T14_samples_days = T14_samples_days[T14_samples_days > 0]
        if len(T14_samples_days) == 0:
            raise ValueError("Todas las muestras de T14 son no físicas (<= 0).")

        # Si se reduce el número de muestras, ajustamos n_samples
        n_eff = len(T14_samples_days)

        # Muestras de b
        if geometric_b_prior:
            # p(b) ∝ b en [0, b_max]  -> b = b_max * sqrt(u)
            u = rng.random(n_eff)
            b_samples = b_max * np.sqrt(u)
        else:
            # Uniforme en [0, b_max]
            b_samples = rng.uniform(0, b_max, size=n_eff)

        # Convertimos T14 a segundos
        T14_samples_sec = T14_samples_days * 86400.0

        # Calculamos períodos usando la fórmula aproximada invertida
        # T14 = (P/pi) * (R*/a) * sqrt((1+k)^2 - b^2)
        # con a = (G M P^2 / (4 pi^2))^(1/3)
        # => P = [ (pi * T14) / (R* * sqrt((1+k)^2 - b^2) * (4 pi^2 / (G M))^(1/3)) ]^3

        # Factor común que no depende de P ni de b
        factor_kepler = (4 * np.pi ** 2) / (G * M_star)  # (1/s^2) / m^3
        factor_kepler_13 = factor_kepler ** (1.0 / 3.0)

        S = np.sqrt((1.0 + k) ** 2 - b_samples ** 2)  # sqrt((1+k)^2 - b^2)
        # Evitar valores no físicos cuando b > 1+k (por redondeos)
        valid = S > 0

        T14_valid = T14_samples_sec[valid]
        b_valid = b_samples[valid]
        S_valid = S[valid]

        num_valid = len(T14_valid)
        if num_valid == 0:
            raise RuntimeError("No quedan combinaciones físicas de T14 y b.")

        # C = R* * S * (4 pi^2 / (G M))^(1/3)
        C = R_star * S_valid * factor_kepler_13  # unidades: s^{-2/3}

        # P^(1/3) = (pi * T14) / C
        P13 = (np.pi * T14_valid) / C

        # P en segundos
        P_sec = P13 ** 3

        # Convertimos a días y filtramos valores positivos
        P_days = P_sec / 86400.0
        P_days = P_days[P_days > 0]

        if len(P_days) == 0:
            raise RuntimeError("No se obtuvo ningún período positivo.")

        # Estadísticos resumen
        median = np.median(P_days)
        p16, p84 = np.percentile(P_days, [16, 84])
        p5, p95 = np.percentile(P_days, [5, 95])

        summary = {
            "median_days": median,
            "p16_days": p16,
            "p84_days": p84,
            "p5_days": p5,
            "p95_days": p95,
            "n_samples": len(P_days),
        }

        return P_days, summary

    def object_dir(self):
        return self.object_dir
