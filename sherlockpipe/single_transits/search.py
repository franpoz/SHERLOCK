import glob
import io
import logging
import os
import shutil
import zipfile
import matplotlib.pyplot as plt
import requests
from exoml.santo.SANTO import SANTO
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from watson.watson import Watson, SingleTransitProcessInput

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
        predictions, spectra, target_stats_df = SANTO().predict(self.object_dir, ['/lc.csv'],
                                                                f"{model_dir}/SANTO_model_chk.h5",
                                                                batch_size=self.batch_size, plot=True,
                                                                plot_positives=False,
                                                                tagged_data=False,
                                                                smooth=False, sherlock=True,
                                                                threshold=self.threshold)
        moriarty_dir = f"{self.object_dir}/moriarty" if len(self.search_candidates_df) == 0 else \
            f"{self.object_dir}/moriarty_{','.join(self.search_candidates_df['run'].astype(int).astype(str))}"
        if os.path.exists(moriarty_dir):
            shutil.rmtree(moriarty_dir)
        os.mkdir(moriarty_dir)
        shutil.move(f"{self.object_dir}/lc.csv.png", f"{moriarty_dir}/scores.png")
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
        np.savetxt(f"{moriarty_dir}/predictions.csv", predictions, delimiter=',')
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
            min_it_index = int(np.max([t0_index - half_duration_points]))
            max_it_index = int(np.min([t0_index + half_duration_points, len(time) - 1]))
            Watson.plot_single_transit(SingleTransitProcessInput(self.object_dir, self.object_id, 0, f"{self.object_dir}/lc.csv",
                                                                 f"{self.object_dir}/lc_data.csv", f"{self.object_dir}/tpfs",
                                                                 apertures, stats_row['t0'], 1 - stats_row['depth'],
                                                                 duration_points * cadence / 60, 10, 0.05, 10, None))
            single_transit_file_name = "single_transit_" + str(0) + "_T0_" + str(stats_row['t0']) + ".png"
            shutil.move(self.object_dir + '/' + single_transit_file_name,
                        f"{moriarty_dir}/{self.object_id}_t0_{stats_row['t0']}_st.png")
            min_large_index = int(np.max([0, t0_index - half_duration_points * 100]))
            max_large_index = int(np.min([t0_index + half_duration_points * 100, len(time) - 1]))
            self.plot_light_curve_broken_x(time[min_large_index:max_large_index], flux[min_large_index:max_large_index],
                                      half_duration_points, stats_row['t0'],
                                      cadence, stats_row, moriarty_dir, self.object_id,
                                      gap_threshold=0.5,
                                      max_panels=6,  # límite de paneles
                                      min_points_per_panel=200)
            fig, axs = plt.subplots(1, 1, figsize=(14, 7), constrained_layout=True)
            fig.suptitle(f"{stats_row['target_file']} T0={stats_row['t0']} Duration(h)={round(stats_row['duration_points'] * cadence / 3600, 2)} "
                         f"S/N={round(stats_row['snr'], 2)} Max Score={round(stats_row['max_score'], 2)}")
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
                color='red')
            #axs.plot(time[min_index:max_index], model_for_target[min_index:max_index], color='red')
            axs.set_xlabel("Time (TBJD)", size=30)
            axs.set_ylabel("Flux norm.", size=30)
            plt.savefig(f"{moriarty_dir}/{self.object_id}_t0_{stats_row['t0']}_focus.png", bbox_inches='tight')
            plt.close()
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
                              cadence, stats_row, moriarty_dir, object_id,
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
            f"{stats_row['target_file']} "
            f"T0={stats_row['t0']} "
            f"Duration(h)={round(stats_row['duration_points'] * cadence / 3600, 2)} "
            f"S/N={round(stats_row['snr'], 2)} "
            f"Max Score={round(stats_row['max_score'], 2)}"
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

        # Annotate real gaps between shown panels
        for k in range(len(segs) - 1):
            s_prev, e_prev, _ = segs[k]
            s_next, e_next, _ = segs[k + 1]
            gap = time[s_next] - time[e_prev]
            axes[k].text(axes[k].get_xlim()[1], yhi, f"gap≈{gap:.2f}",
                         ha="right", va="bottom", fontsize=9)

        out_path = f"{moriarty_dir}/{object_id}_t0_{stats_row['t0']}_all.png"
        plt.savefig(out_path, bbox_inches='tight')
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

    def nearest_epoch_offset(ts, t0, P, signed=True):
        """
        Devuelve la diferencia temporal entre ts y el t0 periódico más cercano.
        ts, t0, P en las mismas unidades (p.ej., días BJD_TDB).
        Acepta escalares o arrays (vectorizado).
        """
        n = np.rint((np.asarray(ts) - t0) / P).astype(int)
        dt = np.asarray(ts) - (t0 + n * P)
        return dt if signed else np.abs(dt)

    def object_dir(self):
        return self.object_dir
