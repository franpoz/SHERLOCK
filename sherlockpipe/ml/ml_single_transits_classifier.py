import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from lightkurve import TessTargetPixelFile
from sshkeyboard import listen_keyboard, stop_listening


class MLSingleTransitsClassifier:
    def __init__(self) -> None:
        super().__init__()

    def load_candidate_single_transits(self, training_data_dir, inner_dir):
        single_transits_dir = training_data_dir + "/single_transits/"
        single_transits_inner_dir = single_transits_dir + inner_dir
        if not os.path.exists(single_transits_dir):
            os.mkdir(single_transits_dir)
        if not os.path.exists(single_transits_inner_dir):
            os.mkdir(single_transits_inner_dir)
        files = os.listdir(single_transits_inner_dir)
        files_to_process = os.listdir(training_data_dir + "/" + inner_dir)
        files_to_process.sort()
        if len(files) > 0:
            files.sort()
            last_file = files[-1]
            file_name_matches = re.search("(TIC [0-9]+)", last_file)
            target = file_name_matches[1]
            target_index = files_to_process.index(target) + 1
        else:
            target_index = 0
        files_to_process = files_to_process[target_index:]
        for file in files_to_process:
            target_dir = training_data_dir + "/" + inner_dir + "/" + file
            tpfs_short_dir = target_dir + "/tpfs_short/"
            if not os.path.exists(tpfs_short_dir):
                continue
            ts_short = pd.read_csv(target_dir + "/time_series_short.csv")
            ois = pd.read_csv(target_dir + "/ois.csv")
            tpfs_short = []
            for tpf_file in os.listdir(tpfs_short_dir):
                tpfs_short.append(TessTargetPixelFile(tpfs_short_dir + "/" + tpf_file))
            for oi in ois.iterrows():
                initial_t0 = oi[1]["t0"]
                duration = oi[1]["duration"] / 24 * 2
                period = oi[1]["period"]
                transit = 0
                for t0 in np.arange(initial_t0, ts_short["time"].max(), period):
                    fig, axs = plt.subplots(1, 1, figsize=(16, 16), constrained_layout=True)
                    tpf_short_framed = None
                    for tpf in tpfs_short:
                        if tpf.time[0].value < t0 and tpf.time[-1].value > t0:
                            tpf_short_framed = tpf[(tpf.time.value > t0 - duration) & (tpf.time.value < t0 + duration)]
                            if len(tpf_short_framed) == 0:
                                break
                            tpf_short_framed.plot_pixels(axs, aperture_mask=tpf_short_framed.pipeline_mask)
                            break
                    if tpf_short_framed is None or len(tpf_short_framed) == 0:
                        continue
                    fig.suptitle("Single Transit Analysis")
                    plt.show()
                    fig, axs = plt.subplots(4, 1, figsize=(16, 16), constrained_layout=True)
                    ts_short_framed = ts_short[(ts_short["time"] > t0 - duration) & (ts_short["time"] < t0 + duration)]
                    axs[0].scatter(ts_short_framed["time"], ts_short_framed["centroids_x"].to_numpy(), color="black")
                    axs[0].scatter(ts_short_framed["time"], ts_short_framed["motion_x"].to_numpy(), color="red")
                    axs[1].scatter(ts_short_framed["time"], ts_short_framed["centroids_y"].to_numpy(), color="black")
                    axs[1].scatter(ts_short_framed["time"], ts_short_framed["motion_y"].to_numpy(), color="red")
                    axs[2].scatter(ts_short_framed["time"], ts_short_framed["background_flux"].to_numpy(), color="blue")
                    axs[3].scatter(ts_short_framed["time"], ts_short_framed["flux"].to_numpy(), color="blue")
                    fig.suptitle("Single Transit Analysis")
                    plt.show()
                    selection = None

                    def press(key):
                        print(f"'{key}' pressed")
                        global selection
                        if key == "0":
                            selection = 0.0
                        elif key == "1":
                            selection = 0.25
                        elif key == "2":
                            selection = 0.5
                        elif key == "3":
                            selection = 0.75
                        elif key == "4":
                            selection = 1.0
                        if selection is not None:
                            single_transit_path = single_transits_inner_dir + "/" + file + "/S" + str(
                                transit) + "_" + str(
                                selection)
                            pathlib.Path(single_transit_path).mkdir(parents=True, exist_ok=True)
                            ts_short_framed.to_csv(single_transit_path + "/ts_short_framed.csv")
                            tpf_short_framed.to_fits(single_transit_path + "/tpf_short_framed.fits", True)
                            stop_listening()

                    listen_keyboard(on_press=press)
                    transit = transit + 1