import foldedleastsquares
import keras
import numpy as np
import pandas as pd


class LcDataInputGenerator(keras.utils.Sequence):

    def __init__(self, lc_filenames, labels, batch_size, lc_len, mode="short"):
        self.lc_filenames = lc_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.lc_len = lc_len

    def __len__(self):
        return (np.ceil(len(self.lc_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.lc_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        # TODO inputs: flux (2500, 7)
        # TODO inputs: centroidsbck (2500, 5)

        # TODO inputs: global centroids bck  (2500, 5)
        # TODO inputs: glbal flux (2500, 7)
        # TODO inputs: focus odd harmonic centroids (500, 5)
        # TODO inputs: focus odd subharmonic centroids (500, 5)
        # TODO inputs: focus odd centroids (500, 5)
        # TODO inputs: focus even harmonic centroids (500, 5)
        # TODO inputs: focus even subharmonic centroids (500, 5)
        # TODO inputs: stellar model (6, 1)
        # TODO inputs: stellar model (6, 1)
        # TODO inputs: stellar model (6, 1)
        # TODO inputs: stellar model (6, 1)
        # TODO inputs: stellar model (6, 1)
        # TODO inputs: stellar model (6, 1)
        for lc_filename in self.lc_filenames:
            star_df = pd.read_csv(lc_filename + "/params_star.csv")
            star_input = [star_df.iloc[0]["h"], star_df.iloc[0]["j"], star_df.iloc[0]["k"], star_df.iloc[0]["v"],
                          star_df.iloc[0]["R_star"], star_df.iloc[0]["M_star"], star_df.iloc[0]["Teff_star"],
                          star_df.iloc[0]["logg"], star_df.iloc[0]["feh"],
                          star_df.iloc[0]["ld_a"], star_df.iloc[0]["ld_b"]]
            # TODO normalize star_input values
            # TODO check for empty star_input values and select replacement
            flux_input = []
            lc_data = pd.read_csv(lc_filename + "/lc_data_" + self.mode + ".csv")
            if "_tp_" in lc_filename:
                ois_df = pd.read_csv(lc_filename + "/ois.csv")
                oi_index = int(lc_filename.split("_tp_")[1])
                period = ois_df.iloc[oi_index]["period"]
                t0 = ois_df.iloc[oi_index]["t0"]
            elif "_fp_" in lc_filename:
                ois_df = pd.read_csv(lc_filename + "/ois.csv")
                oi_index = int(lc_filename.split("_fp_")[1])
                period = ois_df.iloc[oi_index]["period"]
                t0 = ois_df.iloc[oi_index]["t0"]
            else:
                min_time = np.nanmin(lc_data["time"])
                period = np.random.uniform(size=1, low=0.5, high=20)
                t0 = np.random.uniform(size=1, low=min_time, high=min_time + period)
            lc_data["time_folded_period"] = foldedleastsquares.fold(lc_data["time"].to_numpy(), period, T0=t0)
            lc_data["time_folded_period"] = foldedleastsquares.fold(lc_data["time"].to_numpy(), period, T0=t0)
            lc_data["time_folded_period"] = foldedleastsquares.fold(lc_data["time"].to_numpy(), period, T0=t0)
            lc_data["time_folded_period"] = foldedleastsquares.fold(lc_data["time"].to_numpy(), period, T0=t0)
            lc_data = lc_data.sort_values(by=['time'], ascending=True)
            # Discover, visualize, and preprocess data using pandas if needed.

            data = data.to_numpy()
        return np.array([
            resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
            for file_name in batch_x]) / 255.0, np.array(batch_y)