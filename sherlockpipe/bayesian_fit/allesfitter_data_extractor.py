class AllesfitterDataExtractor:
    def __init__(self):
        pass

    @staticmethod
    def extract_period(candidate_number: int, ns_table_results_df: pd.DataFrame, allesclass):
        period_row = ns_table_results_df[ns_table_results_df["#name"].str.contains("_period")].iloc[i]
        period = period_row["median"]
        period_distribution = allesclass.posterior_params[period_row["#name"]]
        period_low_err = period - np.percentile(period_distribution, 50 - percentile / 2)
        period_up_err = np.percentile(period_distribution, 50 + percentile / 2) - period
        return period, period_low_err, period_up_err

    @staticmethod
    def extract_epoch(candidate_number: int, ns_table_results_df: pd.DataFrame, allesclass):
        epoch_row = ns_table_results_df[ns_table_results_df["#name"].str.contains("_epoch")].iloc[i]
        epoch = epoch_row["median"].item()
        epoch_distribution = alles.posterior_params[epoch_row["#name"]]
        epoch_low_err = epoch - np.percentile(epoch_distribution, 50 - percentile / 2)
        epoch_up_err = np.percentile(epoch_distribution, 50 + percentile / 2) - epoch
        return epoch, epoch_low_err, epoch_up_err

    @staticmethod
    def extract_duration(candidate_number: int, ns_derived_table_results_df: pd.DataFrame):
        duration_row = ns_derived_table_results_df[ns_derived_table_results_df["#property"].str.contains("Total transit duration")].iloc[i]
        duration = duration_row["value"].item()
        duration_low_err = float(duration_row["lower_error"])
        duration_up_err = float(duration_row["upper_error"])
        return duration, duration_low_err, duration_up_err

    @staticmethod
    def extract_depth(candidate_number: int, ns_derived_table_results_df: pd.DataFrame):
        depth_row = ns_derived_table_results_df[ns_derived_table_results_df["#property"].str.contains("depth \(dil.\)")].iloc[i]
        depth = depth_row["value"] * 1000
        depth_low_err = depth_row["lower_error"] * 1000
        depth_up_err = depth_row["upper_error"] * 1000
        return depth, depth_low_err, depth_up_err

    @staticmethod
    def extract_radius(candidate_number: int, ns_derived_table_results_df: pd.DataFrame):
        radius_row = ns_derived_table_results_df[ns_derived_table_results_df["#property"].str.contains("oplus")].iloc[i]
        radius = radius_row["value"]
        radius_low_err = radius_row["lower_error"]
        radius_up_err = radius_row["upper_error"]
        return radius, radius_low_err, radius_up_err

    @staticmethod
    def extract_semimajor_axis(candidate_number: int, ns_derived_table_results_df: pd.DataFrame):
        a_row = ns_derived_table_results_df[ns_derived_table_results_df["#property"].str.contains("(AU)")].iloc[i]
        a = a_row["value"]
        a_low_err = a_row["lower_error"]
        a_up_err = a_row["upper_error"]
        return a, a_low_err, a_up_err

    @staticmethod
    def extract_planet_name(candidate_number: int, ns_table_results_df: pd.DataFrame):
        period_row = ns_table_results_df[ns_table_results_df["#name"].str.contains("_period")].iloc[i]
        name = object_id + "_" + period_row["#name"].replace("_period", "")
        return name