import pandas as pd
import numpy as np

class AllesfitterDataExtractor:
    """
    Utility to extract fitted transit parameters from allesfitter/alesfitter nested sampling results.

    Provides static methods to read period, epoch, duration, depth, radius, semi-major axis,
    and planet name from the nested sampling output tables.
    """
    def __init__(self):
        """Initializes the data extractor. No setup required."""
        pass

    @staticmethod
    def extract_period(candidate_number: int, ns_table_results_df: pd.DataFrame, allesclass, percentile=68):
        """
        Extracts the orbital period for a candidate from the nested sampling results.

        Parameters
        ----------
        candidate_number : int
            The zero-based index of the candidate.
        ns_table_results_df : pandas.DataFrame
            The nested sampling table of fitted parameters.
        allesclass : alexfitter.allesclass
            The allesfitter analysis object containing posterior distributions.
        percentile : float
            The credible interval percentile (e.g. 68 for 1-sigma).

        Returns
        -------
        tuple of (float, float, float)
            The median period, lower error, and upper error.
        """
        period_row = ns_table_results_df[ns_table_results_df["#name"].str.contains("_period")].iloc[candidate_number]
        period = period_row["median"]
        period_distribution = allesclass.posterior_params[period_row["#name"]]
        period_low_err = period - np.percentile(period_distribution, 50 - percentile / 2)
        period_up_err = np.percentile(period_distribution, 50 + percentile / 2) - period
        return period, period_low_err, period_up_err

    @staticmethod
    def extract_epoch(candidate_number: int, ns_table_results_df: pd.DataFrame, allesclass, percentile=68):
        """
        Extracts the transit epoch (T0) for a candidate from the nested sampling results.

        Parameters
        ----------
        candidate_number : int
            The zero-based index of the candidate.
        ns_table_results_df : pandas.DataFrame
            The nested sampling table of fitted parameters.
        allesclass : alexfitter.allesclass
            The allesfitter analysis object containing posterior distributions.
        percentile : float
            The credible interval percentile.

        Returns
        -------
        tuple of (float, float, float)
            The median epoch, lower error, and upper error.
        """
        epoch_row = ns_table_results_df[ns_table_results_df["#name"].str.contains("_epoch")].iloc[candidate_number]
        epoch = epoch_row["median"].item()
        epoch_distribution = allesclass.posterior_params[epoch_row["#name"]]
        epoch_low_err = epoch - np.percentile(epoch_distribution, 50 - percentile / 2)
        epoch_up_err = np.percentile(epoch_distribution, 50 + percentile / 2) - epoch
        return epoch, epoch_low_err, epoch_up_err

    @staticmethod
    def extract_duration(candidate_number: int, ns_derived_table_results_df: pd.DataFrame):
        """
        Extracts the total transit duration for a candidate from the derived parameters table.

        Parameters
        ----------
        candidate_number : int
            The zero-based index of the candidate.
        ns_derived_table_results_df : pandas.DataFrame
            The nested sampling derived parameters table.

        Returns
        -------
        tuple of (float, float, float)
            The median duration (hours), lower error, and upper error.
        """
        duration_row = ns_derived_table_results_df[ns_derived_table_results_df["#property"].str.contains("Total transit duration")].iloc[candidate_number]
        duration = duration_row["value"].item()
        duration_low_err = float(duration_row["lower_error"])
        duration_up_err = float(duration_row["upper_error"])
        return duration, duration_low_err, duration_up_err

    @staticmethod
    def extract_depth(candidate_number: int, ns_derived_table_results_df: pd.DataFrame):
        """
        Extracts the transit depth (in ppt) for a candidate from the derived parameters table.

        Parameters
        ----------
        candidate_number : int
            The zero-based index of the candidate.
        ns_derived_table_results_df : pandas.DataFrame
            The nested sampling derived parameters table.

        Returns
        -------
        tuple of (float, float, float)
            The median depth in parts-per-thousand, lower error, and upper error.
        """
        depth_row = ns_derived_table_results_df[ns_derived_table_results_df["#property"].str.contains("depth \(dil.\)")].iloc[candidate_number]
        depth = depth_row["value"] * 1000
        depth_low_err = depth_row["lower_error"] * 1000
        depth_up_err = depth_row["upper_error"] * 1000
        return depth, depth_low_err, depth_up_err

    @staticmethod
    def extract_radius(candidate_number: int, ns_derived_table_results_df: pd.DataFrame):
        """
        Extracts the planet radius in Earth units for a candidate from the derived parameters table.

        Parameters
        ----------
        candidate_number : int
            The zero-based index of the candidate.
        ns_derived_table_results_df : pandas.DataFrame
            The nested sampling derived parameters table.

        Returns
        -------
        tuple of (float, float, float)
            The median radius in Earth radii, lower error, and upper error.
        """
        radius_row = ns_derived_table_results_df[ns_derived_table_results_df["#property"].str.contains("oplus")].iloc[candidate_number]
        radius = radius_row["value"]
        radius_low_err = radius_row["lower_error"]
        radius_up_err = radius_row["upper_error"]
        return radius, radius_low_err, radius_up_err

    @staticmethod
    def extract_semimajor_axis(candidate_number: int, ns_derived_table_results_df: pd.DataFrame):
        """
        Extracts the semi-major axis in AU for a candidate from the derived parameters table.

        Parameters
        ----------
        candidate_number : int
            The zero-based index of the candidate.
        ns_derived_table_results_df : pandas.DataFrame
            The nested sampling derived parameters table.

        Returns
        -------
        tuple of (float, float, float)
            The median semi-major axis in AU, lower error, and upper error.
        """
        a_row = ns_derived_table_results_df[ns_derived_table_results_df["#property"].str.contains("(AU)")].iloc[candidate_number]
        a = a_row["value"]
        a_low_err = a_row["lower_error"]
        a_up_err = a_row["upper_error"]
        return a, a_low_err, a_up_err

    @staticmethod
    def extract_planet_name(object_id: str, candidate_number: int, ns_table_results_df: pd.DataFrame):
        """
        Constructs a planet name from the object ID and fitted parameter name.

        Parameters
        ----------
        object_id : str
            The target object identifier.
        candidate_number : int
            The zero-based index of the candidate.
        ns_table_results_df : pandas.DataFrame
            The nested sampling table of fitted parameters.

        Returns
        -------
        str
            The constructed planet name (e.g. ``TIC123456_b``).
        """
        period_row = ns_table_results_df[ns_table_results_df["#name"].str.contains("_period")].iloc[candidate_number]
        name = object_id + "_" + period_row["#name"].replace("_period", "")
        return name