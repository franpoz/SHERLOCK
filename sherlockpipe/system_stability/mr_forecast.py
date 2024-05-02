import logging
from os import path
from pathlib import Path

import numpy as np
from lcbuilder.helper import LcbuilderHelper
from scipy.stats import truncnorm, norm
import h5py
from astropy import units as u


resources_dir = path.join(path.dirname(__file__))
resources_dir = str(Path(resources_dir).parent.absolute())

class MrForecast:
    N_POPS: int = 4
    ## boundary
    M_LOWER: float = 3e-4
    M_UPPER: float = 3e5

    @staticmethod
    def read_parameters():
        ## read parameter file
        hyper_file = resources_dir + '/resources/mr_forecaster/fitting_parameters.h5'
        h5 = h5py.File(hyper_file, 'r')
        all_hyper = h5['hyper_posterior'][:]
        h5.close()
        return all_hyper

    ##############################################

    @staticmethod
    def Mpost2R(mass: float, classify='No'):
        """
        Forecast the Radius distribution given the mass distribution.

        Parameters
        ---------------
        mass: one dimensional array
            The mass distribution in earth radii
        classify: string (optional)
            If you want the object to be classifed.
            Options are 'Yes' and 'No'. Default is 'No'.
            Result will be printed, not returned.

        Returns
        ---------------
        radius: one dimensional array
            Predicted radius distribution in the input unit.
        """

        # mass input
        mass = np.array(mass)
        assert len(mass.shape) == 1, "Input mass must be 1-D."

        # mass range
        if np.min(mass) < 3e-4 or np.max(mass) > 3e5:
            logging.error('Mass range out of model expectation. Returning None.')
            return None

        ## convert to radius
        sample_size = len(mass)
        logm = np.log10(mass)
        prob = np.random.random(sample_size)
        logr = np.ones_like(logm)
        all_hyper = MrForecast.read_parameters()
        hyper_ind = np.random.randint(low=0, high=np.shape(all_hyper)[0], size=sample_size)
        hyper = all_hyper[hyper_ind, :]

        if classify == 'Yes':
            MrForecast._classification(logm, hyper[:, -3:])

        for i in range(sample_size):
            logr[i] = MrForecast._piece_linear(hyper[i], logm[i], prob[i])

        radius_sample = 10. ** logr

        return radius_sample


    @staticmethod
    def Mstat2R(mean: float, std: float, sample_size: int = 1000, classify='No'):
        """
        Forecast the mean and standard deviation of radius given the mena and standard deviation of the mass.
        Assuming normal distribution with the mean and standard deviation truncated at the mass range limit of the model.

        Parameters
        ---------------
        mean: float
            Mean (average) of mass in earth radii
        std: float
            Standard deviation of mass.
        sample_size: int (optional)
            Number of mass samples to draw with the mean and std provided.
        Returns
        ---------------
        mean: float
            Predicted mean of radius in the input unit.
        std: float
            Predicted standard deviation of radius.
        """
        # draw samples
        mass = truncnorm.rvs((MrForecast.M_LOWER - mean) / std, (MrForecast.M_UPPER - mean) / std, loc=mean, scale=std, size=sample_size)
        if classify == 'Yes':
            radius = MrForecast.Mpost2R(mass, classify='Yes')
        else:
            radius = MrForecast.Mpost2R(mass)

        r_med = np.median(radius)
        onesigma = 34.1
        r_up = np.percentile(radius, 50. + onesigma, interpolation='nearest')
        r_down = np.percentile(radius, 50. - onesigma, interpolation='nearest')

        return r_med, r_up - r_med, r_med - r_down


    @staticmethod
    def Rpost2M(radius: float, grid_size: int = 1e3, classify='No'):
        """
        Forecast the mass distribution given the radius distribution.

        Parameters
        ---------------
        radius: one dimensional array
            The radius distribution in earth radii
        grid_size: int (optional)
            Number of grid in the mass axis when sampling mass from radius.
            The more the better results, but slower process.
        classify: string (optional)
            If you want the object to be classifed.
            Options are 'Yes' and 'No'. Default is 'No'.
            Result will be printed, not returned.

        Returns
        ---------------
        mass: one dimensional array
            Predicted mass distribution in the input unit.
        """
        # radius range
        if np.min(radius) < 1e-1 or np.max(radius) > 1e2:
            logging.error('Radius range out of model expectation. Returning None.')
            return None

        # sample_grid
        if grid_size < 10:
            logging.error('The sample grid is too sparse. Using 10 sample grid instead.')
            grid_size = 10

        ## convert to mass
        sample_size = len(radius)
        logr = np.log10(radius)
        logm = np.ones_like(logr)

        all_hyper = MrForecast.read_parameters()
        hyper_ind = np.random.randint(low=0, high=np.shape(all_hyper)[0], size=sample_size)
        hyper = all_hyper[hyper_ind, :]

        logm_grid = np.linspace(-3.522, 5.477, int(grid_size))

        for i in range(sample_size):
            prob = MrForecast._ProbRGivenM(logr[i], logm_grid, hyper[i, :])
            logm[i] = np.random.choice(logm_grid, size=1, p=prob)

        mass_sample = 10. ** logm

        if classify == 'Yes':
            MrForecast._classification(logm, hyper[:, -3:])

        return mass_sample


    @staticmethod
    def Rstat2M(mean: float, std: float, sample_size: int = 1e3, grid_size: int = 1e3, classify='No'):
        """
        Forecast the mean and standard deviation of mass given the mean and standard deviation of the radius.

        Parameters
        ---------------
        mean: float
            Mean (average) of radius in earth radii
        std: float
            Standard deviation of radius.
        sample_size: int (optional)
            Number of radius samples to draw with the mean and std provided.
        grid_size: int (optional)
            Number of grid in the mass axis when sampling mass from radius.
            The more the better results, but slower process.
        Returns
        ---------------
        mean: float
            Predicted mean of mass in the input unit.
        std: float
            Predicted standard deviation of mass.
        """
        sample_size = int(sample_size)
        grid_size = int(grid_size)
        # draw samples
        radius = truncnorm.rvs((0. - mean) / std, np.inf, loc=mean, scale=std, size=sample_size)
        if classify == 'Yes':
            mass = MrForecast.Rpost2M(radius, grid_size, classify='Yes')
        else:
            mass = MrForecast.Rpost2M(radius, grid_size)
        if mass is None:
            return None
        m_med = np.median(mass)
        onesigma = 34.1
        m_up = np.percentile(mass, 50. + onesigma, interpolation='nearest')
        m_down = np.percentile(mass, 50. - onesigma, interpolation='nearest')
        return m_med, m_up - m_med, m_med - m_down

    @staticmethod
    def _indicate(M, trans, i):
        '''
        indicate which M belongs to population i given transition parameter
        '''
        ts = np.insert(np.insert(trans, MrForecast.N_POPS - 1, np.inf), 0, -np.inf)
        ind = (M >= ts[i]) & (M < ts[i + 1])
        return ind

    @staticmethod
    def _split_hyper_linear(hyper):
        '''
        split hyper and derive c
        '''
        c0, slope, sigma, trans = \
            hyper[0], hyper[1:1 + MrForecast.N_POPS], hyper[1 + MrForecast.N_POPS:1 + 2 * MrForecast.N_POPS], hyper[1 + 2 * MrForecast.N_POPS:]

        c = np.zeros_like(slope)
        c[0] = c0
        for i in range(1, MrForecast.N_POPS):
            c[i] = c[i - 1] + trans[i - 1] * (slope[i - 1] - slope[i])

        return c, slope, sigma, trans

    @staticmethod
    def _piece_linear(hyper, M, prob_R):
        '''
        model: straight line
        '''
        c, slope, sigma, trans = MrForecast._split_hyper_linear(hyper)
        R = np.zeros_like(M)
        for i in range(4):
            ind = MrForecast._indicate(M, trans, i)
            mu = c[i] + M[ind] * slope[i]
            R[ind] = norm.ppf(prob_R[ind], mu, sigma[i])

        return R

    @staticmethod
    def _ProbRGivenM(radii, M, hyper):
        '''
        p(radii|M)
        '''
        c, slope, sigma, trans = MrForecast._split_hyper_linear(hyper)
        prob = np.zeros_like(M)

        for i in range(4):
            ind = MrForecast._indicate(M, trans, i)
            mu = c[i] + M[ind] * slope[i]
            sig = sigma[i]
            prob[ind] = norm.pdf(radii, mu, sig)

        prob = prob / np.sum(prob)

        return prob

    @staticmethod
    def _classification(logm, trans):
        '''
        classify as four worlds
        '''
        count = np.zeros(4)
        sample_size = len(logm)

        for iclass in range(4):
            for isample in range(sample_size):
                ind = MrForecast._indicate(logm[isample], trans[isample], iclass)
                count[iclass] = count[iclass] + ind

        prob = count / np.sum(count) * 100.
        logging.info(f'Terran {prob[0]} %%, Neptunian {prob[1]} %%, Jovian {prob[2]} %%, Star {prob[3]} %%')


