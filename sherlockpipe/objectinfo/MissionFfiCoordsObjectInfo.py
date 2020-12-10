from sherlockpipe.objectinfo.ObjectInfo import ObjectInfo

class MissionFfiCoordsObjectInfo(ObjectInfo):
    """
    Implementation of ObjectInfo to be used to characterize long-cadence objects from TESS by providing the RA and Dec.
    """
    def __init__(self, ra, dec, sectors, initial_mask=None, initial_detrend_period=None, star_info=None,
                 aperture_file=None):
        """
        @param ra: the objects right ascension.
        @param dec: the objects declination.
        @param sectors: an array of integers specifying which sectors will be analysed for the object
        @param initial_mask: an array of time ranges provided to mask them into the initial object light curve.
        @param initial_detrend_period: integer value specifying a fixed value for an initial period to be detrended
        @param star_info: input star information
        @param aperture_file: the file containing 1s and 0s specifying the user selected aperture
        from the initial light curve before processing.
        """
        super().__init__(initial_mask, initial_detrend_period, star_info, aperture_file)
        self.ra = ra
        self.dec = dec
        self.sectors = sectors

    def sherlock_id(self):
        return str(self.ra) + "_" + str(self.dec) + "_FFI_" + str(self.sectors)

    def mission_id(self):
        return None


