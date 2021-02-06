from contextlib import contextmanager
from timeit import default_timer
from experimental import sherlock_explorer


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: str(default_timer() - start)
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: str(end - start)

# tpl = tp.TransitsPipeline()
# tpl.load_tois()\
#     .filter_hj_tois()\
#     .limit_tois(0, 5)\
#     .retrieve_pfs_list()\
#     .compute_lcs()\
#     .flatten_lcs()\
#     .transit_adjusts()

# TODO bin the lightcurve so noise gets smooth?
# TODO subtract transit instead of masking?
# TODO reduce scattering: regression lightkurve (is it already done i PDCSAP?), different bitmask
# TODO ignore subarmonics of max period? maybe we can edit TLS to allow period ranges
# TODO modify TLS so we can extract all relevant transits in 1 run per detrend
#

with elapsed_timer() as elapsed:
    sherlock = sherlock_explorer.SherlockExplorer()
    sherlock.explore_object("TIC 120609760", auto_detrend_periodic_signals=False, smooth=True, sectors=None)

    # sherlock = sherlock_class.Sherlock(n_detrends=3, auto_detrend_periodic_signals=True,
    #                                    mission_ids=["TIC 277539431"], mask_mode="subtract", id_lc={"TIC 277539431": 'sinteticos/modeled3.csv'})
    # sherlock.run()

        #.filter_hj_tois().limit_tois(2, 5).run() \
        #.filter_tois(lambda tois: tois.sort_values(by=['TIC ID', 'TOI'])) \
        #.filter_tois(lambda tois: tois[tois["TIC ID"].isin([149603524])])\
        #.run()
        #.filter_tois(lambda tois: tois.sort_values(by=['TIC ID', 'TOI']))\
        #.filter_tois(lambda tois: tois[tois["TIC ID"].isin([231912935])])\
        #.run()
    # for i in range(1, 6):
    #     sherlock = sherlock_class.Sherlock(n_detrends=6, auto_detrend_periodic_signals=True, snr_min=4,
    #                                        mission_ids=["TIC 277539431"], mask_mode="mask",
    #                                        id_lc={"TIC 277539431": 'sinteticos/modeled' + str(i) + '.csv'},
    #                                        auto_detrend_ratio=1/4, periodic_detrend_method="cosine")
    #     sherlock.run()
    #     shutil.move("TIC 277539431", "sinteticos/cosine_mask/TIC 277539431_" + str(i) + "_1-4")
    # for i in range(1, 6):
    #     sherlock = sherlock_class.Sherlock(n_detrends=6, auto_detrend_periodic_signals=True, snr_min=4,
    #                                        mission_ids=["TIC 277539431"], mask_mode="mask",
    #                                        id_lc={"TIC 277539431": 'sinteticos/modeled' + str(i) + '.csv'},
    #                                        auto_detrend_ratio=1/2, periodic_detrend_method="cosine")
    #     sherlock.run()
    #     shutil.move("TIC 277539431", "sinteticos/cosine_mask/TIC 277539431_" + str(i) + "_1-2")
    # for i in range(1, 6):
    #     sherlock = sherlock_class.Sherlock(n_detrends=6, auto_detrend_periodic_signals=True, snr_min=4,
    #                                        mission_ids=["TIC 277539431"],
    #                                        id_lc={"TIC 277539431": 'sinteticos/modeled' + str(i) + '.csv'},
    #                                        auto_detrend_ratio=1/4, periodic_detrend_method="biweight")
    #     sherlock.run()
    #     shutil.move("TIC 277539431", "sinteticos/bw/TIC 277539431_" + str(i) + "_1-4")
    # for i in range(1, 6):
    #     sherlock = sherlock_class.Sherlock(n_detrends=6, auto_detrend_periodic_signals=True, snr_min=4,
    #                                        mission_ids=["TIC 277539431"],
    #                                        id_lc={"TIC 277539431": 'sinteticos/modeled' + str(i) + '.csv'},
    #                                        auto_detrend_ratio=1/2, periodic_detrend_method="biweight")
    #     sherlock.run()
    print("Analysis took " + elapsed() + "s")
