"""Compatibility helpers for loading older alexfitter/dynesty results."""

import alexfitter


def load_alexfitter_results(data_dir):
    """Load an alexfitter result directory across dynesty result formats."""
    try:
        from dynesty.utils import Results
    except ImportError:
        return alexfitter.allesclass(data_dir)

    original_setitem = getattr(Results, "__setitem__", None)

    def setitem(result, key, value):
        keys = getattr(result, "_keys", None)
        if keys is None:
            keys = []
            object.__setattr__(result, "_keys", keys)
        if key not in keys:
            keys.append(key)
        object.__setattr__(result, key, value)

    Results.__setitem__ = setitem
    try:
        return alexfitter.allesclass(data_dir)
    finally:
        if original_setitem is None:
            del Results.__setitem__
        else:
            Results.__setitem__ = original_setitem
