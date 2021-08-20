"""Utils."""


def sget(dct: dict, *keys):
    """Safe version of get.

    example usage:
        my_dict = {'a': [{'b': {'c': 'my_val'} } ] }

        sget(my_dict, 'a', 0, 'b', 'c')  # returns 'my_val'
        sget(my_dict, 'a', 0, 'b', 'c', 'd')  # returns None
    """
    dct = dct.copy()
    for key in keys:
        try:
            dct = dct[key]
        except (KeyError, IndexError):
            return None
    return dct
