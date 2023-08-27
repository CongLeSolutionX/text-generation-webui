def get_alpha_value(alpha, base):
    '''
    Gets alpha_value from alpha_value and rope_freq_base
    '''
    return (base/10000.) ** (63/64.) if base > 0 else alpha


def get_rope_freq_base(alpha, base):
    '''
    Gets rope_freq_base from alpha_value and rope_freq_base
    '''
    return base if base > 0 else 10000 * alpha ** (64/63.)
