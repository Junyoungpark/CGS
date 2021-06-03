import numpy as np


def sample_pore_diameters(num_pores: int,
                          max_radius: float = None,
                          dist_name: str = 'lognormal',
                          **kwargs):
    support_dists = ['lognormal', 'uniform']
    micro, milli, centi = 1e-6, 1e-3, 1e-2

    if dist_name == 'uniform' and max_radius is not None and 'high' in kwargs:
        if kwargs['high'] >= 2 * max_radius:
            kwargs['high'] = 2 * max_radius
            kwargs['low'] = 2 * max_radius * milli

    if dist_name == "lognormal":
        if 'sigma' not in kwargs:
            kwargs['sigma'] = milli
        if 'mean' not in kwargs:
            kwargs['mean'] = np.log(10 * micro)

        diameter = np.random.lognormal(size=num_pores,
                                       **kwargs)
    elif dist_name == "uniform":
        if 'low' not in kwargs:
            kwargs['low'] = 9.9 * micro
        if 'high' not in kwargs:
            kwargs['high'] = 10.1 * micro
        diameter = np.random.uniform(size=num_pores,
                                     **kwargs)
    else:
        raise RuntimeError("Please specify the distribution of the pore diameters from {}".format(support_dists))

    if max_radius is not None:
        diameter = diameter.clip(max=2 * max_radius)

    return diameter


if __name__ == "__main__":
    dias = sample_pore_diameters(1000, 'lognormal')
