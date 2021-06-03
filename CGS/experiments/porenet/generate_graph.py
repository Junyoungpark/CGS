import numpy as np
import ray

from CGS.experiments.porenet.graph_gen_utils.dgl_utils import pn_to_dgl
from CGS.experiments.porenet.graph_gen_utils.openopm_utils import (compute_min_pore_dist,
                                                                   generate_delaunay_pn,
                                                                   generate_voronoi_pn,
                                                                   get_geom,
                                                                   solve_pnm)
from CGS.experiments.porenet.graph_gen_utils.pore_diameter_sampling import sample_pore_diameters
from CGS.utils.mp_utils import chunks

_1BAR = 101325.0  # [Pascal (N/m^2)] # 101325 (=1Bar)
SAFEY_MARGIN = 0.90


def generate_graphs_seq(n_graphs, nS_bd, tessellation):
    nS = np.random.randint(nS_bd[0], nS_bd[1], size=n_graphs)
    gs = [generate_graph(num_pores=ns, tessellation=tessellation) for ns in nS]
    return gs


@ray.remote(num_cpus=1)
def remote_generate_graph(nss, tessellation):
    return [generate_graph(ns, tessellation) for ns in nss]


def generate_graphs(n_graphs: int,
                    nS_bd: list,
                    tessellation: str,
                    n_workers: int):
    nS = np.random.randint(nS_bd[0], nS_bd[1], size=n_graphs)
    nS_chunk = chunks(nS, n_workers)

    gs = [remote_generate_graph.remote(nss, tessellation) for nss in nS_chunk]
    gs = [g for sub_gs in ray.get(gs) for g in sub_gs]
    return gs


def generate_graph(num_pores: int,
                   tessellation: str = 'Voronoi',
                   rectangle_shape=[1e-3, 1e-3, 1e-3],
                   mu: float = 1e-5,  # dynamic viscosity [unit: Pa.m]
                   temperature: float = 298.0,  # [unit: K (Kelvin)] K - 273.15 => Celsius
                   pore_diameter: np.ndarray = None,
                   pore_diameter_dist: str = 'lognormal',
                   pore_dist_kwargs: dict = {'mean': np.log(1e-5), 'sigma': .1},
                   pressure_difference: float = _1BAR,  # [Pascal (N/m^2)] # 101325 (=1Bar)
                   nf_keys: list = ['coords', 'boundary', 'pressure_masked'],
                   ef_keys: list = ['diameter', 'length', 'k'],
                   verbose: bool = False):
    # Note: This code don't generate graph that has exactly 'num_pores'
    # number of nodes because of tessellation strategy.

    assert tessellation in ['Delaunay', 'Voronoi']

    while True:  # by chance, the Open PNM solver doesn't converges.
        try:
            if tessellation == 'Delaunay':
                pn = generate_delaunay_pn(num_pores, shape=rectangle_shape)
            else:  # Voronoi
                pn = generate_voronoi_pn(num_pores, shape=rectangle_shape)
            min_pore_dist = compute_min_pore_dist(pn)
            max_pore_radius = 0.5 * min_pore_dist * SAFEY_MARGIN

            if pore_diameter is None:
                pore_diameter = sample_pore_diameters(num_pores=pn.Np,
                                                      max_radius=max_pore_radius,
                                                      dist_name=pore_diameter_dist,
                                                      **pore_dist_kwargs)

            else:
                pore_diameter = pore_diameter.flatten()
                assert pore_diameter.size == pn.Np

            geom = get_geom(pn, pore_diameter)
            permeability, pressure = solve_pnm(pn=pn,
                                               geom=geom,
                                               temperature=temperature,
                                               mu=mu,
                                               pressure_difference=pressure_difference,
                                               domain_x_width=rectangle_shape[0],
                                               domain_y_width=rectangle_shape[1],
                                               domain_z_width=rectangle_shape[2],
                                               verbose=verbose)

            # get dgl graph
            g = pn_to_dgl(pn,
                          pressure=pressure,
                          mu=mu,
                          nf_keys=nf_keys,
                          ef_keys=ef_keys,
                          verbose=verbose)
            break
        except KeyboardInterrupt:
            break
        except:  # In case, the random graph has no (unique) solution.
            pass
    return g
