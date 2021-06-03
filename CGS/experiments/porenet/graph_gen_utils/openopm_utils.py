import timeit

import networkx as nx
import numpy as np
import openpnm as op
from openpnm.utils import logging

logger = logging.getLogger(__name__)


def pn_to_networkx(network):
    G = nx.Graph()
    # Extracting node list and connectivity matrix from Network
    nodes = map(int, network.Ps)
    conns = network['throat.conns']

    # Explicitly add nodes and connectivity matrix
    G.add_nodes_from(nodes)
    G.add_edges_from(conns)
    return G


def generate_cubic_pn(num_pores_x: int,
                      num_pores_y: int,
                      num_pores_z: int,
                      pore_spacing: float = 1e-4):
    shape = [num_pores_x, num_pores_y, num_pores_z]
    pn = op.network.Cubic(shape=shape, spacing=pore_spacing)
    return pn


def generate_voronoi_pn(num_pores: int,
                        shape=[1.0, 1.0, 1.0]):
    pn = op.network.Voronoi(shape=shape, num_points=num_pores)
    return pn


def generate_delaunay_pn(num_pores: int,
                         shape=[1.0, 1.0, 1.0]):
    pn = op.network.Delaunay(shape=shape, num_points=num_pores)
    return pn


def get_geom(pn, pore_diameter):
    geom = op.geometry.GenericGeometry(network=pn, pores=pn.Ps, throats=pn.Ts)
    assert pore_diameter.size == pn.Np

    geom['pore.diameter'] = pore_diameter  # Np = number of pores
    pore_ij = pn['throat.conns']  # num_throats x 2 list of pores at the end of each throat
    pore_diameters_ij = geom['pore.diameter'][pore_ij]
    throat_diameters = np.amin(pore_diameters_ij, axis=1)
    geom['throat.diameter'] = throat_diameters

    pore_radii = 0.5 * geom['pore.diameter']
    geom['pore.volume'] = (4. / 3.) * np.pi * (pore_radii) ** 3

    pore_radii_ij = pore_radii[pn['throat.conns']]
    src_coord = pn['pore.coords'][pore_ij][:, 0, :]
    dst_coord = pn['pore.coords'][pore_ij][:, 1, :]
    throat_length = np.linalg.norm(src_coord - dst_coord, axis=-1) - np.sum(pore_radii_ij, axis=1)
    geom['throat.length'] = throat_length

    throat_radii = 0.5 * geom['throat.diameter']
    throat_lengths = geom['throat.length']
    geom['throat.volume'] = np.pi * (throat_radii) ** 2 * throat_lengths  # throats are assumed to be cylinders
    return geom


def solve_pnm(pn, geom, temperature, mu, pressure_difference,
              domain_x_width, domain_y_width, domain_z_width,
              verbose=False):
    water = op.phases.GenericPhase(network=pn)
    water['pore.temperature'] = temperature
    water['pore.viscosity'] = mu

    throat_radii = 0.5 * geom['throat.diameter']
    throat_lengths = geom['throat.length']

    phys_water = op.physics.GenericPhysics(network=pn, phase=water, geometry=geom)
    phys_water['throat.hydraulic_conductance'] = np.pi * throat_radii ** 4 / (8 * mu * throat_lengths)

    # Solve the flow problem
    # Set a physics solver based on Stokes' flow
    # (Stokes flow: simplified version of Navier-Stokes equation ignoring inertial effects)
    physics_solver = op.algorithms.StokesFlow(network=pn)
    physics_solver.setup(phase=water)

    # Give boundary conditions
    BC1_pores = pn.pores('front')
    physics_solver.set_value_BC(values=pressure_difference, pores=BC1_pores)
    BC2_pores = pn.pores('back')
    physics_solver.set_value_BC(values=0, pores=BC2_pores)

    # Solve the problem
    st = timeit.default_timer()
    physics_solver.run()
    et = timeit.default_timer()

    if verbose:
        print("Solving time : {}".format(et - st))

    # Compute the overall effective permeability
    Q = physics_solver.rate(pores=pn.pores('front'))
    A = domain_x_width * domain_y_width  # assuming rectangular network, cross-sectional area
    L = domain_z_width  # Length of flow path
    permeability = Q * mu * L / (A * pressure_difference)  # Overall (effective) permeability [m^2]
    pressure = physics_solver.results()['pore.pressure']  # Per node pressures
    return permeability, pressure


def compute_min_pore_dist(pn):
    pore_ij = pn['throat.conns']
    src_coord = pn['pore.coords'][pore_ij][:, 0, :]
    dst_coord = pn['pore.coords'][pore_ij][:, 1, :]
    dists = np.linalg.norm(src_coord - dst_coord, axis=-1)
    return dists.min()
