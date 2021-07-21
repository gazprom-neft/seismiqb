""" Functions for generation of 2d and 3d synthetic seismic arrays.
"""
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import ricker
from numba import njit


def make_surfaces(num_surfaces, grid_shape, shape, kind='cubic', perturbation_share=0.25, shares=None,
                  rng=None, seed=None):
    """ Make arrays representing heights of surfaces in a 3d/2d-array.

    Parameters
    ----------
    num_surfaces : int
        The number of resulting surfaces.
    grid_shape : tuple
        Shape of a grid of points used for interpolating surfaces.
    shape : tuple
        Shape of a 3d/2d array inside which the surfaces are created.
    kind : str
        Surfaces are interpolated from values on the grid of points. This is the type of interpolation
        to use (see `scipy.interpolate.intepr1d` for all possible options).
    perturbation_share : float
        Maximum allowed surface-perturbation w.r.t. the distance between subsequent surfaces.
    shares : np.ndarray
        Array representing height-distances between subsequent surfaces as shares of unit-interval.
    rng : np.random.Generator or None
        generator of random numbers.
    seed : int or None
        seed used for creation of random generator (check out `np.random.default_rng`).

    Returns
    -------
    np.ndarray
        Array of size num_surfaces X shape[:2] representing resulting surfaces-heights.
    """
    rng = rng or np.random.default_rng(seed)

    # check shapes and select interpolation-method
    grid_shape = (grid_shape, ) if isinstance(grid_shape, int) else grid_shape
    if len(shape) != len(grid_shape) + 1:
        raise ValueError("`(len(shape) - 1)` should be equal to `len(grid_shape)`.")

    if len(shape) == 2:
        interp = interp1d
    elif len(shape) == 3:
        interp = interp2d
    else:
        raise ValueError('The function only supports the generation of 1d curves and 2d-surfaces.')

    # make the grid
    grid = [np.linspace(0, 1, num_points) for num_points in grid_shape]

    # make the first curve
    curves = [np.zeros(grid_shape)]
    shares = shares if shares is not None else np.ones((num_surfaces, ))
    shares = np.array(shares) / np.sum(shares)
    for delta_h in shares:
        epsilon = perturbation_share * delta_h

        # make each curve in unit-terms
        curves.append(curves[-1] + delta_h * np.ones_like(curves[0])
                      + rng.uniform(low=-epsilon, high=epsilon, size=curves[0].shape))

    # interpolate and scale each curve to cube-shape
    results = []
    for curve in curves:
        func = interp(*grid, curve, kind=kind)
        results.append((func(*[np.arange(num_points) / num_points for num_points in shape[:-1]])
                        * shape[-1]).astype(np.int).T)
    return np.array(results)


def reflectivity(v, rho):
    """ Compute reflectivity coefficients given velocity and density models.
    Velocities and reflectivity coefficients can be either 2d or 3d.
    """
    rc = np.zeros_like(v)
    v_rho = v * rho
    rc[..., 1:] = (v_rho[..., 1:] - v_rho[..., :-1]) / (v_rho[..., 1:] + v_rho[..., :-1])
    return rc

@njit
def convolve_2d(array, kernel):
    """ Shape-preserving vector-wise convolution of a 2d-array with a kernel-vector.
    """
    # calculate offsets to trim arrays resulting from the convolution
    result = np.zeros_like(array)
    left_offset = (len(kernel) - 1) // 2
    right_offset = len(kernel) - 1 - left_offset

    for i in range(array.shape[0]):
        result[i, :] = np.convolve(array[i], kernel)[left_offset:-right_offset]
    return result


@njit
def convolve_3d(array, kernel):
    """ Shape-preserving vector-wise convolution of a 3d-array with a kernel-vector.
    """
    # calculate offsets to trim arrays resulting from the convolution
    result = np.zeros_like(array)
    left_offset = (len(kernel) - 1) // 2
    right_offset = len(kernel) - 1 - left_offset

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            result[i, j, :] = np.convolve(array[i, j], kernel)[left_offset:-right_offset]
    return result


@njit
def make_colors_array_2d(colors, levels, shape):
    """ Color 2d-array in colors according to given levels.
    """
    array = np.zeros(shape=shape)
    for i in range(array.shape[0]):
        vec = array[i, :]
        for j, color in enumerate(colors):
            low = np.minimum(levels[j][i], array.shape[-1])
            vec[low : ] = color
    return array


@njit
def make_colors_array_3d(colors, levels, shape):
    """ Color 3d-array in colors according to given levels.
    """
    array = np.zeros(shape=shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            vec = array[i, j, :]
            for k, color in enumerate(colors):
                low = np.minimum(levels[k][i, j], array.shape[-1])
                vec[low : ] = color
    return array


def make_coords_shift(rng=None, seed=None, n_points=10, zeros_share=0.6, kind='cubic', perturb_values=True,
                      perturb_peak=True, peak_value=0.05, random_invert=True):
    """ Make randomized map [0, 1] -> [0, 1] to use it later as a coordinate-shift.
    """
    rng = rng or np.random.default_rng(seed)
    xs = np.linspace(0, 1, n_points)

    # make zeros-containing postfix and prefix
    n_zeros = int(n_points * zeros_share)
    if perturb_peak:
        delta_position = np.random.randint(-n_zeros // 4, n_zeros // 4 + 1)
    else:
        delta_position = 0
    zeros_prefix = [0] * (n_zeros // 2 + delta_position)
    zeros_postfix = [0] * (n_zeros - len(zeros_prefix))

    # form the values-hump and perturb it if needed
    n_values = n_points - n_zeros
    half = np.linspace(0, peak_value, n_values // 2 + 1 + n_values % 2)[1:]
    values = half.tolist() + half.tolist()[::-1][n_values % 2:]
    if perturb_values:
        step = 2 * peak_value / (n_values)
        values = (rng.uniform(-step / 2, step / 2, (n_values, )) + np.array(values)).tolist()
    spl = interp1d(xs, zeros_prefix + values + zeros_postfix, kind=kind)

    # possibly invert and fetch the coordinates shift
    if random_invert:
        if rng.choice([True, False]):
            return lambda x: x - spl(x)
    return lambda x: x + spl(x)

class SyntheticGenerator():
    """ Class for generation of syhthetic velocity and density models and synthetic seismic - 2D/3D.
    """
    def __init__(self, rng=None, seed=None):
        self.dim = None
        self.rng = rng or np.random.default_rng(seed)
        self.velocities = None
        self.velocity_model = None
        self.density_model = None
        self.synthetic = None
        self._curves = None
        self._horizon_heights = None

    def make_velocities(self, num_reflections=200, vel_limits=(900, 5400), horizon_heights=(1/4, 1/2, 2/3),
                        horizon_multipliers=(7, 5, 4)):
        """ Generate and store array of velocities. Roughly speaking, seismic slide is a stack of layers of constant
        velocities. This method generates the array of velocity-values, that are to be used later for making of
        velocity model.

        Parameters
        ----------
        num_reflections : int
            The number of reflective surfaces.
        vel_limits : sequence
            Contains two floats. Velocities of layers in velocity model gradually change from the
            lower limit (first number) to the upper limit (second number) with some noise added.
        horizon_heights : sequence
            Each element is a float in [0, 1] interval and defines the depth at which a horizon should be located.
            Note that each horizon is yielded by a large velocity gradient.
        horizon_multipliers : sequence
            Each element is float mutiplier >> 1 (or << -1) controling the magnitide of gradients in velocity.
            The larger the gradients, the more prominient are the horizons. The argument should have the same length
            as `horizon_heights`-arg.
        """
        low, high = vel_limits
        llim = (high - low) / num_reflections
        self.velocities = (np.linspace(low, high, num_reflections + 1) +
                           self.rng.uniform(low=-llim, high=llim, size=(num_reflections + 1, )))

        for height_share, jump_mul in zip(horizon_heights, horizon_multipliers):
            self.velocities[int(self.velocities.shape[0] * height_share)] += llim * jump_mul

        self._horizon_heights = horizon_heights
        return self

    def make_velocity_model(self, shape=(50, 400, 800), grid_shape=(10, 10), perturbation_share=.2):
        """ Make 2d or 3d velocity model out of the array of velocities and store it in the class-instance.

        Parameters
        ----------
        shape : tuple
            [n_ilines X n_xlines X n_samples].
        grid_shape : tuple
            Sets the shape of grid of support points for surfaces' interpolation (surfaces represent horizons).
        perturbation_share : float
            Sets the limit of random perturbation for surfaces' creation. The limit is set relative to the depth
            of a layer of constant velocity. The larger the value, more 'curved' are the horizons.
        """
        if len(shape) in (2, 3):
            self.dim = len(shape)
        else:
            raise ValueError('The function only supports the generation of 2d and 3d synthetic seismic.')

        num_reflections = len(self.velocities) - 1
        curves = make_surfaces(num_reflections, grid_shape, perturbation_share=perturbation_share,
                               shape=shape, rng=self.rng)
        make_colors_array = make_colors_array_2d if self.dim == 2 else make_colors_array_3d
        self.velocity_model = make_colors_array(self.velocities, curves, shape)

        # store curves-list to later use them as horizons
        self._curves = curves
        return self

    def add_faults(self, faults=(((100, 50), (100, 370)),
                                 ((50, 320), (50, 470)),
                                 ((150, 320), (150, 470))),
                   num_points=10, zeros_share=0.6, kind='cubic', perturb_values=True,
                   perturb_peak=False, peak_value=0.05, random_invert=False):
        """ Add faults to the velocity model. Faults are basically elastic transforms of patches of
        generated seismic images. Elastic transforms are performed through coordinates-transformation
        in depth-projection. Those are smooth maps [0, 1] -> [0, 1] described as f(x) = x + distortion.
        In current version, distortions are always hump-shaped. Almost all parameters of the function
        are used to define properties of the hump-shaped distortion.

        Parameters
        ----------
        faults : sequence
            iterable containing faults-coordinates in form ((x0, y0), (x1, y1)).
        num_points : int
            number of points used for making coordinate-shifts for faults.
        zeros_share : float
            left and right tails of humps are set to zeros. This is needed to make
            transformations that are identical on the tails. The parameter controls the share
            of zero-values for tails.
        kind : str
            kind of interpolation used for building coordinate-shifts.
        perturb_values : bool
            whether to add random perturbations to a coordinate-shift hump.
        perturb_peak : bool
            if set True, the position of hump's peak is randomly moved.
        peak_value : float
            value of a coordinate-shift transform in the peak of a hump
            (before random perturbation).
        random_invert : bool
            if True, the coordinate-shift is defined as x - "hump" rather than x + "hump".
        """
        if self.velocity_model is None:
            raise ValueError("You need to create velocity model first to add faults later.")

        for fault in faults:
            x = fault[0][0]
            y_low, y_high = fault[0][1], fault[1][1]
            crop = self.velocity_model[:x, y_low:y_high]
            func = make_coords_shift(self.rng, n_points=num_points, peak_value=peak_value, perturb_peak=perturb_peak,
                                     perturb_values=perturb_values, kind=kind, zeros_share=zeros_share,
                                     random_invert=random_invert)
            new_coords = func(np.arange(crop.shape[-1]) / (crop.shape[-1] - 1)) * (crop.shape[-1] - 1)
            crop_elastic = np.array([map_coordinates(trace, [new_coords]) for trace in crop])
            self.velocity_model[:x, y_low:y_high] = crop_elastic
        return self

    def make_density_model(self, density_noise_lims=(0.97, 1.3)):
        """ Make density model out of velocity model and store it in the class-instance.

        Parameters
        ----------
        density_noise_lims : tuple or None
            Density-model is given by (velocity model * noise). The param sets the limits for noise.
            If set to None, density-model is equal to velocity-model.
        """
        if density_noise_lims is not None:
            self.density_model = self.velocity_model * self.rng.uniform(*density_noise_lims,
                                                                        size=self.velocity_model.shape)
        else:
            self.density_model = self.velocity_model
        return self

    def make_synthetic(self, ricker_width=5, ricker_points=50):
        """ Generate and store 2d or 3d synthetic seismic. Synthetic seismic generation relies
        on generated velocity and density models. Hence, can be run only after `generate_velocities`,
        `generate_velocity_model` and `generate_density_model` methods.

        Parameters
        ----------
        ricker_width : float
            Width of the ricker-wave - `a`-parameter of `scipy.signal.ricker`.
        ricker_points : int
            Number of points in the ricker-wave - `points`-parameter of `scipy.signal.ricker`.
        """
        ref_coeffs = reflectivity(self.velocity_model, self.density_model)
        wavelet = ricker(ricker_points, ricker_width)
        convolve = convolve_2d if self.dim == 2 else convolve_3d
        self.synthetic = convolve(ref_coeffs, wavelet)
        return self

    def postprocess_synthetic(self, sigma=1.1, noise_mul=0.5):
        """ Simple postprocessing function for a seismic seismic, containing blur and noise.

        Parameters
        ----------
        sigma : float or None
            sigma used for gaussian blur of the synthetic seismic.
        noise_mul : float or None
            If not None, gaussian noise scale by this number is applied to the synthetic.
        """
        if sigma is not None:
            self.synthetic = gaussian_filter(self.synthetic, sigma=sigma)
        if noise_mul is not None:
            self.synthetic += noise_mul * self.rng.random(self.synthetic.shape) * self.synthetic.std()
        return self

    def fetch_horizons(self, mode='horizons'):
        """ Fetch some (or all) reflective surfaces.
        """
        if mode is None:
            return None
        if isinstance(mode, str):
            if mode == 'all':
                return self._curves
            if mode == 'horizons':
                return self._curves[[int(self._curves.shape[0] * height_share)
                                    for height_share in self._horizon_heights]]
            if 'top' in mode:
                top_k = int(mode.replace('top', ''))
                ixs = np.argsort(np.abs(np.diff(self.velocities)))[::-1][:top_k]
                return self._curves[ixs]
            raise ValueError('Mode can be one of `horizons`, `all` or `top[k]`')
        raise ValueError('Mode must be str and can be one of `horizons`, `all` or `top[k]`')

def generate_synthetic(shape=(50, 400, 800), num_reflections=200, vel_limits=(900, 5400), #pylint: disable=too-many-arguments
                       horizon_heights=(1/4, 1/2, 2/3), horizon_multipliers=(7, 5, 4), grid_shape=(10, 10),
                       perturbation_share=.2, density_noise_lims=(0.97, 1.3),
                       ricker_width=5, ricker_points=50, sigma=1.1, noise_mul=0.5,
                       faults=(((100, 50), (100, 370)),
                               ((50, 320), (50, 470)),
                               ((150, 320), (150, 470))),
                       num_points_faults=10, zeros_share_faults=0.6, fault_shift_interpolation='cubic',
                       perturb_values=True, perturb_peak=False, peak_value=0.05, random_invert=False,
                       fetch_surfaces='horizons', rng=None, seed=None):
    """ Generate and synthetic 3d-cube along with most prominient reflective surfaces ("horizons").

    Parameters
    ----------
    shape : tuple
        [n_ilines X n_xlines X n_samples].
    num_reflections : int
        The number of reflective surfaces.
    vel_limits : tuple
        Contains two floats. Velocities of layers in velocity model gradually change from the
        lower limit (first number) to the upper limit (second number) with some noise added.
    horizon_heights : tuple
        Some reflections are sharper than the others - they represent seismic horizons. The tuple contains
        heights (in [0, 1]-interval) of sharp reflections.
    horizon_multipliers : tuple
        Mutipliers controling the magnitide of sharp jumps. Should have the same length as `horizon_heights`-arg.
    grid_shapes : tuple
        Sets the shape of grid of support points for surfaces' interpolation (surfaces represent horizons).
    perturbation_share : float
        Sets the limit of random perturbation for surfaces' creation. The limit is set relative to the depth
        of a layer of constant velocity. The larger the value, more 'curved' are the horizons.
    density_noise_lims : tuple or None
        Density-model is given by (velocity model * noise). The param sets the limits for noise.
        If set to None, density-model is equal to velocity-model.
    ricker_width : float
        Width of the ricker-wave - `a`-parameter of `scipy.signal.ricker`.
    ricker_points : int
        Number of points in the ricker-wave - `points`-parameter of `scipy.signal.ricker`.
    sigma : float or None
        sigma used for gaussian blur of the synthetic seismic.
    noise_mul : float or None
        If not None, gaussian noise scale by this number is applied to the synthetic.
    faults : tuple or list
        iterable containing faults-coordinates in form ((x0, y0), (x1, y1)).
    num_points_faults : int
        number of points used for making coordinate-shifts for faults.
    zeros_share_faults : float
        left and right tails of humps are set to zeros. This is needed to make
        transformations that are identical on the tails. The parameter controls the share
        of zero-values for tails.
    fault_shift_interpolation : str
        kind of interpolation used for building coordinate-shifts.
    perturb_values : bool
        add random perturbations to a coordinate-shift hump.
    perturb_peak : bool
        if set True, the position of hump's peak is randomly moved.
    peak_value : float
        value of a coordinate-shift transform in the peak of a hump
        (before random perturbation).
    random_invert : bool
        if True, the coordinate-shift is defined as x - "hump" rather than x + "hump".
    fetch_surfaces : str
        Can be either 'horizons', 'all' or None. When 'horizons', only horizon-surfaces
        (option `horizon_heights`) are returned. Choosing 'all' allows to return all of
        the reflections, while 'topK' option leads to fetching K surfaces correpsonding
        to K largest jumps in velocities-array.
    rng : np.random.Generator or None
        generator of random numbers.
    seed : int or None
        sees used for creation of random generator (check out `np.random.default_rng`).

    Returns
    -------
    tuple
        tuple (cube, horizons); horizons can be None if `fetch_surfaces` is set to None.
    """
    if len(shape) in (2, 3):
        dim = len(shape)
    else:
        raise ValueError('The function only supports the generation of 2d and 3d synthetic seismic.')

    gen = (SyntheticGenerator(rng, seed)
           .make_velocities(num_reflections, vel_limits, horizon_heights, horizon_multipliers)
           .make_velocity_model(shape, grid_shape, perturbation_share))

    # add faults if needed and possible
    if faults is not None:
        if dim == 2:
            gen.add_faults(faults, num_points_faults, zeros_share_faults, fault_shift_interpolation,
                           perturb_values, perturb_peak, peak_value, random_invert)
        else:
            raise ValueError("For now, faults are only supported for dim = 2.")

    gen = (gen.make_density_model(density_noise_lims)
              .make_synthetic(ricker_width, ricker_points)
              .postprocess_synthetic(sigma, noise_mul))
    return gen.synthetic, gen.fetch_horizons(fetch_surfaces)


def surface_to_points(surface):
    """ Make points-array by adding ilines-xlines columns and flattening the surface-column.
    No offset is added: ilines and xlines are assumed to be simple ranges 0..ilines_len.

    Parameters
    ----------
    surface : np.ndarray
        array of heights representing the reflective surface in a generated cube.
    """
    n_ilines, n_xlines = surface.shape
    mesh = np.meshgrid(range(n_ilines), range(n_xlines), indexing='ij')
    points = np.stack([mesh[0].reshape(-1), mesh[1].reshape(-1),
                       surface.reshape(-1)], axis=1).astype(np.int)
    return points
