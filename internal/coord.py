from internal import math
from internal import utils
import numpy as np
import torch
# from torch.func import vmap, jacrev


def contract(x):
    """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
    eps = torch.finfo(x.dtype).eps
    # eps = 1e-3
    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
    z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    return z


def inv_contract(z):
    """The inverse of contract()."""
    eps = torch.finfo(z.dtype).eps
    # eps = 1e-3

    # Clamping to eps prevents non-finite gradients when z == 0.
    z_mag_sq = torch.sum(z ** 2, dim=-1, keepdim=True).clamp_min(eps)
    x = torch.where(z_mag_sq <= 1, z, z / (2 * torch.sqrt(z_mag_sq) - z_mag_sq))
    return x


def contract_tuple(x):
    res = contract(x)
    return res, res


@torch.compile
def contract_mean_jacobi(x):
    eps = torch.finfo(x.dtype).eps
    # eps = 1e-3

    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
    x_mag_sqrt = torch.sqrt(x_mag_sq)
    x_xT = math.matmul(x[..., None], x[..., None, :])
    mask = x_mag_sq <= 1
    z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)

    eye = torch.broadcast_to(torch.eye(3, device=x.device), z.shape[:-1] + z.shape[-1:] * 2)
    jacobi = (2 * x_xT * (1 - x_mag_sqrt[..., None]) + (2 * x_mag_sqrt[..., None] ** 3 - x_mag_sqrt[..., None] ** 2) * eye) / x_mag_sqrt[..., None] ** 4
    jacobi = torch.where(mask[..., None], eye, jacobi)
    return z, jacobi


@torch.compile
def contract_mean_std(x, std):
    eps = torch.finfo(x.dtype).eps
    # eps = 1e-3
    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
    x_mag_sqrt = torch.sqrt(x_mag_sq)
    mask = x_mag_sq <= 1
    z = torch.where(mask, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    det = ((1 / x_mag_sq) * ((2 / x_mag_sqrt - 1 / x_mag_sq) ** 2))[..., 0]

    std = torch.where(mask[..., 0], std, (det ** (1 / x.shape[-1])) * std)
    return z, std


@torch.no_grad()
def track_linearize(fn, mean, std):
    """Apply function `fn` to a set of means and covariances, ala a Kalman filter.

  We can analytically transform a Gaussian parameterized by `mean` and `cov`
  with a function `fn` by linearizing `fn` around `mean`, and taking advantage
  of the fact that Covar[Ax + y] = A(Covar[x])A^T (see
  https://cs.nyu.edu/~roweis/notes/gaussid.pdf for details).

  Args:
    fn: the function applied to the Gaussians parameterized by (mean, cov).
    mean: a tensor of means, where the last axis is the dimension.
    std: a tensor of covariances, where the last two axes are the dimensions.

  Returns:
    fn_mean: the transformed means.
    fn_cov: the transformed covariances.
  """
    if fn == 'contract':
        fn = contract_mean_jacobi
    else:
        raise NotImplementedError

    pre_shape = mean.shape[:-1]
    mean = mean.reshape(-1, 3)
    std = std.reshape(-1)

    # jvp_1, mean_1 = vmap(jacrev(contract_tuple, has_aux=True))(mean)
    # std_1 = std * torch.linalg.det(jvp_1) ** (1 / mean.shape[-1])
    #
    # mean_2, jvp_2 = fn(mean)
    # std_2 = std * torch.linalg.det(jvp_2) ** (1 / mean.shape[-1])
    #
    # mean_3, std_3 = contract_mean_std(mean, std)  # calculate det explicitly by using eigenvalues
    # torch.allclose(std_1, std_3, atol=1e-7)  # True
    # torch.allclose(mean_1, mean_3)  # True
    # import ipdb; ipdb.set_trace()
    mean, std = contract_mean_std(mean, std)  # calculate det explicitly by using eigenvalues

    mean = mean.reshape(*pre_shape, 3)
    std = std.reshape(*pre_shape)
    return mean, std


def power_transformation(x, lam):
    """
    power transformation for Eq(4) in zip-nerf
    """
    lam_1 = np.abs(lam - 1)
    return lam_1 / lam * ((x / lam_1 + 1) ** lam - 1)


def inv_power_transformation(x, lam):
    """
    inverse power transformation
    """
    lam_1 = np.abs(lam - 1)
    eps = torch.finfo(x.dtype).eps  # may cause inf
    # eps = 1e-3
    return ((x * lam / lam_1 + 1 + eps) ** (1 / lam) - 1) * lam_1


def construct_ray_warps(fn, t_near, t_far, lam=None):
    """Construct a bijection between metric distances and normalized distances.

  See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
  detailed explanation.

  Args:
    fn: the function to ray distances.
    t_near: a tensor of near-plane distances.
    t_far: a tensor of far-plane distances.
    lam: for lam in Eq(4) in zip-nerf

  Returns:
    t_to_s: a function that maps distances to normalized distances in [0, 1].
    s_to_t: the inverse of t_to_s.
  """
    if fn is None:
        fn_fwd = lambda x: x
        fn_inv = lambda x: x
    elif fn == 'piecewise':
        # Piecewise spacing combining identity and 1/x functions to allow t_near=0.
        fn_fwd = lambda x: torch.where(x < 1, .5 * x, 1 - .5 / x)
        fn_inv = lambda x: torch.where(x < .5, 2 * x, .5 / (1 - x))
    elif fn == 'power_transformation':
        fn_fwd = lambda x: power_transformation(x * 2, lam=lam)
        fn_inv = lambda y: inv_power_transformation(y, lam=lam) / 2
    else:
        inv_mapping = {
            'reciprocal': torch.reciprocal,
            'log': torch.exp,
            'exp': torch.log,
            'sqrt': torch.square,
            'square': torch.sqrt,
            'power_transformation': power_transformation,
        }
        fn_fwd = fn
        fn_inv = inv_mapping[fn.__name__]

    s_near, s_far = [fn_fwd(x) for x in (t_near, t_far)]
    t_to_s = lambda t: (fn_fwd(t) - s_near) / (s_far - s_near)
    s_to_t = lambda s: fn_inv(s * s_far + (1 - s) * s_near)
    return t_to_s, s_to_t


def expected_sin(mean, var):
    """Compute the mean of sin(x), x ~ N(mean, var)."""
    return torch.exp(-0.5 * var) * math.safe_sin(mean)  # large var -> small value.


def integrated_pos_enc(mean, var, min_deg, max_deg):
    """Encode `x` with sinusoids scaled by 2^[min_deg, max_deg).

  Args:
    mean: tensor, the mean coordinates to be encoded
    var: tensor, the variance of the coordinates to be encoded.
    min_deg: int, the min degree of the encoding.
    max_deg: int, the max degree of the encoding.

  Returns:
    encoded: tensor, encoded variables.
  """
    scales = 2 ** torch.arange(min_deg, max_deg, device=mean.device)
    shape = mean.shape[:-1] + (-1,)
    scaled_mean = (mean[..., None, :] * scales[:, None]).reshape(*shape)
    scaled_var = (var[..., None, :] * scales[:, None] ** 2).reshape(*shape)

    return expected_sin(
        torch.cat([scaled_mean, scaled_mean + 0.5 * torch.pi], dim=-1),
        torch.cat([scaled_var] * 2, dim=-1))


def lift_and_diagonalize(mean, cov, basis):
    """Project `mean` and `cov` onto basis and diagonalize the projected cov."""
    fn_mean = math.matmul(mean, basis)
    fn_cov_diag = torch.sum(basis * math.matmul(cov, basis), dim=-2)
    return fn_mean, fn_cov_diag


def pos_enc(x, min_deg, max_deg, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = 2 ** torch.arange(min_deg, max_deg, device=x.device)
    shape = x.shape[:-1] + (-1,)
    scaled_x = (x[..., None, :] * scales[:, None]).reshape(*shape)
    # Note that we're not using safe_sin, unlike IPE.
    four_feat = torch.sin(
        torch.cat([scaled_x, scaled_x + 0.5 * torch.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat
