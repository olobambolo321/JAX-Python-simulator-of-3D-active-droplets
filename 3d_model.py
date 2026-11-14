"""
Three-dimensional reaction--Cahn--Hilliard droplet simulation.

Model
-----
    dc/dt = laplacian(mu) + Da * reaction_flux(c)
    mu    = 0.5 * (c**3 - c) - 0.25 * laplacian(c)
-----

The spatial derivatives are evaluated pseudospectrally on a periodic
Cartesian grid.  The -0.25*laplacian(laplacian(c)) and the linear part
of reaction flux are term treated implicitly;
the remaining terms are explicit.

The reaction law is a piecewise C1 source with a cubic interpolant.

The code first tries a nonlinear radial steady-state BVP.  If that BVP
converges to the homogeneous exterior state, the script falls back to a
prescribed smooth ellipsoidal tanh interface.  The fallback is an initial
condition, not an equilibrium solution.
"""

from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from skimage.measure import marching_cubes

import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())

if not any(d.platform == "gpu" for d in jax.devices()):
    raise RuntimeError("No JAX GPU device found.")

print("Using GPU:", jax.devices("gpu")[0])




# =============================================================================
# 1. USER PARAMETERS
# =============================================================================


# Domain and grid.
NX = 2**8                   # datapoints per domain length
NY = 2**8                   ##
NZ = 2**8                   ##
LX = 144.0                  # domain lengths
LY = 144.0                  ##
LZ = 144.0                  ##

# Initial droplet and geometry.
DROPLET_SIZE = 3.0e-2       # R0 / Lx
DROPLET_SQUISH = 1.01       # 1 = sphere; >1 elongates along rotated x axis
DROPLET_ANGLE = np.pi / np.sqrt(2.0)    # with respect to x
INTERFACE_WIDTH_FRACTION = 0.15  # width in ellipsoidal-radius coordinates
C_IN = 1.0                  # concentration inside the droplet
NOISE_AMPLITUDE = 1.0e-3    # optional Gaussian noise over whole domain
RNG_SEED = 151232           # for reproducibility

# Reaction parameters.
EPSILON = 0.21              # supersaturation
ALPHA = 0.70                # turnover
DA = 0.015                   # Damkohler number

# Bounds of the cubic interpolation interval (q_p < q_m).
Q_P = -0.5
Q_M = 0.5

# Split the reaction source into an implicit globally linear part and
# an explicit remainder: r(phi) = A0 - B0*phi + g(phi).
A0 = 2.0 * EPSILON
B0 = 1.0

# Time integration.
DT = 1.0e-1                 # initial time-step size
N_STEPS = 10000000          # number of simulation steps
SAVE_EVERY = 1000           # plot figures every . steps

# Radial BVP initialization.
TRY_RADIAL_BVP = True       # if False initialize tanh profile
BVP_TOL = 1.0e-5            # ODE solver tolerance
BVP_MAX_NODES = 30000
BVP_MESH_POINTS = 400       # datapoints for radial profile

# Visualization/output.
ISO_LEVEL = 0.0             # surface level for the 3D plot
SURFACE_STEP_SIZE = 2
OUTPUT_DIR = Path("droplet_simulation_output")
SHOW_PLOTS = False          # if True display plots during simulation

# Adaptive time-step integration controls.  DT is the initial timestep.
DT_MIN = 1.0e-4             # minimum time-step size
DT_MAX = 1.0e1              # maximum time-step size
DT_TARGET_CHANGE = 1.0e-3
DT_SAFETY = 0.85
MAX_REJECTIONS = 30
DEALIAS_NONLINEAR_TERMS = True # apply 2/3 rule for dealiasing (recommended)

# Legacy visualization controls.
SPECTRAL_CROP_MODES = 32
FADE_STEPS = 20
FADE_ALPHA_MIN = 0.25
FADE_GREY_MAX = 0.70




# =============================================================================
# 2. REACTION KINETICS
# =============================================================================


# Cubic interpolation from value and derivative matching (C^1) at q_p, q_m.
i_p_target = 2.0 * EPSILON - (1.0 + Q_P)
i_m_target = -2.0 * ALPHA + (1.0 - Q_M)
di_p_target = -1.0
di_m_target = -1.0

A_interp = np.array([
    [1.0, Q_P, Q_P**2, Q_P**3],
    [1.0, Q_M, Q_M**2, Q_M**3],
    [0.0, 1.0, 2.0 * Q_P, 3.0 * Q_P**2],
    [0.0, 1.0, 2.0 * Q_M, 3.0 * Q_M**2],
], dtype=np.float64)
b_interp = np.array([
    i_p_target, i_m_target, di_p_target, di_m_target
], dtype=np.float64)
a0, a1, a2, a3 = map(float, np.linalg.solve(A_interp, b_interp))
print(
    "C1 reaction interpolation coefficients "
    f"[a0, a1, a2, a3] = {[a0, a1, a2, a3]}"
)

def _reaction_cubic(phi):
    return a0 + a1 * phi + a2 * phi**2 + a3 * phi**3

def _reaction_cubic_derivative(phi):
    return a1 + 2.0 * a2 * phi + 3.0 * a3 * phi**2

# Check if C1 is satisfied.
for _q, _outer_value in (
    (Q_P, lambda q: 2.0 * EPSILON - (1.0 + q)),
    (Q_M, lambda q: -2.0 * ALPHA + (1.0 - q)),
):
    _outer_slope = -1.0
    if not (
        np.isclose(_reaction_cubic(_q), _outer_value(_q), atol=1e-12)
        and np.isclose(_reaction_cubic_derivative(_q), _outer_slope, atol=1e-12)
    ):
        raise RuntimeError(f"Reaction interpolation is not C1 at phi={_q}.")

# Define reaction flux functions.
def reaction_flux_numpy(phi: np.ndarray | float) -> np.ndarray:
    values = np.asarray(phi, dtype=np.float64)
    s_out = 2.0 * EPSILON - (1.0 + values)
    s_in = -2.0 * ALPHA + (1.0 - values)
    s_mid = a0 + a1 * values + a2 * values**2 + a3 * values**3
    return np.where(values < Q_P, s_out, np.where(values > Q_M, s_in, s_mid))


def reaction_flux_jax(phi: jnp.ndarray) -> jnp.ndarray:
    s_out = 2.0 * EPSILON - (1.0 + phi)
    s_in = -2.0 * ALPHA + (1.0 - phi)
    s_mid = a0 + a1 * phi + a2 * phi**2 + a3 * phi**3
    return jnp.where(phi < Q_P, s_out, jnp.where(phi > Q_M, s_in, s_mid))

# Explicit helper function.
def reaction_g_jax(phi):
    return reaction_flux_jax(phi) - A0 + B0 * phi




# =============================================================================
# 3. RADIAL STEADY-STATE BOUNDARY-VALUE PROBLEM
# =============================================================================


def try_solve_radial_steady_profile(
    c_in: float,
    c_out: float,
    radius_guess: float,
    interface_width: float,
    r_max: float,
):
    """
    Attempt a nonlinear spherical steady-state reaction-CH profile.

    For a stationary radial profile, with p = dc/dr and q = dmu/dr,
        c'  = p
        p'  = 2(c^3-c) - 4*mu - 2p/r
        mu' = q
        q'  = -Da*reaction_flux(c) - 2q/r.

    Regularity at the centre is approximated at a small positive radius
    by p=0 and q=0.  At the far boundary, c and mu are fixed to their
    homogeneous outside values.

    A converged BVP is accepted only if it contains a resolved interface.
    """
    source_out = float(reaction_flux_numpy(c_out))
    if abs(source_out) > 1.0e-8:
        print(
            "Radial BVP skipped: the prescribed outside concentration "
            f"is not a reaction equilibrium (source={source_out:.3e})."
        )
        return None

    r_min = max(1.0e-3 * interface_width, 1.0e-8)
    if r_max <= r_min or radius_guess + 5.0 * interface_width >= r_max:
        print("Radial BVP skipped: radial domain is too small.")
        return None

    r = np.linspace(r_min, r_max, BVP_MESH_POINTS)
    xi = (r - radius_guess) / interface_width
    tanh_xi = np.tanh(xi)
    sech2 = 1.0 - tanh_xi**2
    delta_c = c_in - c_out

    c_guess = c_out + 0.5 * delta_c * (1.0 - tanh_xi)
    cp_guess = -0.5 * delta_c / interface_width * sech2
    cpp_guess = delta_c / interface_width**2 * sech2 * tanh_xi
    mu_guess = (
        0.5 * (c_guess**3 - c_guess)
        - 0.25 * (cpp_guess + 2.0 * cp_guess / r)
    )
    mup_guess = np.gradient(mu_guess, r)
    y_guess = np.vstack((c_guess, cp_guess, mu_guess, mup_guess))

    mu_out = 0.5 * (c_out**3 - c_out)

    def ode(radial_coordinate, y):
        c, cp, mu, mup = y
        return np.vstack((
            cp,
            2.0 * (c**3 - c) - 4.0 * mu - 2.0 * cp / radial_coordinate,
            mup,
            -DA * reaction_flux_numpy(c) - 2.0 * mup / radial_coordinate,
        ))

    def bc(ya, yb):
        return np.array((
            ya[1],             # dc/dr = 0 at the centre
            ya[3],             # dmu/dr = 0 at the centre
            yb[0] - c_out,     # outside concentration
            yb[2] - mu_out,    # outside chemical potential
        ))

    try:
        solution = solve_bvp(
            ode, bc, r, y_guess,
            tol=BVP_TOL,
            max_nodes=BVP_MAX_NODES,
            verbose=0,
        )
    except Exception as exc:
        print(f"Radial BVP failed; using tanh initial condition. ({exc})")
        return None

    if not solution.success:
        print(
            "Radial BVP did not converge; using tanh initial condition. "
            f"({solution.message})"
        )
        return None

    r_profile = np.linspace(r_min, r_max, 2000)
    y_profile = solution.sol(r_profile)
    c_profile = y_profile[0]

    if np.ptp(c_profile) < 1.0e-2:
        print(
            "Radial BVP converged to the homogeneous outside state, not a "
            "droplet. Using a prescribed tanh interface for the simulation."
        )
        return None

    c_mid = 0.5 * (c_in + c_out)
    crossings = np.flatnonzero(
        (c_profile[:-1] - c_mid) * (c_profile[1:] - c_mid) <= 0.0
    )
    if crossings.size == 0:
        print(
            "Radial BVP has no interior-to-exterior interface crossing; "
            "using a prescribed tanh interface."
        )
        return None

    i = int(crossings[0])
    denominator = c_profile[i + 1] - c_profile[i]
    if abs(denominator) < 1.0e-14:
        print("Radial BVP interface crossing is degenerate; using tanh.")
        return None

    fraction = (c_mid - c_profile[i]) / denominator
    interface_radius = r_profile[i] + fraction * (r_profile[i + 1] - r_profile[i])

    print("A non-homogeneous radial steady profile was found.")
    print(f"  BVP interface radius = {interface_radius:.6g}")
    return r_profile, c_profile, interface_radius




# =============================================================================
# 4. ROTATED ELLIPSOID INITIAL CONDITION
# =============================================================================


def make_initial_condition(x, y, z):
    """Build a rotated ellipsoidal droplet without large 3D mesh arrays."""
    r0 = DROPLET_SIZE * LX
    if DROPLET_SQUISH <= 0.0:
        raise ValueError("DROPLET_SQUISH must be positive.")

    # Volume-preserving principal semi-axes: a*b*c = r0**3.
    a = r0 * DROPLET_SQUISH
    b = r0 / np.sqrt(DROPLET_SQUISH)
    c_axis = r0 / np.sqrt(DROPLET_SQUISH)

    if a >= 0.5 * LX or b >= 0.5 * LY or c_axis >= 0.5 * LZ:
        raise ValueError(
            "The ellipsoid must fit inside half the periodic box in each "
            "principal direction. Reduce DROPLET_SIZE or DROPLET_SQUISH."
        )

    cos_angle = np.cos(DROPLET_ANGLE)
    sin_angle = np.sin(DROPLET_ANGLE)

    r_guess = r0
    width_physical = max(
        r0 * INTERFACE_WIDTH_FRACTION,
        1.5 * min(LX / NX, LY / NY, LZ / NZ),
    )

    radial_profile = None
    if TRY_RADIAL_BVP:
        radial_profile = try_solve_radial_steady_profile(
            C_IN,
            2.0 * EPSILON - 1.0,
            r_guess,
            width_physical,
            0.5 * min(LX, LY, LZ),
        )

    c_out = 2.0 * EPSILON - 1.0
    c0 = np.empty((NX, NY, NZ), dtype=np.float32)

    yy, zz = np.meshgrid(y, z, indexing="ij")

    if radial_profile is not None:
        r_profile, c_profile, interface_radius = radial_profile
        print("Mapping the radial BVP profile onto the ellipsoid.")
    else:
        print("Initializing a smooth, prescribed tanh ellipsoidal droplet.")
        print("This is a non-equilibrium initial condition, not a steady state.")

    for ix, xv in enumerate(x):
        x_rot = cos_angle * xv + sin_angle * yy
        y_rot = -sin_angle * xv + cos_angle * yy

        ellipsoidal_radius = np.sqrt(
            (x_rot / a)**2 + (y_rot / b)**2 + (zz / c_axis)**2
        )

        if radial_profile is not None:
            physical_radius = interface_radius * ellipsoidal_radius
            c_slice = np.interp(
                physical_radius.ravel(),
                r_profile,
                c_profile,
                left=float(c_profile[0]),
                right=c_out,
            ).reshape((NY, NZ))
        else:
            width_rho = INTERFACE_WIDTH_FRACTION
            c_slice = c_out + 0.5 * (C_IN - c_out) * (
                1.0 - np.tanh((ellipsoidal_radius - 1.0) / width_rho)
            )

        c0[ix, :, :] = c_slice.astype(np.float32)

    if NOISE_AMPLITUDE > 0.0:
        rng = np.random.default_rng(RNG_SEED)
        c0 += NOISE_AMPLITUDE * rng.standard_normal(c0.shape).astype(np.float32)

    return c0




# =============================================================================
# 5. FOURIER OPERATORS AND SEMI-IMPLICIT TIME STEPPER
# =============================================================================


def make_spectral_stepper(
    k2_np: np.ndarray,
    k4_np: np.ndarray,
    dealias_mask_np: np.ndarray
):
    """
    Build a semi-implicit Fourier step for a real-valued field.
    A 2/3 rule is applied for dealiasing.

    The spectrum is recomputed from the real field at the start of each
    step and the result is brought back to real space at the end, so any
    non-Hermitian round-off is discarded every step.
    """
    k2 = jnp.asarray(k2_np, dtype=jnp.float32)
    k4 = jnp.asarray(k4_np, dtype=jnp.float32)
    mask = jnp.asarray(dealias_mask_np, dtype=jnp.float32)
    
    linear_operator = 0.5 * k2 - 0.25 * k4 - DA * B0

    @jax.jit
    def step(phi, dt):
        phi_hat = jnp.fft.fftn(phi)
        phi3_hat = jnp.fft.fftn(phi**3) * mask
        g_hat = jnp.fft.fftn(reaction_g_jax(phi)) * mask

        rhs = phi_hat - dt * 0.5 * k2 * phi3_hat + dt * DA * g_hat

        # Constant source A0 only feeds the k = 0 mode.
        rhs = rhs.at[0, 0, 0].add(dt * A0 * DA * phi.size)

        denominator = 1.0 - dt * linear_operator
        phi_hat_next = rhs / denominator
        phi_next = jnp.fft.ifftn(phi_hat_next).real

        max_change = jnp.max(jnp.abs(phi_next - phi))
        finite = jnp.all(jnp.isfinite(phi_next))
        return phi_next, max_change, finite

    return step




# =============================================================================
# 6. DIAGNOSTICS AND VISUALIZATION
# =============================================================================


def _param_string():
    """String of relevant parameters used for figure titles."""
    return (rf"$\epsilon={EPSILON:.2f}$, $\alpha={ALPHA:.2f}$, "
            rf"Da$={DA:.3f}$, noise$={NOISE_AMPLITUDE:.1e}$")


def equivalent_positive_phase_radius(c_field: np.ndarray) -> float:
    """Equivalent spherical radius of the region where c > ISO_LEVEL."""
    volume_element = (LX / NX) * (LY / NY) * (LZ / NZ)
    volume = np.count_nonzero(c_field > ISO_LEVEL) * volume_element
    if volume <= 0.0:
        return 0.0
    return float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))


def save_midplane_plot(c_field: np.ndarray, filename: Path, title: str):
    """Save a concentration slice through the centre of the box."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(
        c_field[:, :, NZ // 2].T,
        origin="lower",
        extent=(-0.5 * LX, 0.5 * LX, -0.5 * LY, 0.5 * LY),
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    ax.contour(
        np.linspace(-0.5 * LX, 0.5 * LX, NX),
        np.linspace(-0.5 * LY, 0.5 * LY, NY),
        c_field[:, :, NZ // 2].T,
        levels=[ISO_LEVEL],
        colors="black",
        linewidths=0.8,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title + "\n" + _param_string(), fontsize=10)
    fig.colorbar(image, ax=ax, label=r"Dimless concentration density $\varphi$")
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def save_mean_absolute_log_plot(times, mean_absolute_values, filename):
    """Save log-scale history of the spatial mean absolute density field."""
    times = np.asarray(times, dtype=float)
    mean_absolute_values = np.asarray(mean_absolute_values, dtype=float)
    if times.shape != mean_absolute_values.shape:
        raise ValueError("times and mean_absolute_values must have the same shape.")
    if times.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(
        times+DT,
        np.maximum(mean_absolute_values, np.finfo(float).tiny),
        linewidth=1.8,
        marker="o",
        markersize=3,
    )
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\langle |\varphi| \rangle$")
    ax.set_title("Mean absolute field evolution on a log-log scale")
    fig.tight_layout()
    ax.grid(True, which="both", alpha=0.5)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_surface_plot(
    c_field: np.ndarray,
    filename: Path,
    title: str,
    times=None,
    mean_absolute_values=None,
):
    """Plot the 3D isosurface with the mean absolute density field below."""
    c_min = float(np.nanmin(c_field))
    c_max = float(np.nanmax(c_field))

    if not (c_min < ISO_LEVEL < c_max):
        print(f"No phi={ISO_LEVEL:g} isosurface in {filename.name}; skipping.")
        return

    spacing = (LX / NX, LY / NY, LZ / NZ)
    volume_for_surface = np.array(c_field, dtype=np.float32, copy=True)

    vertices, faces, _, _ = marching_cubes(
        volume_for_surface,
        level=ISO_LEVEL,
        spacing=spacing,
        step_size=SURFACE_STEP_SIZE,
        allow_degenerate=False,
    )

    vertices[:, 0] -= 0.5 * LX
    vertices[:, 1] -= 0.5 * LY
    vertices[:, 2] -= 0.5 * LZ

    triangles = vertices[faces]

    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[5.0, 0.9], hspace=0.16)

    ax = fig.add_subplot(gs[0], projection="3d")
    ax_mean = fig.add_subplot(gs[1])

    surface = Poly3DCollection(
        triangles,
        facecolor=(0.25, 0.55, 0.82, 0.82),
        edgecolor="none",
        linewidth=0.0,
    )
    ax.add_collection3d(surface)

    ax.set_xlim(-0.5 * LX, 0.5 * LX)
    ax.set_ylim(-0.5 * LY, 0.5 * LY)
    ax.set_zlim(-0.5 * LZ, 0.5 * LZ)
    ax.set_box_aspect((LX, LY, LZ))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title + "\n" + _param_string(), fontsize=10)
    ax.view_init(elev=24, azim=35)

    if times is not None and mean_absolute_values is not None:
        times = np.asarray(times, dtype=float)
        mean_absolute_values = np.asarray(mean_absolute_values, dtype=float)

        if times.shape != mean_absolute_values.shape:
            raise ValueError(
                "times and mean_absolute_values must have the same shape."
            )

        mean_magnitude = np.maximum(mean_absolute_values, np.finfo(float).tiny)

        ax_mean.semilogy(
            times, mean_magnitude, linewidth=1.8, marker="o", markersize=3
        )
        ax_mean.set_ylabel(r"$\langle |\varphi| \rangle$")
        ax_mean.set_xlabel("time")
        ax_mean.set_xlim(left=0.0)
        ax_mean.grid(True, which="both", alpha=0.3)
        ax_mean.tick_params(labelsize=8)
        ax_mean.set_title("Mean absolute field evolution on a log-log scale", fontsize=9, pad=3)
    else:
        ax_mean.set_visible(False)

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_profile_plot(c_field: np.ndarray, filename: Path, title: str):
    """Plot the three Cartesian centre-line cuts of the 3D field."""
    ix, iy, iz = NX // 2, NY // 2, NZ // 2
    x = np.linspace(-0.5 * LX, 0.5 * LX, NX, endpoint=False)
    y = np.linspace(-0.5 * LY, 0.5 * LY, NY, endpoint=False)
    z = np.linspace(-0.5 * LZ, 0.5 * LZ, NZ, endpoint=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, c_field[:, iy, iz], label="x cut through centre")
    ax.plot(y, c_field[ix, :, iz], label="y cut through centre")
    ax.plot(z, c_field[ix, iy, :], label="z cut through centre")
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("coordinate")
    ax.set_ylabel(r"Dimless concentration density $\varphi$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def save_diagnostics_plot(times, masses, radii, filename: Path):
    """Save mean concentration and positive-phase equivalent radius."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(times, masses)
    axes[0].set_ylabel("mean concentration")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, radii)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"equivalent radius of \varphi > 0")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


# Zwicker et al. 2017 droplet phase diagram regions
def _phase_region(epsilon, alpha):
    x, y = round(float(epsilon), 2), round(float(alpha), 2)
    if x == 0.2 and y == 1.8:
        return "stable"
    if x == 0.19 and y > 1.7:
        return "stable"
    if x == 0.18 and y > 1.6:
        return "stable"
    if x == 0.15 and (y < 1.1 and y > 0.8):
        return "two"
    if x == 0.16 and y == 0.8:
        return "three"
    if x == 0.15 and y == 0.8:
        return "three"
    if x == 0.14 and (y > 0.4 and y < 0.8):
        return "elongation"
    if x == 0.11 and y > 1.6:
        return "none"
    if x == 0.1 and y > 1.0:
        return "none"
    if x == 0.09 and y > 0.6:
        return "none"
    if x == 0.08 and y == 0.1:
        return "stable"
    if x < 0.09:
        return "none"
    if x < 0.14 and y > 0.3:
        return "stable"
    if x < 0.13 and y < 2.0:
        return "stable"
    if y > 0.8 and x < 0.16:
        return "stable"
    if y > 0.5 and x < 0.15 and not (y < 0.9 and x > 0.13):
        return "stable"
    if y > 1.3 and x < 0.18:
        return "stable"
    if x == 0.13 and y < 0.4:
        return "elongation"
    if x < 0.17 and y < 0.8:
        return "three"
    return "two"


def _draw_restored_2d(c_field, sim_time, mean_value, filename):
    """Concentration slice, centre profiles, reaction flux and bulk mu."""
    iz, ix, iy = NZ // 2, NX // 2, NY // 2
    x = np.arange(NX) * LX / NX - LX / 2
    y = np.arange(NY) * LY / NY - LY / 2
    c_slice = c_field[:, :, iz]

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    ax_slice = fig.add_subplot(gs[:, 0])
    ax_profiles = fig.add_subplot(gs[0, 1])
    ax_flux = fig.add_subplot(gs[1, 1])
    c_plot = np.linspace(-1, 1, 1000)

    # 2D central slice
    im = ax_slice.imshow(
        c_slice.T, origin="lower",
        extent=(-LX/2, LX/2, -LY/2, LY/2),
        vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal"
    )
    ax_slice.contour(x, y, c_slice.T, levels=[ISO_LEVEL],
                     colors="black", linewidths=0.7)
    ax_slice.axhline(0, color="orange", ls=":", lw=0.9)
    ax_slice.axvline(0, color="green", ls=":", lw=0.9)
    ax_slice.set(xlabel="x", ylabel="y",
                 title=(rf"$t={sim_time:.3g}$, "
                        rf"$\langle \varphi \rangle={mean_value:.5f}$" + "\n"
                        + _param_string()))
    ax_slice.tick_params(which="both", direction="in", top=True, right=True,
                      axis="both")
    fig.colorbar(im, ax=ax_slice, label=r"$\varphi$", shrink=0.85)
    
    # Centre profiles
    ax_profiles.plot(x, c_slice[:, iy], color="orange", label="x centre cut")
    ax_profiles.plot(y, c_slice[ix, :], color="green", label="y centre cut")
    ax_profiles.set(xlabel="position", ylabel=r"$\varphi$", ylim=(-1.05, 1.05))
    ax_profiles.legend(frameon=False, fontsize=8)
    ax_profiles.tick_params(which="both", direction="in", top=True, right=True,
                      axis="both")

    # Reaction flux function
    ax_flux.plot(c_plot, reaction_flux_numpy(c_plot),
                 color="blue", label=r"$r(\varphi)$")
    ax_flux.axhline(0, color="blue", ls="--", lw=0.7)
    ax_flux.set(xlabel=r"$\varphi$", xlim=(-1, 1))
    ax_flux.set_ylabel(r"$r(\varphi)$",color="blue")
    ax_flux.tick_params(axis="x", which="both", direction="in", top=True, right=True)
    ax_flux.tick_params(which="both", direction="in", top=True, right=True,
                      axis="y", labelcolor="blue")
    
    # Bulk chemical potential function
    ax_mu = ax_flux.twinx()
    ax_mu.plot(c_plot, 0.5 * (c_plot**3 - c_plot),
               color="darkorange", label=r"$\mu_{\rm bulk}(\varphi)$")
    ax_mu.axhline(0, color="darkorange", ls="--", lw=0.7)
    ax_mu.set_ylabel(r"$\mu_{\rm bulk}$", color="darkorange")
    ax_mu.tick_params(which="both", direction="in", top=True, right=True,
                      axis="y", labelcolor="darkorange")

    ax_flux.axvline(Q_P, color="red", ls=":", lw=0.9)
    ax_flux.axvline(Q_M, color="red", ls=":", lw=0.9)
    ax_flux.text(Q_P, 0.93, r"$q_+$", color="red",
         transform=ax_flux.get_xaxis_transform(),
         ha="right", va="top")
    ax_flux.text(Q_M, 0.93, r"$q_-$", color="red",
         transform=ax_flux.get_xaxis_transform(),
         ha="left", va="top")
    
    # Save figure
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_restored_3d(c_field, sim_time, mean_value, marker_history, filename):
    """3D interface plus stability and qualitative droplet phase diagrams."""
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[2.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.30,
        hspace=0.30,
    )
    ax_surface = fig.add_subplot(gs[:, 0], projection="3d")
    ax_stability = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[1, 1])

    cmin, cmax = float(np.min(c_field)), float(np.max(c_field))
    if cmin < ISO_LEVEL < cmax:
        volume_for_surface = np.array(
            c_field, dtype=np.float32, copy=True, order="C"
        )
        verts, faces, _, _ = marching_cubes(
            volume_for_surface,
            level=ISO_LEVEL,
            spacing=(LX/NX, LY/NY, LZ/NZ),
            step_size=SURFACE_STEP_SIZE,
            allow_degenerate=False
        )
        verts -= np.array([LX/2, LY/2, LZ/2])
        triangles = verts[faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0]
        )
        face_normals /= np.maximum(
            np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12
        )
        az, alt = np.deg2rad(160.0), np.deg2rad(45.0)
        light_dir = np.array([
            np.cos(alt) * np.cos(az),
            np.cos(alt) * np.sin(az),
            np.sin(alt),
        ])
        intensity = np.maximum(face_normals @ light_dir, 0.0)
        shade = 0.30 + 0.70 * intensity
        base_rgb = np.array([0.0, 0.5, 0.0])
        face_colors = np.clip(base_rgb[None, :] * shade[:, None], 0.0, 1.0)
        mesh = Poly3DCollection(
            triangles, facecolors=face_colors, alpha=1.0, edgecolor="none"
        )
        ax_surface.add_collection3d(mesh)
    ax_surface.set_xlim(-LX/2, LX/2)
    ax_surface.set_ylim(-LY/2, LY/2)
    ax_surface.set_zlim(-LZ/2, LZ/2)
    ax_surface.set_box_aspect((LX, LY, LZ))
    ax_surface.set(xlabel="x", ylabel="y", zlabel="z",
                   title=(rf"$t={sim_time:.3g}$, $\varphi = {ISO_LEVEL:.1f}$, "
                          rf"$\langle \varphi\rangle={mean_value:.5f}$" + "\n"
                          + _param_string()))

    # Stability diagram
    cplot = np.linspace(-1, 1, 1000)
    spin = np.linspace(-1/np.sqrt(3), 1/np.sqrt(3), 1000)
    coexist = (cplot**2 - 1)**2 / 16
    critical = (3*spin**2 - 1)**2 / 16
    critical_interp = np.interp(cplot, spin, critical)

    coex_line = ax_stability.plot(cplot, coexist, color="blue", label="coexistence")
    crit_line = ax_stability.plot(spin, critical, color="red", label="critical")
    coex_fill = ax_stability.fill_between(cplot, critical_interp, coexist,
                              color="blue", alpha=0.2)
    crit_fill = ax_stability.fill_between(
        spin, critical, 0,
        facecolor="red",
        alpha=0.2,
        hatch="///",
        edgecolor="red",
        linewidth=0.0
    )

    # Fading mean-field marker
    c_out = 2*EPSILON - 1
    ax_stability.axvline(c_out, color="grey", ls="--", lw=0.9)
    ax_stability.axvline(mean_value, color="grey", ls=":", lw=0.8)
    ax_stability.set(xlim=(-1, 1), ylim=(0, 0.1),
                     xlabel=r"$\varphi$", ylabel="Da",
                     title="Stability diagram")
    ax_stability.text(c_out+0.02, 0.93, r"$\varphi_{\infty}$", color="gray",
         transform=ax_stability.get_xaxis_transform(),
         ha="left", va="top")

    marker_history.append((float(mean_value), float(DA)))
    for i, (mx, my) in enumerate(marker_history):
        age = len(marker_history) - 1 - i
        frac = min(1.0, age / max(FADE_STEPS, 1))
        grey = FADE_GREY_MAX * frac
        alpha = 1.0 - (1.0 - FADE_ALPHA_MIN) * frac
        ax_stability.plot(
            mx, my, marker="x", ls="none", ms=4,
            color=(grey, grey, grey), alpha=alpha
        )
    
    ax_stability.legend(
        [(coex_line[0], coex_fill), (crit_line[0], crit_fill)],
        ["Coexistence", "Critical"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        frameon=False,
        fontsize=7,
        loc="upper right"
    ) 
        
    ax_stability.set_ylim(0, 0.07)
    ax_stability.set_xlim(-1.0, 1.0)
    ax_stability.tick_params(axis="both", which="both", direction="in",top=True,right=True)
    ax_stability.set_xticks([-1.0,-0.8,-0.6,-1/np.sqrt(3),-0.4,-0.2,0.0,
                             0.2,0.4,0.6,1/np.sqrt(3),0.8,1.0])
    ax_stability.set_xticklabels([-1.0,"","","-0.5774","",-0.2,0.0,0.2,"","0.5774","","",1.0])
    colorst = ["black","black","black","red","black","black",
              "black","black","black", "red","black","black","black"]
    for label, colort in zip(ax_stability.get_xticklabels(), colorst):
        label.set_color(colort)
    for tick, x in zip(ax_stability.xaxis.get_major_ticks(),
                   ax_stability.get_xticks()):
        if np.isclose(abs(x), 1/np.sqrt(3)):
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)

    for x in [-1/np.sqrt(3), 1/np.sqrt(3)]:
        ax_stability.plot(x, 0, marker='^', color='red',
                      markersize=5, clip_on=False,
                      transform=ax_stability.get_xaxis_transform())

    xs = np.round(np.arange(0.05, 0.23, 0.01), 2)
    ys = np.round(np.arange(0.1, 2.0, 0.1), 2)
    colors = {"none": "#111111", "stable": "#244b91",
              "two": "#f00000", "three": "#e00000",
              "elongation": "#e00000"}
    markers = {"none": ".", "stable": "o", "two": "o",
               "three": "^", "elongation": "s"}
    sizes = {"none": 20, "stable": 40, "two": 40,
             "three": 40, "elongation": 40}
    for key in ("none", "stable", "two", "three", "elongation"):
        pts = [(xx, yy) for xx in xs for yy in ys
               if _phase_region(xx, yy) == key]
        if pts:
            px, py = np.asarray(pts).T
            ax_phase.scatter(px, py, s=sizes[key], c=colors[key],
                             marker=markers[key], linewidths=0)
    current_key = _phase_region(EPSILON, ALPHA)
    ax_phase.scatter([EPSILON], [ALPHA], s=sizes[current_key],
                     marker=markers[current_key],
                     facecolors=colors[current_key], edgecolors="black",
                     linewidths=1.6, zorder=10)
    ax_phase.set_title("Droplet phase diagram", fontsize=10, pad=8)
    ax_phase.set_xlim(0.045, 0.225)
    ax_phase.set_ylim(0.0, 2.0)
    ax_phase.set_xlabel(r"Supersaturation, $\epsilon$", fontsize=10)
    ax_phase.set_ylabel(r"Turnover, $\nu_-/\nu_0$", fontsize=10)
    ax_phase.set_xticks([0.1, 0.2])
    ax_phase.set_yticks(np.arange(0, 2.01, 0.5))
    ax_phase.tick_params(labelsize=8, axis="both", which="both",
                         direction="in", top=True, right=True,
                         length=3, width=0.7)
    for spine in ax_phase.spines.values():
        spine.set_linewidth(0.7)
    legend = [
        Line2D([], [], marker=".", linestyle="None", color="black", markersize=3,
               label="No\ndroplets"),
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="#244b91",
               markeredgecolor="#244b91", markersize=3.5, label="Stable\ndroplets"),
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="#f00000",
               markeredgecolor="#f00000", markersize=3.5, label="Division into\n2 droplets"),
        Line2D([], [], marker="^", linestyle="None", markerfacecolor="#e00000",
               markeredgecolor="#e00000", markersize=3.5, label="Division into\n3 droplets"),
        Line2D([], [], marker="s", linestyle="None", markerfacecolor="#e00000",
               markeredgecolor="#e00000", markersize=3.5, label="Droplet\nelongation"),
    ]
    ax_phase.legend(handles=legend, loc="center left", bbox_to_anchor=(1.04, 0.5),
                    frameon=False, fontsize=7, handlelength=0.6,
                    handletextpad=0.5, borderaxespad=0.0, labelspacing=0.9)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_spectral(c_hat, sim_time, filename):
    """Three orthogonal Fourier slices and their 3D arrangement."""
    nk = min(SPECTRAL_CROP_MODES, NX//2 - 1, NY//2 - 1, NZ//2 - 1)
    raw = np.abs(np.fft.fftshift(np.asarray(c_hat)))
    raw = raw / float(NX * NY * NZ)
    centre = np.array(raw.shape) // 2
    sl = tuple(slice(int(v-nk), int(v+nk+1)) for v in centre)
    amp = np.log10(raw[sl] + 1e-12)
    levels = np.linspace(-8, 0, 17)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(levels[0], levels[-1])
    kaxes = [
        np.fft.fftshift(2*np.pi*np.fft.fftfreq(NX, d=LX/NX))[NX//2-nk:NX//2+nk+1],
        np.fft.fftshift(2*np.pi*np.fft.fftfreq(NY, d=LY/NY))[NY//2-nk:NY//2+nk+1],
        np.fft.fftshift(2*np.pi*np.fft.fftfreq(NZ, d=LZ/NZ))[NZ//2-nk:NZ//2+nk+1],
    ]
    i0 = nk
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_yz = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1], projection="3d")
    kw = dict(levels=levels, cmap=cmap, extend="both")
    ax_xy.contourf(kaxes[0], kaxes[1], amp[:, :, i0].T, **kw)
    ax_yz.contourf(kaxes[1], kaxes[2], amp[i0, :, :].T, **kw)
    ax_xz.contourf(kaxes[0], kaxes[2], amp[:, i0, :].T, **kw)
    ax_xy.set(xlabel=r"$k_x$", ylabel=r"$k_y$", title=rf"$k_z=0$, t={sim_time:.3g}")
    ax_yz.set(xlabel=r"$k_y$", ylabel=r"$k_z$", title=r"$k_x=0$")
    ax_xz.set(xlabel=r"$k_x$", ylabel=r"$k_z$", title=r"$k_y=0$")
    for ax in (ax_xy, ax_yz, ax_xz):
        ax.set_aspect("equal")
    KX, KY = np.meshgrid(kaxes[0], kaxes[1], indexing="ij")
    KX2, KZ = np.meshgrid(kaxes[0], kaxes[2], indexing="ij")
    KY2, KZ2 = np.meshgrid(kaxes[1], kaxes[2], indexing="ij")
    surf_kw = dict(shade=False, linewidth=0, antialiased=False)
    ax3.plot_surface(KX, KY, np.zeros_like(KX),
                     facecolors=cmap(norm(amp[:, :, i0])), **surf_kw)
    ax3.plot_surface(KX2, np.zeros_like(KX2), KZ,
                     facecolors=cmap(norm(amp[:, i0, :])), **surf_kw)
    ax3.plot_surface(np.zeros_like(KY2), KY2, KZ2,
                     facecolors=cmap(norm(amp[i0, :, :])), **surf_kw)
    ax3.set(xlabel=r"$k_x$", ylabel=r"$k_y$", zlabel=r"$k_z$")
    ax3.set_box_aspect((1, 1, 1))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=[ax_xy, ax_yz, ax_xz, ax3],
                 shrink=0.8, label=r"$\log_{10}|\hat \varphi |/N^3$")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)




# =============================================================================
# 7. MAIN SIMULATION
# =============================================================================


def main():
    # Create all the directories.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_2d = OUTPUT_DIR / "frames_2d"
    frames_3d = OUTPUT_DIR / "frames_3d"
    frames_spectral = OUTPUT_DIR / "frames_spectral"
    for folder in (frames_2d, frames_3d, frames_spectral):
        folder.mkdir(parents=True, exist_ok=True)

    # Useful starting messages.
    print("=" * 72)
    print("3D reaction--Cahn--Hilliard droplet simulation")
    print("=" * 72)
    print(f"Grid: {NX} x {NY} x {NZ}")
    print(f"Domain: {LX:g} x {LY:g} x {LZ:g}")
    print(f"Grid spacing: {LX/NX:.6g}, {LY/NY:.6g}, {LZ/NZ:.6g}")
    print("-" * 72)
    print(f"Reaction parameters: epsilon={EPSILON:g}, alpha={ALPHA:g}, Da={DA:g}")
    c_out = 2.0 * EPSILON - 1.0
    print(f"Reaction at c_out={c_out:.6g}: {float(reaction_flux_numpy(c_out)):.6g}")
    print(f"Reaction at c_in={C_IN:.6g}: {float(reaction_flux_numpy(C_IN)):.6g}")
    print("-" * 72)

    x = np.arange(NX, dtype=np.float32) * (LX/NX) - 0.5*LX
    y = np.arange(NY, dtype=np.float32) * (LY/NY) - 0.5*LY
    z = np.arange(NZ, dtype=np.float32) * (LZ/NZ) - 0.5*LZ
    c0 = make_initial_condition(x, y, z)
    np.save(OUTPUT_DIR / "initial_field.npy", c0)

    # State of the integrator: the REAL field.
    c_state = jnp.asarray(c0, dtype=jnp.float32)

    # Fourier wave numbers and 2/3-rule mask for nonlinear products.
    kx = (2*np.pi*np.fft.fftfreq(NX, d=LX/NX)).astype(np.float32)
    ky = (2*np.pi*np.fft.fftfreq(NY, d=LY/NY)).astype(np.float32)
    kz = (2*np.pi*np.fft.fftfreq(NZ, d=LZ/NZ)).astype(np.float32)

    k2 = (kx[:, None, None]**2 + ky[None, :, None]**2
          + kz[None, None, :]**2).astype(np.float32)
    k4 = (k2*k2).astype(np.float32)

    if DEALIAS_NONLINEAR_TERMS:
        mx = np.abs(np.fft.fftfreq(NX) * NX) <= NX/3
        my = np.abs(np.fft.fftfreq(NY) * NY) <= NY/3
        mz = np.abs(np.fft.fftfreq(NZ) * NZ) <= NZ/3
        dealias_mask = (mx[:, None, None] & my[None, :, None]
                        & mz[None, None, :]).astype(np.float32)
    else:
        dealias_mask = np.ones((NX, NY, NZ), dtype=np.float32)

    # Compute the spectral step.
    step = make_spectral_stepper(k2, k4, dealias_mask)

    times = [0.0]
    mean_concentrations = [float(np.mean(c0, dtype=np.float64))]
    mean_absolute_values = [float(np.mean(np.abs(c0), dtype=np.float64))]
    equivalent_radii = [equivalent_positive_phase_radius(c0)]
    snapshot_slices = [(0.0, c0[:, :, NZ // 2].copy())]
    print(f"Initial mean concentration: {mean_concentrations[-1]:.8g}")
    print(f"Initial positive-phase equivalent radius: {equivalent_radii[-1]:.6g}")
    print("=" * 72)
    print(f"Integrating up to {N_STEPS} accepted steps; initial dt={DT:g}, "
          f"dt range=[{DT_MIN:g}, {DT_MAX:g}]")
    print("=" * 72)

    # Setup time-stepper.
    marker_history = []
    accepted = 0
    rejected_total = 0
    sim_time = 0.0
    dt = min(DT, DT_MAX)
    start_time = time.perf_counter()

    _draw_restored_2d(c0, 0.0, mean_concentrations[0],
                          frames_2d / "c_00000000.png")
    _draw_restored_3d(c0, 0.0, mean_concentrations[0], marker_history,
                          frames_3d / "c3d_00000000.png")
    _draw_spectral(jnp.fft.fftn(c_state), 0.0,
                       frames_spectral / "ck_00000000.png")

    save_midplane_plot(c0, OUTPUT_DIR / "initial_midplane.png",
                       r"Initial $\varphi$: central z slice")
    save_surface_plot(c0, OUTPUT_DIR / "initial_surface.png",
                      rf"Initial isosurface $\varphi={ISO_LEVEL:g}$",
                      times=times, mean_absolute_values=mean_absolute_values)
    save_mean_absolute_log_plot(
        times, mean_absolute_values, OUTPUT_DIR / "mean_absolute_phi_log.png"
    )

    # Time loop starts here.
    print("Time-stepping starts now.")
    while accepted < N_STEPS:
        trial, max_change_jax, finite_jax = step(
            c_state, jnp.asarray(dt, dtype=jnp.float32)
        )
        max_change = float(max_change_jax)
        finite = bool(finite_jax)

        # Reject unstable/non-finite trial states before committing them.
        if (not finite) or (not np.isfinite(max_change)):
            dt *= 0.5
            rejected_total += 1
        elif max_change > 2.0 * DT_TARGET_CHANGE and dt > DT_MIN:
            factor = np.clip(
                DT_SAFETY * np.sqrt(DT_TARGET_CHANGE / max(max_change, 1e-14)),
                0.2, 0.8
            )
            dt = max(DT_MIN, dt * float(factor))
            rejected_total += 1
        else:
            c_state = trial
            accepted += 1
            sim_time += dt

            factor = DT_SAFETY * np.sqrt(
                DT_TARGET_CHANGE / max(max_change, 1e-14)
            )
            dt = float(np.clip(dt * np.clip(factor, 0.5, 1.2),
                               DT_MIN, DT_MAX))

            if accepted % SAVE_EVERY == 0 or accepted == N_STEPS:
                c_now = np.asarray(c_state, dtype=np.float32)
                if not np.isfinite(c_now).all():
                    raise FloatingPointError(
                        f"Non-finite concentration after accepted step {accepted}; "
                        "the trial should have been rejected."
                    )
                mean_now = float(np.mean(c_now, dtype=np.float64))
                phi_min_now = float(np.min(c_now))
                phi_max_now = float(np.max(c_now))
                radius_now = equivalent_positive_phase_radius(c_now)
                times.append(sim_time)
                mean_concentrations.append(mean_now)
                mean_absolute_values.append(
                    float(np.mean(np.abs(c_now), dtype=np.float64))
                )
                equivalent_radii.append(radius_now)
                save_mean_absolute_log_plot(
                    times, mean_absolute_values,
                    OUTPUT_DIR / "mean_absolute_phi_log.png"
                )
                snapshot_slices.append((sim_time, c_now[:, :, NZ // 2].copy()))
                elapsed = time.perf_counter() - start_time
                print(
                    f"step {accepted:7d}/{N_STEPS} | t={sim_time:11.5g} | "
                    f"dt={dt:.3e} | max|dc|={max_change:.3e} | "
                    f"mean(phi)={mean_now:.6e} | "
                    f"range(phi)=[{phi_min_now:.4g}, {phi_max_now:.4g}] | "
                    f"R_eq={radius_now:.6g} | rejected={rejected_total} | "
                    f"elapsed={elapsed:.1f}s"
                )

                tag = f"{accepted:08d}"
                _draw_restored_2d(
                    c_now, sim_time, mean_now, frames_2d / f"c_{tag}.png"
                )
                _draw_restored_3d(
                    c_now, sim_time, mean_now, marker_history,
                    frames_3d / f"c3d_{tag}.png"
                )
                _draw_spectral(
                    jnp.fft.fftn(c_state), sim_time,
                    frames_spectral / f"ck_{tag}.png"
                )
                del c_now

        if dt < DT_MIN * (1.0 - 1e-8):
            raise FloatingPointError(
                f"Adaptive timestep fell below DT_MIN={DT_MIN:g} at "
                f"accepted step {accepted}, t={sim_time:g}. "
                "Try a smoother initial interface, smaller DT_TARGET_CHANGE, "
                "or a larger grid resolution."
            )
        if rejected_total > 0 and rejected_total % 100 == 0:
            print(f"  adaptive controller: {rejected_total} rejected trials; "
                  f"current dt={dt:.3e}")
        if rejected_total > MAX_REJECTIONS * max(1, accepted + 1):
            raise FloatingPointError(
                "Too many rejected steps. The explicit nonlinear terms remain "
                "too stiff for the configured timestep limits."
            )

    c_final = np.asarray(c_state, dtype=np.float32)
    np.save(OUTPUT_DIR / "final_field.npy", c_final)
    np.savez_compressed(
        OUTPUT_DIR / "diagnostics.npz",
        time=np.asarray(times),
        mean_concentration=np.asarray(mean_concentrations),
        mean_absolute_value=np.asarray(mean_absolute_values),
        equivalent_radius=np.asarray(equivalent_radii),
    )

    final_time = sim_time
    save_midplane_plot(c_final, OUTPUT_DIR / "final_midplane.png",
                       rf"Final $\varphi$ at t={final_time:g}: central z slice")
    save_surface_plot(c_final, OUTPUT_DIR / "final_surface.png",
                      rf"Final isosurface $\varphi={ISO_LEVEL:g}$ at t={final_time:g}",
                      times=times, mean_absolute_values=mean_absolute_values)
    save_profile_plot(c_final, OUTPUT_DIR / "final_centreline_profiles.png",
                      f"Final central Cartesian cuts at t={final_time:g}")
    save_diagnostics_plot(times, mean_concentrations, equivalent_radii,
                          OUTPUT_DIR / "time_diagnostics.png")
    save_mean_absolute_log_plot(
        times, mean_absolute_values, OUTPUT_DIR / "mean_absolute_phi_log.png"
    )

    n_snaps = len(snapshot_slices)
    fig, axes = plt.subplots(
        1, n_snaps, figsize=(4.0*n_snaps, 4.0), squeeze=False
    )
    for ax, (snap_time, snap) in zip(axes[0], snapshot_slices):
        im = ax.imshow(
            snap.T, origin="lower",
            extent=(-LX/2, LX/2, -LY/2, LY/2),
            cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
        )
        ax.contour(
            np.linspace(-LX/2, LX/2, NX),
            np.linspace(-LY/2, LY/2, NY), snap.T,
            levels=[ISO_LEVEL], colors="black", linewidths=0.6
        )
        ax.set_title(f"t={snap_time:.3g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.colorbar(im, ax=list(axes[0]), label=r"Dimless concentration density $\varphi$", shrink=0.8)
    fig.suptitle(r"Evolution of the central $\varphi$ slice")
    fig.savefig(OUTPUT_DIR / "slice_evolution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nSimulation complete.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Final time: {final_time:.8g}; accepted steps: {accepted}; "
          f"rejected trials: {rejected_total}")
    print(f"Final concentration range: [{c_final.min():.6g}, {c_final.max():.6g}]")
    print(f"Final mean concentration: {np.mean(c_final, dtype=np.float64):.8g}")

    if SHOW_PLOTS:
        for image_name in (
            "initial_midplane.png", "initial_surface.png", "final_midplane.png",
            "final_surface.png", "final_centreline_profiles.png",
            "time_diagnostics.png", "slice_evolution.png"
        ):
            path = OUTPUT_DIR / image_name
            if path.exists():
                img = plt.imread(path)
                plt.figure(figsize=(9, 6))
                plt.imshow(img)
                plt.axis("off")
                plt.title(image_name)
        plt.show()


if __name__ == "__main__":
    main()
