"""
Reaction-diffusion fields on a 2D grid, batched over B independent tissues.

Every field obeys one equation (protocol step 12):

    dc/dt = div( D(x) grad c ) - k(x) c + S(x, t)

with no-flux edges, where k(x) = lam + mu * rho(x) (clearance plus uptake by
cells) and S is production by cells, vessels, or the basal background source.

Shapes: concentration fields are (B, ny, nx) float64 tensors.

Two solvers:
  explicit_step  - the old forward-Euler step, kept only for the legacy port.
  implicit_step  - backward Euler, solved with batched conjugate gradient.
                   Stable at any step size (protocol step 14).
  steady_state   - the same solver with no time derivative, for oxygen
                   (protocol step 15), with optional fixed values at vessels.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Legacy: constant-D Laplacian with replicate padding (as in pmn_preliminary.py)
# ---------------------------------------------------------------------------
def laplacian_const(c: torch.Tensor, dx: float) -> torch.Tensor:
    """5-point Laplacian, no-flux edges via replicate padding. c: (B, ny, nx)."""
    p = F.pad(c.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    return (p[:, 2:, 1:-1] + p[:, :-2, 1:-1] + p[:, 1:-1, 2:] + p[:, 1:-1, :-2]
            - 4.0 * c) / dx ** 2


def explicit_step(c, D, lam, S, dt, dx, floor=None):
    """Old forward-Euler step. floor reproduces the old clamp-to-basal."""
    c = c + dt * (D * laplacian_const(c, dx) - lam * c + S)
    if floor is not None:
        c = c.clamp(min=floor)
    return c


# ---------------------------------------------------------------------------
# New: variable-coefficient diffusion in conservative (flux) form
# ---------------------------------------------------------------------------
def face_diffusivity(D: torch.Tensor):
    """
    Diffusivity on the faces between neighbouring squares (harmonic mean),
    so that a low-D square (dense CAFs) blocks flux on both sides.
    D: (B, ny, nx). Returns (Dx, Dy) with shapes (B, ny, nx-1), (B, ny-1, nx).
    """
    Dx = 2.0 * D[:, :, 1:] * D[:, :, :-1] / (D[:, :, 1:] + D[:, :, :-1])
    Dy = 2.0 * D[:, 1:, :] * D[:, :-1, :] / (D[:, 1:, :] + D[:, :-1, :])
    return Dx, Dy


def div_D_grad(c, Dx, Dy, dx):
    """div(D grad c) with zero flux through the outer edges. Conserves mass exactly."""
    out = torch.zeros_like(c)
    fx = Dx * (c[:, :, 1:] - c[:, :, :-1]) / dx ** 2
    fy = Dy * (c[:, 1:, :] - c[:, :-1, :]) / dx ** 2
    out[:, :, :-1] += fx
    out[:, :, 1:] -= fx
    out[:, :-1, :] += fy
    out[:, 1:, :] -= fy
    return out


def _diag(Dx, Dy, k, dt, dx, shape):
    """Diagonal of the operator, for the Jacobi preconditioner."""
    s = torch.zeros(shape, dtype=k.dtype, device=k.device)
    s[:, :, :-1] += Dx
    s[:, :, 1:] += Dx
    s[:, :-1, :] += Dy
    s[:, 1:, :] += Dy
    s = s / dx ** 2
    if dt is None:                      # steady state: k c - div(D grad c)
        return k + s
    return 1.0 + dt * (k + s)           # backward Euler: c + dt(k c - div)


def _apply(c, Dx, Dy, k, dt, dx):
    if dt is None:
        return k * c - div_D_grad(c, Dx, Dy, dx)
    return c + dt * (k * c - div_D_grad(c, Dx, Dy, dx))


def _batched_dot(a, b):
    return (a * b).flatten(1).sum(1)


def _cg(apply_A, b, x0, diag, free, tol, max_iter):
    """
    Preconditioned conjugate gradient, batched over the first dimension.
    free: bool mask of unknowns (False = fixed value, e.g. a vessel).
    Returns (x, iterations, final relative residual per batch).
    """
    x = x0.clone()
    r = (b - apply_A(x)) * free
    z = r / diag
    p = z.clone()
    rz = _batched_dot(r, z)
    bnorm = torch.sqrt(_batched_dot(b * free, b * free)).clamp(min=1e-300)
    it = 0
    rel = torch.sqrt(_batched_dot(r, r)) / bnorm
    while it < max_iter and bool((rel > tol).any()):
        Ap = apply_A(p) * free
        alpha = rz / _batched_dot(p, Ap).clamp(min=1e-300)
        active = (rel > tol).to(x.dtype)            # finished tissues stop moving
        x = x + (active * alpha)[:, None, None] * p
        r = r - (active * alpha)[:, None, None] * Ap
        z = r / diag
        rz_new = _batched_dot(r, z)
        beta = rz_new / rz.clamp(min=1e-300)
        p = z + beta[:, None, None] * p
        rz = rz_new
        rel = torch.sqrt(_batched_dot(r, r)) / bnorm
        it += 1
    return x, it, rel


def implicit_step(c, D, k, S, dt, dx, tol=1e-10, max_iter=2000, fixed_mask=None, fixed_value=None):
    """
    One backward-Euler step:  (I + dt*k - dt*div(D grad)) c_new = c + dt*S.
    Warm-starts from the current field (fix 3). D, k, S broadcast to (B, ny, nx).
    fixed_mask/fixed_value pin c_new at chosen squares (vessels for the drug);
    fixed_value is a scalar or a (B,) tensor (one value per tissue).
    """
    D = torch.broadcast_to(torch.as_tensor(D, dtype=c.dtype, device=c.device), c.shape)
    k = torch.broadcast_to(torch.as_tensor(k, dtype=c.dtype, device=c.device), c.shape)
    Dx, Dy = face_diffusivity(D)
    A = lambda v: _apply(v, Dx, Dy, k, dt, dx)
    b = c + dt * S
    diag = _diag(Dx, Dy, k, dt, dx, c.shape)
    if fixed_mask is None:
        free = torch.ones_like(c)
        x, it, rel = _cg(A, b, c, diag, free, tol, max_iter)
        return x, it, rel
    fv = torch.as_tensor(fixed_value, dtype=c.dtype, device=c.device).reshape(-1, 1, 1)
    c_fix = torch.where(fixed_mask, fv.expand_as(c), torch.zeros((), dtype=c.dtype, device=c.device))
    free = (~fixed_mask).to(c.dtype)
    b = (b - A(c_fix)) * free
    x, it, rel = _cg(A, b, c * free, diag, free, tol, max_iter)
    return x * free + c_fix, it, rel


def steady_state(D, k, S, dx, fixed_mask=None, fixed_value=None, x0=None,
                 tol=1e-10, max_iter=20000):
    """
    Solve  k c - div(D grad c) = S  (no time derivative).
    fixed_mask/fixed_value pin c at chosen squares (vessels for oxygen).
    """
    S = torch.as_tensor(S)
    shape = S.shape
    D = torch.broadcast_to(torch.as_tensor(D, dtype=S.dtype, device=S.device), shape)
    k = torch.broadcast_to(torch.as_tensor(k, dtype=S.dtype, device=S.device), shape)
    Dx, Dy = face_diffusivity(D)
    A = lambda v: _apply(v, Dx, Dy, k, None, dx)
    if fixed_mask is None:
        free = torch.ones(shape, dtype=S.dtype, device=S.device)
        c_fix = torch.zeros(shape, dtype=S.dtype, device=S.device)
    else:
        free = (~fixed_mask).to(S.dtype)
        fv = torch.as_tensor(fixed_value, dtype=S.dtype, device=S.device)
        c_fix = torch.where(fixed_mask, fv.expand_as(fixed_mask) if fv.dim() else fv,
                            torch.zeros((), dtype=S.dtype, device=S.device))
    b = (S - A(c_fix)) * free
    diag = _diag(Dx, Dy, k, None, dx, shape)
    x0 = torch.zeros(shape, dtype=S.dtype, device=S.device) if x0 is None else x0 * free
    x, it, rel = _cg(A, b, x0, diag.clamp(min=1e-300), free, tol, max_iter)
    return x * free + c_fix, it, rel


def basal_source(lam, c0):
    """Background production that holds an empty tissue at its basal level (step 13)."""
    return lam * c0


def pde_residual(c_new, c_old, D, k, S, dt, dx):
    """
    Residual of the backward-Euler equation for a computed step, per tissue,
    relative to the size of the right-hand side. Used by the physics gate.
    """
    D = torch.broadcast_to(torch.as_tensor(D, dtype=c_new.dtype, device=c_new.device), c_new.shape)
    k = torch.broadcast_to(torch.as_tensor(k, dtype=c_new.dtype, device=c_new.device), c_new.shape)
    Dx, Dy = face_diffusivity(D)
    lhs = _apply(c_new, Dx, Dy, k, dt, dx)
    rhs = c_old + dt * S
    num = torch.sqrt(_batched_dot(lhs - rhs, lhs - rhs))
    den = torch.sqrt(_batched_dot(rhs, rhs)).clamp(min=1e-300)
    return num / den
