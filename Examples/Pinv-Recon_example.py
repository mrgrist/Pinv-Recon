# Scientific Reports paper submission Fig. 3 (Python version)
# Pinv-Recon simulations using the numerical Shepp-Logan phantom
# combining different encoding and distortion mechanisms

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# -----------------------------
# Helpers
# -----------------------------
def interpn_2d(x_grid, y_grid, Z, Xq, Yq, method="linear", fill_value=0.0):
    # RegularGridInterpolator expects axes order (y, x) for Z with ndgrid convention.
    interp = RegularGridInterpolator((y_grid, x_grid), Z,
                                     method=method, bounds_error=False,
                                     fill_value=fill_value)
    pts = np.column_stack([Yq.ravel(), Xq.ravel()])
    out = interp(pts)
    return out.reshape(Xq.shape)

def svd_pinv_threshold(A, condNumb):
    """
    Pseudo-inverse via SVD with truncation
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    smax = s.max() if s.size else 0.0
    keep = s >= (smax / condNumb) if smax > 0 else np.zeros_like(s, dtype=bool)
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    return (Vh.conj().T @ np.diag(s_inv) @ U.conj().T), (U, s, Vh)

def as_single(x):
    return np.asarray(x, dtype=np.complex64 if np.iscomplexobj(x) else np.float32)

# -----------------------------
# Parameters
# -----------------------------
CondNumb = 30
ntx = 64  # 64,96,128

# -----------------------------
print('>> Loading relevant data...')
# -----------------------------
# Load spiral trajectories (expects variables: Kx0, Ky0, time0, Kx1, Ky1, time1)
vd = loadmat("data/vd_spiral.mat")
Kx = vd["Kx0"].astype(np.float32)
Ky = vd["Ky0"].astype(np.float32)
time0 = vd["time0"].astype(np.float32).ravel()

# spatial discretization (centered and shifted)
xi = np.linspace(-2, 2, ntx+1)[:-1].astype(np.float32)
dri = np.mean(np.diff(xi))
# MATLAB ndgrid => (Y, X) order; we'll follow that (YM, XM)
XM, YM = np.meshgrid(xi, xi, indexing='ij')

# Generating phantom (Modified Shepp-Logan)
phantom_m = shepp_logan_phantom()  # returns Modified Shepp-Logan at 400x400
imgPhantom = resize(phantom_m, (ntx, ntx), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
imgPhantom[imgPhantom > 0.75] = 0.75
imgPhantom = (imgPhantom / imgPhantom.max()).astype(np.float32)

# -----------------------------
print('>> Reconstructing ...')
# -----------------------------
# Build encoding matrices (vectorized like MATLAB Kx(:)*XM(:).' + Ky(:)*YM(:).')
def build_encode(Kx, Ky, XM, YM):
    kx = Kx.ravel()[:, None]
    ky = Ky.ravel()[:, None]
    grid_x = XM.ravel()[None, :]
    grid_y = YM.ravel()[None, :]
    phase = 1j*np.pi*(kx @ grid_x + ky @ grid_y)
    return as_single(np.exp(phase))

ENCODE = build_encode(Kx, Ky, XM, YM)

RECON, (U00, S00, Vh00) = svd_pinv_threshold(ENCODE, CondNumb)
IMG = (RECON @ (ENCODE @ imgPhantom.ravel())).reshape(ntx, ntx)

# -----------------------------
print('>> Plotting all results...')
# -----------------------------
fig, axes = plt.subplots(1, 4, figsize=(13.5, 11.0), dpi=100)
FntSz = 12

# Spiral trajectory plot
axes[0].plot(Kx.ravel(), Ky.ravel(), '-k', linewidth=0.5)
axes[0].set_xticks(np.linspace(-ntx/2, ntx/2, 5))
axes[0].set_yticks(np.linspace(-ntx/2, ntx/2, 5))
axes[0].set_aspect('equal', 'box')
axes[0].grid(True)
axes[0].set_xlim([-72, 72])
axes[0].set_ylim([-72, 72])
axes[0].set_title('Trajectory', fontsize=18)
axes[0].tick_params(labelsize=FntSz)

# Singular value plots (log scale)
axes[1].plot(S00, '-k', linewidth=2)
axes[1].set_yscale('log')
axes[1].set_xticks([1,12000,24000])
axes[1].set_xticklabels([str(x) for x in [1,12000,24000]])
axes[1].set_yticks(10.0**np.arange(-1, 4, 2))
axes[1].grid(True)
axes[1].set_xlim([1, len(S00)])
axes[1].set_ylim([1e-1, 1e3])
axes[1].tick_params(labelsize=FntSz)
axes[1].set_title('Singular values', fontsize=18)

# image
axes[2].imshow(np.abs(imgPhantom), cmap='gray', vmin=-0.02, vmax=1.02, aspect='equal')
axes[2].set_title('Generated image', fontsize=18)

# recontructed
axes[3].imshow(np.abs(IMG), cmap='gray', vmin=-0.02, vmax=1.02, aspect='equal')
axes[3].set_title('Reconstructed image', fontsize=18)

plt.show()
