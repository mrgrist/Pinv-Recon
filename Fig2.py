import numpy as np
from scipy.linalg import svd
import time
from phantominator import shepp_logan
import matplotlib.pyplot as plt

svd_threshold = 0.99
mtx=32
ntx=32
plt_svd = 0

# Generate TMP and create the grid
TMP = np.linspace(-1, 1, ntx + 1)
Rx, Ry = np.meshgrid(TMP[:-1], TMP[:-1])

# Create the circular mask
circMASK = np.sqrt(Rx**2 + Ry**2) < 1

M0, T1, T2 = shepp_logan((mtx, mtx,1), MR=True, zlims=(-.25, .25))

IMG = M0

# Normalize the image
IMG = IMG.astype(np.float64) / np.max(IMG)

# Display message
print('FW>> 1. Equidistant Cartesian Sampling')

# k-space sampling
TMP = np.linspace(-mtx/2, mtx/2, mtx+1)
Kx, Ky = np.meshgrid(TMP[1:], TMP[1:])
Kx1 = Kx
Ky1 = Ky

# Encoding matrix exp(i*k*r)
ENCODE1 = np.exp(1j * np.pi * (Kx.flatten()[:, None] * Rx.flatten() + Ky.flatten()[:, None] * Ry.flatten()))

# SVD
start_time = time.time()

U, Sl, Vh = svd(ENCODE1, full_matrices=False)

print("SVD computation time:", time.time() - start_time)

S = np.diag(Sl)  

V = Vh.T

svdE = np.diag(S)

if plt_svd:
    plt.figure()
    plt.plot(svdE)
    plt.title('SVD Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

cumulative_energy = np.cumsum(svdE,axis=0) / np.sum(svdE)

if plt_svd:
    plt.figure()
    plt.plot(cumulative_energy)
    plt.title('cumulative energy')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

# Find the index where cumulative energy exceeds the threshold
g = np.argmax(cumulative_energy >= svd_threshold)

# If g is empty, set it to the size of svdE
if cumulative_energy[g] < svd_threshold:
    g = len(svdE) - 1

# Compute invS

invS = np.copy(svdE)

invS = 1.0 / invS

invS[g+1:] = 0

invS = np.diag(invS)

# Reconstruction
start_time = time.time()

RECON1 = Vh.T @ invS @ U.T

print("Reconstruction computation time:", time.time() - start_time)

IMG_flat = IMG.flatten()

IMG1 = RECON1 @ (ENCODE1 @ IMG_flat)

IMG1 = IMG1.reshape(ntx, ntx)

NOISE1 = np.sqrt(np.diag(RECON1 @ RECON1.T).reshape(ntx, ntx) / RECON1.shape[1])

SRF1 = np.diag(RECON1 @ ENCODE1).reshape(ntx, ntx)

combined_image = np.abs(IMG1)

#EPI 
# K-space sampling
print('FW>> 2. EPI Sampling')

Gread = np.concatenate([np.linspace(0, 1, 4), np.ones(mtx), np.linspace(1, 0, 4)])
Gphas = np.concatenate([np.linspace(1, 0, 4), np.zeros(mtx), np.linspace(0, 1, 4)]) / 4
Gx = Gread[4:] 
Gy = Gphas[4:]  

for i1 in range(1, mtx): 
    Gx = np.concatenate([Gx, Gread * (-1) ** i1])
    Gy = np.concatenate([Gy, Gphas])

Kx = np.cumsum(Gx) - mtx
Ky = np.cumsum(Gy) - mtx
Kx2 = Kx
Ky2 = Ky

temp = Kx2[:, None]

# Encoding matrix
ENCODE2 = np.exp(1j * np.pi * (Kx2[:, None] * Rx.flatten() + Ky2[:, None] * Ry.flatten()))

# SVD
start_time = time.time()

U, Sl, Vh = svd(ENCODE2, full_matrices=False)

print("SVD computation time:", time.time() - start_time)

S = np.diag(Sl)  

V = Vh.T

svdE = np.diag(S)

if plt_svd:
    plt.figure()
    plt.plot(svdE)
    plt.title('SVD Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

cumulative_energy = np.cumsum(svdE,axis=0) / np.sum(svdE)

if plt_svd:
    plt.figure()
    plt.plot(cumulative_energy)
    plt.title('cumulative energy')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

# Find the index where cumulative energy exceeds the threshold
g = np.argmax(cumulative_energy >= svd_threshold)

# If g is empty, set it to the size of svdE
if cumulative_energy[g] < svd_threshold:
    g = len(svdE) - 1

# Compute invS

invS = np.copy(svdE)

invS = 1.0 / invS

invS[g+1:] = 0

invS = np.diag(invS)

# Reconstruction
start_time = time.time()

RECON2 = Vh.T @ invS @ U.T

print("Reconstruction computation time:", time.time() - start_time)

IMG_flat = IMG.flatten()

IMG2 = RECON2 @ (ENCODE2 @ IMG_flat)

IMG2 = IMG2.reshape(ntx, ntx)

NOISE2 = np.sqrt(np.diag(RECON2 @ RECON2.T).reshape(ntx, ntx) / RECON2.shape[1])

SRF2 = np.diag(RECON2 @ ENCODE2).reshape(ntx, ntx)

combined_image = np.abs(IMG2)

# Calculate parameters
print('FW>> 3. Golden angle radial Sampling')
gaPhi = np.pi * (3 - np.sqrt(5))
nSpk = mtx
TMP = np.arange(-mtx/2, mtx/2, 1)  # Equivalent to -mtx/2:1:mtx/2
iPhi = 0
Kx3 = []
Ky3 = []

for iSpk in range(nSpk):
    Kx3.append(TMP * np.cos(iPhi))
    Ky3.append(TMP * np.sin(iPhi))
    iPhi += gaPhi

Kx3 = np.array(Kx3).flatten()
Ky3 = np.array(Ky3).flatten()

# Encoding matrix
ENCODE3 = np.exp(1j * np.pi * (Kx3[:,None]* Rx.flatten() + Ky3[:,None] * Ry.flatten()))

# SVD
start_time = time.time()

U, Sl, Vh = svd(ENCODE3, full_matrices=False)

print("SVD computation time:", time.time() - start_time)

S = np.diag(Sl)  

svdE = np.diag(S)

if plt_svd:
    plt.figure()
    plt.plot(svdE)
    plt.title('SVD Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

cumulative_energy = np.cumsum(svdE,axis=0) / np.sum(svdE)

if plt_svd:
    plt.figure()
    plt.plot(cumulative_energy)
    plt.title('cumulative energy')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()
    
# Find the index where cumulative energy exceeds the threshold
g = np.argmax(cumulative_energy >= svd_threshold)

# If g is empty, set it to the size of svdE
if cumulative_energy[g] < svd_threshold:
    g = len(svdE) - 1

# Compute invS

invS = np.copy(svdE)

invS = 1.0 / invS

invS[g+1:] = 0

invS = np.diag(invS)

# Reconstruction
start_time = time.time()

RECON3 = V @ invS @ U.T

print("Reconstruction computation time:", time.time() - start_time)

IMG_flat = IMG.flatten()

IMG3 = RECON3 @ (ENCODE3 @ IMG_flat)

IMG3 = IMG3.reshape(ntx, ntx)

NOISE3 = np.sqrt(np.diag(RECON3 @ RECON3.T).reshape(ntx, ntx) / RECON3.shape[1])

SRF3 = np.diag(RECON3 @ ENCODE3).reshape(ntx, ntx)

IMG4 = IMG1
IMG5 = IMG1

SRF4 = SRF1
SRF5 = SRF1

NOISE4 = NOISE1
NOISE5 = NOISE1

FntSz = 12

# Create the figure
plt.figure(99, figsize=(11.14, 11.21))
plt.subplots_adjust(hspace=0.3)

# Subplot for IMAGE
plt.subplot(3, 1, 1)
plt.imshow(np.abs(np.concatenate((IMG1/IMG1.max(), IMG2/IMG2.max(), IMG3/IMG3.max()), axis=1)), vmin=-0.02, vmax=1.02, cmap='gray')
plt.axis('image')
plt.axis('off')
cbar = plt.colorbar(ticks=np.arange(0, 1.25, 0.25), orientation='horizontal', pad=0.04)
cbar.ax.tick_params(labelsize=FntSz)
cbar.set_label('', fontsize=FntSz)

# Subplot for SRF
plt.subplot(3, 1, 2)
plt.imshow(np.abs(np.concatenate((SRF1, SRF2, SRF3), axis=1)), vmin=-0.05, vmax=1.02, cmap='gray')
plt.axis('image')
plt.axis('off')
cbar = plt.colorbar(ticks=np.arange(0, 1.25, 0.25), orientation='horizontal', pad=0.04)
cbar.ax.tick_params(labelsize=FntSz)
cbar.set_label('', fontsize=FntSz)

# Subplot for NOISE
plt.subplot(3, 1, 3)
plt.imshow(np.abs(np.concatenate((NOISE1, NOISE2, NOISE3), axis=1)), vmin=-0.02 * 1e-4, vmax=1.02 * 1e-4, cmap='gray')
plt.axis('image')
plt.axis('off')
cbar = plt.colorbar(ticks=1e-4 * np.arange(0, 1.25, 0.25), orientation='horizontal', pad=0.04)
cbar.ax.tick_params(labelsize=FntSz)
cbar.set_label('', fontsize=FntSz)

plt.show()
