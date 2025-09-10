
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.ndimage import zoom
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
# Assuming pinv_recon and other necessary functions are already defined

def main():
    # Load data
    os.chdir('/Users/jamesgrist/Documents/GitHub/Pinv-Recon') #To allow it to find the files 
    data = loadmat("data/structural_phantom_mtx96.mat")['data']
    wf = loadmat("data/structural_phantom_mtx96.mat")['wf']
    fieldmap = np.fliplr(np.flipud(loadmat("data/fieldmap.mat")['fieldmap']))

    fieldmap.astype(np.float64)

    # Parameters
    mtx_acq = wf['npix'][0][0]

    mtx_reco = mtx_acq

    print('Welcome to PINV Recon! ')

    # Perform reconstructions
    image, imageabs, Recon = pinv_recon(data, wf, None, mtx_reco,None,0.95,0)

    image_b0, imageabs_b0, Recon_b0 = pinv_recon(data, wf, None, mtx_reco, fieldmap,0.95,0)

    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.fliplr(fieldmap), cmap='gray')
    plt.colorbar()
    plt.title("B_0 map")

    plt.subplot(1, 3, 2)
    plt.imshow(np.fliplr(np.squeeze(imageabs)), cmap='gray')
    plt.title("Pinv")

    plt.subplot(1, 3, 3)
    plt.imshow(np.fliplr(np.squeeze(imageabs_b0)), cmap='gray')
    plt.title("Pinv + OffRes")

    plt.show()

def pinv_recon(data, wf, Recon=None, mtx_reco=None, B0Map=None, svd_threshold=0.95, plt_svd=False):
    """
    Reconstruction with pseudoinverse of encoding matrix.
    
    Parameters:
        data (np.ndarray): Raw data in shape (ncoils, npts, ntimesteps, nslices).
        wf (dict): Waveform dictionary with keys 'k' and 't'.
        Recon (np.ndarray, optional): Recon matrix (2D or 3D).
        mtx_reco (int, optional): Matrix size after reconstruction (assumed square).
        B0Map (np.ndarray, optional): B0 map for off-resonance frequency correction.
        svd_threshold (float, optional): Threshold for pseudoinverse.
        plt_svd (bool, optional): Whether to plot SVD values.

    Returns:
        image (np.ndarray): Complex image.
        imageabs (np.ndarray): Absolute image with coil combination.
        Recon (np.ndarray): Reconstruction matrix.
    """

    pinv_start_time = time.time()
    
    # Extract dimensions and pre-define values
    data_shape = data.shape
    ncoils = 1
    npts = 1
    ntimesteps = 1
    nslices = 1

    for i, dim in enumerate(data_shape):
        if i == 0: 
            ncoils = dim

        if i == 1:
            npts = dim
            
        if i == 2:
            ntimesteps = dim
            
        if i == 3:
            nslices = dim

    maxdim = i 

    if Recon is None:
        Recon, pinv_time = wf2Recon(wf, mtx_reco, B0Map, svd_threshold, plt_svd,nslices)
    else:
        pinv_time = [0]
    
    mtx_reco = int(np.sqrt(Recon.shape[0]))

    image = np.zeros((mtx_reco, mtx_reco, nslices, ntimesteps, ncoils), dtype=complex)
    
    if Recon.ndim == 3:
        for ls in range(nslices):
            dd_s = data[:, :, :, ls].reshape((npts, ntimesteps * ncoils)).T
            data_inverted = np.dot(Recon[:, :, ls], dd_s)
            image[:, :, ls, :, :] = data_inverted.reshape((mtx_reco, mtx_reco, ntimesteps, ncoils))
    else:
        if maxdim == 1:
            data = data.transpose(1, 0).reshape((npts, ncoils))
        if maxdim == 2: 
            data = data.transpose(1, 2, 0).reshape((npts, ntimesteps * ncoils))
        if maxdim == 3:
            data = data.transpose(1, 3, 2, 0).reshape((npts, nslices * ntimesteps * ncoils))
        
        data_inverted = np.dot(Recon, data)
        
        image = data_inverted.reshape((mtx_reco, mtx_reco, nslices, ntimesteps, ncoils))
    
    if ncoils > 1:
        imageabs = np.sqrt(np.mean(np.abs(image) ** 2, axis=4))
    else:
        imageabs = np.abs(image)
    
    # Print timing
    print('recon_pinv: runtime = ', time.time() - pinv_start_time, ' s')
    
    return image, imageabs, Recon

def wf2Recon(wf, mtx_reco, B0Map, svd_threshold, plt_svd,nslices):

    k = wf['k'][0][0]

    t = wf['t'][0][0]

    if not np.isrealobj(k):

        if k.shape[0] != 1:
            raise ValueError('size(k,1) != 1')
        
        k_temp = k.copy()

        k = np.zeros((k.shape[0], k.shape[1], 2), dtype=np.float64)

        k[0, :, 0] = np.real(k_temp)
        k[0, :, 1] = np.imag(k_temp)
    
    mtx_acq = wf['npix'][0][0]

    if mtx_reco is None:
        mtx_reco = mtx_acq

    XM, YM = np.meshgrid(np.linspace(-((mtx_reco - 1) / 2), ((mtx_reco - 1) / 2),mtx_reco[0][0]),
                         np.linspace(-((mtx_reco - 1) / 2), ((mtx_reco - 1) / 2),mtx_reco[0][0]), indexing='ij')
    
    Kx = k[0, :, 0] / 0.5 * np.pi / mtx_reco * mtx_acq

    Ky = k[0, :, 1] / 0.5 * np.pi / mtx_reco * mtx_acq
    
    E = np.exp(1j * (Kx.flatten()[:, None] * XM.flatten() + Ky.flatten()[:, None] * YM.flatten()))
    
    if B0Map is not None:

        print('Getting encoding matrix with B0 encoding...')

        x = mtx_reco[0][0] / B0Map.shape[0]
            
        y = mtx_reco[0][0] / B0Map.shape[1]

        t_matrix =  t[0,:].reshape((-1, 1))

        if nslices > 1:

            Recon = np.zeros((E.shape[1], E.shape[0], nslices), dtype=complex)

            svd_time = []

            for ls in range(nslices):

                B0Map_s = np.flipud(zoom(B0Map[:, :, ls], (x,y), order=1,prefilter=False,mode='nearest'))

                B0_Transposed = B0Map_s.flatten()[:, np.newaxis].T

                mul_terms = (t_matrix@B0_Transposed)

                D = np.exp(1j * 2 * np.pi * mul_terms)

                E_slice = E * D

                Recon[:, :, ls], time_ls = E2Recon(E_slice, svd_threshold, plt_svd)

                svd_time.append(time_ls)

        else: 

            Recon = np.zeros((E.shape[1], E.shape[0]), dtype=complex)

            svd_time = []

            B0Map_s = zoom(B0Map, (x,y), order=1,prefilter=False,mode='nearest')

            B0_Transposed = B0Map_s.flatten()[:, np.newaxis].T
        
            mul_terms = (t_matrix@B0_Transposed)
 
            D = np.exp(1j * 2 * np.pi * mul_terms)

            E_slice = E * D

            Recon, time_ls = E2Recon(E_slice, svd_threshold, plt_svd)

            svd_time.append(time_ls)   
            
    else:

        Recon, svd_time = E2Recon(E, svd_threshold, plt_svd)
    
    return Recon, svd_time


def E2Recon(E, svd_threshold, plt_svd):

    print('Running SVD')

    start_time = time.time()

    U, Sl, Vh = svd(E, full_matrices=False)

    svd_time = time.time() - start_time

    S = np.diag(Sl)  

    V = Vh.T

    svdE = np.diag(S)

# Plot svdE if plt_svd is True
    if plt_svd:
        plt.figure()
        plt.plot(svdE)
        plt.title('SVD Values')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.show()

# Compute cumulative energy

    cumulative_energy = np.cumsum(svdE,axis=0) / np.sum(svdE)

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

# Timing the reconstruction
    start_time = time.time()
    Recon = V @ invS @ U.T
    end_time = time.time()
    
    return Recon, svd_time

main()
