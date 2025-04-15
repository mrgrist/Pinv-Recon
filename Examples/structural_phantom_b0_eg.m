%% Pinv recon for spiral data with B0 corr
% An example of Pinv-Recon on a Structural Phantom dataset with B0
% correction
close all; clear
addpath("../dependencies/")
load("../data/fieldmap.mat")
load("../data/structural_phantom_mtx96.mat") % contains data and waveform

mtx_acq=wf.npix;
nslices=1;
ntimesteps=1;
ncoils=size(data,1);
    
mtx_reco=mtx_acq;
% mtx_reco=2*mtx_acq; % to overdiscretize recon

[image,imageabs,Recon]=pinv_recon(data,wf,[],mtx_reco);
[image_b0,imageabs_b0,Recon_b0]=pinv_recon(data,wf,[],mtx_reco,fieldmap);

figure;
subplot(131);imagesc(flipud(fieldmap));colorbar; axis square; axis off;title("B_0 map")
subplot(132);imagesc(flipud(imageabs));colormap("gray"); axis square; axis off;title("Pinv")
subplot(133);imagesc(flipud(imageabs_b0));colormap("gray"); axis square; axis off;title("Pinv + OffRes")
    
