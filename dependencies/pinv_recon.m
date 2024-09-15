function [image,imageabs,Recon]=pinv_recon(data,wf,Recon,mtx_reco,B0Map,svd_threshold,plt_svd)
%% Reconstruction with pseudoinverse of encoding matrix
% Provide waveform to calculate recon matrix or feed in recon matrix.
%
% Inputs         d  pfile name/ raw (pfile) data/ data in               
%                   (ncoils,npts,ntimesteps,nslices)
%              wfn  waveform (name or struct)
%            Recon  recon matrix                                       ([])
%                   -2D matrix (image-space x k-space)
%         mtx_reco  matrix size after reconstruction (assumed square)  ([])
%            B0Map  B0 map for off-resonance frequency correction      ([])
%                   (x,y,z)
%    svd_threshold  threshold for pinv                               (0.95)
%          plt_svd  plotting svd and threshold value                    (0)
% Outputs
%            image  complex image
%         imageabs  absolute image with coil combination
%                F  reconstruction matrix
% 08/2024   Kylie Yeung

if (nargin<1), help(mfilename); return; end

tStart=tic;
if ~exist('wf','var');wf=[];end
if ~exist('F','var');Recon=[];end
if ~exist('mtx_reco','var');mtx_reco=[];end
if ~exist('B0Map','var');B0Map=[];end
if ~exist('svd_threshold','var');svd_threshold=[];end
if isempty(svd_threshold);svd_threshold=0.95;end
fprintf('Pinv threshold defined as %d%% of cumulative energy \n', svd_threshold*100)
if ~exist('plt_svd','var');plt_svd=0;end

[ncoils,npts,ntimesteps,nslices]=size(data);

%% get reconstruction matrix
if isempty(Recon)
    [Recon,pinv_time] = wf2Recon(wf,mtx_reco,B0Map, svd_threshold, plt_svd);
end

%% reconstruct

mtx_reco=sqrt(size(Recon,1));
image=zeros(mtx_reco,mtx_reco,nslices,ntimesteps,ncoils);

if ndims(Recon)==3
    % slicewise reconstruction, required for B0 and sense recon
    for ls=1:nslices
        dd_s=reshape(permute(data(:,:,:,ls),[2 1 3]),[size(Recon,2),ntimesteps*ncoils]);
        data_inverted=Recon(:,:,ls)*dd_s;
        image(:,:,ls,:,:)=reshape(data_inverted,[mtx_reco,mtx_reco,1,ntimesteps,ncoils]);
    end
else
    data=reshape(permute(data,[2 4 3 1]),[size(Recon,2),nslices*ntimesteps*ncoils]);
    data_inverted=Recon*data;
    image=reshape(data_inverted,[mtx_reco,mtx_reco,nslices,ntimesteps,ncoils]);
end

%% coil combine
if ncoils>1
    imageabs = sqrt(mean(image.*conj(image),5));
else
    imageabs = abs(image);
end

%% timing
try 
    fprintf('recon_pinv: runtime = %.2f [s] (%.2f [s] for pinv) \n ',toc(tStart), pinv_time(1))
catch
    fprintf('recon_pinv: runtime = %.2f [s] \n ',toc(tStart))
end

%% subfunctions
function [Recon,pinv_time,mtx_acq,mtx_reco] = wf2Recon(wf,mtx_reco,B0Map, svd_threshold, plt_svd)

k=wf.k;t=wf.t;

if ~isreal(k)
    if size(k,1)~=1, error('size(k,1)~=1'); end
    k_temp=k;
    k(1,:,1) = real(k_temp);
    k(1,:,2) = imag(k_temp);
    clear k_temp
end

% K-space trajectory definition
mtx_acq=wf.npix;
if isempty(mtx_reco);mtx_reco=mtx_acq;end

[XM,YM]=ndgrid(single(-(mtx_reco-1)/2:(mtx_reco-1)/2));   % 2D coordinates

% scale from +-0.5 to +-pi
Kx=single(k(:,:,1))./0.5*pi./mtx_reco.*mtx_acq;
Ky=single(k(:,:,2))./0.5*pi./mtx_reco.*mtx_acq;

% Encoding matrix exp(i*k*r)
E=exp(1i*(Kx(:)*(XM(:)).'+Ky(:)*(YM(:)).'));

pinv_time=[];
if ~isempty(B0Map)
    fprintf('Getting encoding matrix with B0 encoding ... \n')
    nslices=size(B0Map,3);
    Recon=zeros(size(E,2),size(E,1),nslices);
    svd_time=[];
    for ls=1:nslices
        B0Map_s=-imresize(B0Map(:,:,ls),[mtx_reco mtx_reco]);
        D=exp(1i*2*pi*t(:)*(B0Map_s(:)).');
        E_slice=E.*D;
        [Recon_s,time_ls]=E2Recon(E_slice,svd_threshold,plt_svd);
        svd_time=[svd_time time_ls];
        Recon(:,:,ls)=Recon_s;
    end
else
    [Recon,svd_time]=E2Recon(E,svd_threshold,plt_svd);
    fprintf('Calculating svd took %f seconds \n',svd_time)
end

end

function [Recon,svd_time]=E2Recon(E,svd_threshold,plt_svd)
svd_start=tic; [U,S,V]=svd(E,'econ'); svd_time=toc(svd_start);
svdE=diag(S);
if plt_svd;figure;plot(svdE);end

cumulative_energy = cumsum(svdE) / sum(svdE);
g = find(cumulative_energy >= svd_threshold, 1);
if isempty(g);g=size(svdE,1);end
% imax=svdE(g);

invS=1./svdE; invS(g+1:end)=0;invS=diag(invS);
tic, Recon=V*invS*U'; toc, 
end

end
