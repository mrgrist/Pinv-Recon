close all;clear
addpath('dependencies\')

%
CondNumb=30; % extent of SVD regularzation
ntx=32;% image matrix size

% coil sensitivity maps
nRCV=8;load(sprintf('data/sens_maps%d.mat',ntx))

% generate Shepp-Logan Phantom
IMG=phantom('Modified Shepp-Logan',ntx);   
IMG(IMG>0.75)=0.75; IMG=single(IMG/max(IMG(:)));

% trajectory
load("data\vd_spiral.mat");Kx=Kx1; Ky=Ky1;
[XM,YM]=ndgrid(-ntx/2:ntx/2-1,-ntx/2:ntx/2-1);

disp('>> RxSens...');
% encode and recon matrices
ENCODE=single(exp(1i*pi*(Kx(:)*XM(:).'+Ky(:)*YM(:).')));
RECON=getRECON(ENCODE, CondNumb); 
% + SENSE
ENCODE_sense_tmp=single(exp(1i*pi*(Kx(:)*XM(:).'+Ky(:)*YM(:).')));
ENCODE_sense=[]; for ircv=1:nRCV, TMPi=sensCtr(:,:,ircv); ENCODE_sense=[ENCODE_sense; ENCODE_sense_tmp.*TMPi(:).']; end
RECON_sense=getRECON(ENCODE_sense, CondNumb); 


% Reconstruct
data=ENCODE*IMG(:);
IMG=reshape(RECON*data,[ntx,ntx]);
data_sense=ENCODE_sense*IMG(:);
IMG_ircv=reshape(RECON*reshape(data_sense,[],ircv),[ntx,ntx,ircv]);
IMG_combined=reshape(RECON_sense*data_sense,[ntx,ntx]);

figure;
subplot(221);mat2montage(IMG); title('no RxSens') % image encoded and reconstructed without coil sensitivity
subplot(222);mat2montage(sensOff); title('Sens map') % coil sensitivity map
subplot(223);mat2montage(IMG_ircv); title('channels reconstructed separately') % image encoded and reconstructed without coil sensitivity
subplot(224);mat2montage(IMG_combined); title('coil combined')% image encoded and reconstructed without coil sensitivity


function RECON=getRECON(E, CondNumb)
% % GPU acceleration
% E=gpuArray(E);
 
tic, [U,S,V]=svd((E),'econ'); toc,
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON=V*invS*U'; toc,



% % faster alternative
% if size(E,1)>size(E,2) %overdetermined
%     lambda=1./CondNumb*norm(E,1); % needs to be adjusted
%     EHE=E'*E+lambda*eye(size(E,2));
% 
%     L = chol(EHE, 'lower');
%     L_inv = inv(L);
%     iEHE = L_inv' * L_inv;
%     RECON=iEHE*E';
% else
%     lambda=1./CondNumb*norm(E,1);
% 
%     EEH=E*E'+lambda*eye(size(E,1));
%     L = chol(EEH, 'lower');
%     L_inv = inv(L);
%     iEEH = L_inv' * L_inv;
%     RECON=E'*iEEH;
% end   
end