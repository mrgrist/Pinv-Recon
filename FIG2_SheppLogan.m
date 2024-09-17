% MRM paper submisson Fig.2
% PinvRecon simulations using the numerical Shepp-Logan phantom
% testing different k-space sampling patterns

%% define parameters
mtx=128; % acquisition matrix size
ntx=mtx; % reconstruction matrix size (reduce to shorten compute time)
svdCUT=32; % SVD threshold


% image-space sampling (over-discretized)
TMP=linspace(-1,1,ntx+1); [Rx,Ry]=ndgrid(TMP(1:end-1)); 
circMASK=sqrt(Rx.^2+Ry.^2)<1;
IMG=phantom('Modified Shepp-Logan',ntx);   
IMG(IMG>0.75)=0.75; IMG=single(IMG/max(IMG(:)));

%==============================================
disp('>> 1. Equidistant Cartesian Sampling');
%==============================================
% k-space sampling
TMP=[-mtx/2:mtx/2]; [Kx,Ky]=ndgrid(TMP(2:end)); Kx1=Kx; Ky1=Ky;
% Encoding matrix exp(i*k*r)
% 1i*kn*rm=1i*n*2*pi/FOV*m*FOV/2*(-1..+1)=1i*n*pi*m*(-1..+1)
ENCODE1=single(exp(1i*pi*(Kx(:)*Rx(:).'+Ky(:)*Ry(:).')));
tic, [U,S,V]=svd(ENCODE1,'econ'); toc, U1=gather(U); S1=gather(S); V1=gather(V); 
U=U1;S=S1; V=V1;
imax=find(diag(S)>svdCUT,1,'last');
invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON1=V*invS*U'; toc, 
IMG1=reshape(RECON1*(ENCODE1*IMG(:)),[ntx,ntx]);
NOISE1=sqrt(reshape(diag(RECON1*RECON1'),[ntx,ntx])/size(RECON1,2));
SRF1=reshape(diag(RECON1*ENCODE1),[ntx,ntx]);

%=============================
disp('>> 2. VDPD Sampling');
%=============================
load(sprintf("data/vdpd_mask_mtx%d.mat",mtx))
% Kspace sampling
TMP=[-mtx/2+0.5:+mtx/2-0.5]; [XM,YM]=ndgrid(TMP);
Kx=XM(vdpd_mask); Ky=YM(vdpd_mask); Kx2=Kx; Ky2=Ky;
ENCODE2=single(exp(1i*pi*(Kx(:)*Rx(:).'+Ky(:)*Ry(:).')));
tic, [U,S,V]=svd(ENCODE2,'econ'); toc, U2=gather(U); S2=gather(S); V2=gather(V);
U=U2;S=S2; V=V2;
imax=find(diag(S)>svdCUT,1,'last');
invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON2=V*invS*U'; toc, 
IMG2=reshape(RECON2*(ENCODE2*IMG(:)),[ntx,ntx]);
NOISE2=sqrt(reshape(diag(RECON2*RECON2'),[ntx,ntx])/size(RECON2,2));
SRF2=reshape(diag(RECON2*ENCODE2),[ntx,ntx]);

%============================
disp('>> 3. EPI Sampling');
%============================
% Kspace sampling
Gread=[linspace(0,1,4), ones(1,128),linspace(1,0,4)];
Gphas=[linspace(1,0,4),zeros(1,128),linspace(0,1,4)]/4;
Gx=Gread(5:end); Gy=Gphas(5:end);
for i1=1:mtx-1
    Gx=[Gx,Gread*(-1)^i1];
    Gy=[Gy,Gphas];
end
Kx=cumsum(Gx)-63; Ky=cumsum(Gy)-63; Kx3=Kx; Ky3=Ky;
ENCODE3=single(exp(1i*pi*(Kx(:)*Rx(:).'+Ky(:)*Ry(:).')));
tic, [U,S,V]=svd(ENCODE3,'econ'); toc, U3=gather(U); S3=gather(S); V3=gather(V);
U=U3;S=S3; V=V3;
imax=find(diag(S)>svdCUT,1,'last');
invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON3=V*invS*U'; toc, 
IMG3=reshape(RECON3*(ENCODE3*IMG(:)),[ntx,ntx]);
NOISE3=sqrt(reshape(diag(RECON3*RECON3'),[ntx,ntx])/size(RECON3,2));
SRF3=reshape(diag(RECON3*ENCODE3),[ntx,ntx]);

%===============================================
disp('>> 4. 2D golden-angle Radial Sampling');
%===============================================
gaPhi=pi*(3-sqrt(5));
nSpk=mtx;
TMP=-mtx/2:1:mtx/2;
iPhi=0; Kx=[]; Ky=[];
% figure(99), subplot('position',[0.44,0.73,0.13,0.22]); 
for iSpk=1:nSpk
    Kx=[Kx,TMP*cos(iPhi)];
    Ky=[Ky,TMP*sin(iPhi)];
    iPhi=iPhi+gaPhi;
    % plot(TMP*cos(iPhi),TMP*sin(iPhi),'.-k','MarkerSize',2); hold on, 
end
Kx4=Kx; Ky4=Ky;
% set(gca,'XTick',-64:32:64,'YTick',-64:32:64,'FontSize',14); axis image, grid on, axis(72*[-1 1 -1 1]), 
ENCODE4=single(exp(1i*pi*(Kx(:)*Rx(:).'+Ky(:)*Ry(:).')));
tic, [U,S,V]=svd(ENCODE4,'econ'); toc, U4=gather(U); S4=gather(S); V4=gather(V);
U=U4;S=S4; V=V4;
imax=find(diag(S)>svdCUT,1,'last');
invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON4=V*invS*U'; toc, 
IMG4=reshape(RECON4*(ENCODE4*IMG(:)),[ntx,ntx]);
NOISE4=sqrt(reshape(diag(RECON4*RECON4'),[ntx,ntx])/size(RECON4,2));
SRF4=reshape(diag(RECON4*ENCODE4),[ntx,ntx]);

%=============================================
disp('>> 5. 2D Spiral design (Grad & k-traj)');
% dk=2*pi/FOV=gamma*Gread*dt
%=============================================
FOV=0.240;          % [m]
res=FOV/mtx;        % [m]
Gmax=0.200;         % [T/m]
Smax=500;           % [T/(m*s)]
BW=125e3;           % [Hz]
dt=1/(2*BW);        % [s]
load(sprintf("data/vd_spiral_mtx%d.mat",mtx))
Kx=mtx/2*real(k/abs(k(end)));
Ky=mtx/2*imag(k/abs(k(end)));
Kx5=Kx; Ky5=Ky;
ENCODE5=single(exp(1i*pi*(Kx(:)*Rx(:).'+Ky(:)*Ry(:).')));
tic, [U,S,V]=svd(ENCODE5,'econ'); toc, U5=gather(U); S5=gather(S); V5=gather(V);
U=U5;S=S5; V=V5;
imax=find(diag(S)>svdCUT,1,'last');
invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON5=V*invS*U'; toc, 
IMG5=reshape(RECON5*(ENCODE5*IMG(:)),[ntx,ntx]);
NOISE5=sqrt(reshape(diag(RECON5*RECON5'),[ntx,ntx])/size(RECON5,2));
SRF5=reshape(diag(RECON5*ENCODE5),[ntx,ntx]);

figure(99); set(gcf,'Position',[104,4,1114,1121]); FntSz=12; 
subplot('position',[0.00,0.40,1.00,0.2]);   % IMAGE
imagesc(abs([IMG1,IMG3,IMG2,IMG4,IMG5]),[-0.02,1.02]); axis image off, colormap(gca,'gray');
h=colorbar(gca,'south',Ticks=(0:0.25:1.0),Position=[0.002 0.40 0.996 0.005],Color=0.9*[1,1,1],FontSize=FntSz,AxisLocation='in');
subplot('position',[0.00,0.20,1.00,0.2]);   % SRF
imagesc(abs([SRF1,SRF3,SRF2,SRF4,SRF5]),[-0.05,1.02]); axis image off, colormap(gca,'gray'); 
h=colorbar(gca,'south',Ticks=(0:0.25:1.0),Position=[0.002 0.20 0.996 0.005],Color=0.1*[1,1,1],FontSize=FntSz,AxisLocation='in');
subplot('position',[0.00,0.00,1.00,0.2]);   % NOISE
imagesc(abs([NOISE1,NOISE3,NOISE2,NOISE4,NOISE5]),[-0.02,1.02]*1e-4); axis image off, colormap(gca,'gray'); 
h=colorbar(gca,'south',Ticks=1e-4*(0:0.25:1.0),Position=[0.002 0.00 0.996 0.005],Color=0.1*[1,1,1],FontSize=FntSz,AxisLocation='in');
ilist=[1,3,2,4,5]; for i1=1:5               % k-Trajectory & SVD
    eval(['ikx=Kx',num2str(ilist(i1)),'(:); iky=Ky',num2str(ilist(i1)),'(:);']);
    eval(['iEigVal=diag(S',num2str(ilist(i1)),');']);
    subplot('Position',[0.03+(i1-1)*0.20,0.64,0.16,0.14]);
    plot(iEigVal,'-k','LineWidth',4); axis([1,length(iEigVal),0,256]), 
    set(gca,'FontSize',FntSz,'xtick',8192*(0:2),'ytick',[0,svdCUT,128,256,512]); grid on,
    subplot('Position',[0.03+(i1-1)*0.20,0.80,0.16,0.20]);
    plot(ikx(:),iky(:),'.k','MarkerSize',1); 
    set(gca,'XTick',-64:32:64,'YTick',-64:32:64,'FontSize',FntSz), axis image, grid on, axis(72*[-1 1 -1 1]), 
end
