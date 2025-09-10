<<<<<<< Updated upstream:FigsReproduce/FIG3_SheppLogan.m
% Scientific reports paper submisson Fig. 3
% Pinv-Recon simulations using the numerical Shepp-Logan phantom
% combining different encoding and distortion mechanisms

CondNumb=30;
ntx=128;%64,96,128

%==============================================
disp('>> Loading and generating relevant maps...');
%==============================================
load("data/vd_spiral.mat")
% Kx0, Ky0 and time0 correspond to Fully-sampled R=1
% Kx1, Ky1 and time1 correspond toUnder-sampled R=2x2

% spatial discretization (centered and shifted)
xiLarge=linspace(-2,2,2*ntx+1); xiLarge=xiLarge(1:end-1); dri=mean(diff(xiLarge)); 
[XMlarge,YMlarge]=ndgrid(xiLarge,xiLarge);
ictr=ntx+(-ntx/2:ntx/2-1);    xiCtr=xiLarge(ictr); [XMctr,YMctr]=ndgrid(xiCtr,xiCtr);
ioff=ntx+(-ntx/2:ntx/2-1)+floor(ntx/10); xiOff=xiLarge(ioff); [XMoff,YMoff]=ndgrid(xiCtr,xiOff);

% Generating phantom
imgPhantom=phantom('Modified Shepp-Logan',ntx); 
imgPhantom(imgPhantom>0.75)=0.75; imgPhantom=single(imgPhantom/max(imgPhantom(:)));
imgLarge=zeros(2*ntx,2*ntx); 
imgLarge(ictr,ictr)=imgPhantom;
imgCtr=interpn(XMlarge,YMlarge,imgLarge,XMctr,YMctr);
imgOff=interpn(XMlarge,YMlarge,imgLarge,XMoff,YMoff);

% Loading Biot-Savart simulated coil sensitivity maps
nRCV=8;
load(sprintf('data/sens_maps%d.mat',ntx))

% Generating B0 maps
f0Large=125*YMlarge.^2-30; 
f0Ctr=interpn(XMlarge,YMlarge,f0Large,XMctr,YMctr);
f0Off=interpn(XMlarge,YMlarge,f0Large,XMoff,YMoff);

% Generating gradient nonlinearity
GradNonLinLarge=-(0.15*(XMlarge.^3));
GradNonLinCtr=interpn(XMlarge,YMlarge,GradNonLinLarge,XMctr,YMctr);
GradNonLinOff=interpn(XMlarge,YMlarge,GradNonLinLarge,XMoff,YMoff);

%==============================================
disp('>> 1. Reconstructing with offCtr...');
%==============================================
% + shift
Kx=Kx0; Ky=Ky0; time=time0;
ENCODE00=single(exp(1i*pi*(Kx(:)*XMoff(:).'+Ky(:)*YMoff(:).')));
ENCODE01=single(exp(1i*pi*(Kx(:)*XMctr(:).'+Ky(:)*YMctr(:).')));
tic, [U00,S00,V00]=svd((ENCODE00),'econ'); toc, U=U00; S=S00; V=V00; diagS00=diag(S00);
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON00=V*invS*U'; toc, 
IMG00=reshape(RECON00*(ENCODE00*imgOff(:)),[ntx,ntx]);
tic, [U01,S01,V01]=svd((ENCODE01),'econ'); toc, U=U01; S=S01; V=V01; diagS01=diag(S01); 
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON01=V*invS*U'; toc, 
IMG01=reshape(RECON01*(ENCODE00*imgOff(:)),[ntx,ntx]);

disp('>> 2. + GradNonLin...');
% + non-linear x-gradient
ENCODE10=single(exp(1i*pi*(Kx(:)*(XMoff(:)+1*GradNonLinOff(:)).'+Ky(:)*(YMoff(:)+0*GradNonLinOff(:)).')));
ENCODE11=single(exp(1i*pi*(Kx(:)*(XMctr(:)+1*GradNonLinCtr(:)).'+Ky(:)*(YMctr(:)+0*GradNonLinCtr(:)).')));
tic, [U11,S11,V11]=svd((ENCODE11),'econ'); toc, U=U11; S=S11; V=V11; diagS11=diag(S11); 
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON11=V*invS*U'; toc, 
IMG10=reshape(RECON00*(ENCODE10*imgOff(:)),[ntx,ntx]);
IMG11=reshape(RECON11*(ENCODE10*imgOff(:)),[ntx,ntx]);

disp('>> 3. + OffRes...');
% + B0
ENCODE20=single(ENCODE10.*exp(1i*2*pi*time(:)*f0Off(:).'));
ENCODE21=single(ENCODE11.*exp(1i*2*pi*time(:)*f0Ctr(:).'));
tic, [U21,S21,V21]=svd((ENCODE21),'econ'); toc, U=U21; S=S21; V=V21; diagS21=diag(S21); 
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON21=V*invS*U'; toc, 
IMG20=reshape(RECON00*(ENCODE20*imgOff(:)),[ntx,ntx]);
IMG21=reshape(RECON21*(ENCODE20*imgOff(:)),[ntx,ntx]);

disp('>> 4. + RxSens...');
Kx=Kx1; Ky=Ky1; time=2*time1;
% + SENSE
ENCODE30i=single(exp(1i*pi*(Kx(:)*XMoff(:).'+Ky(:)*YMoff(:).')));
tic, [U30i,S30i,V30i]=svd((ENCODE30i),'econ'); toc, U=U30i; S=S30i; V=V30i; diagS30i=diag(S30i); 
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON30i=V*invS*U'; toc, 
ENCODE30i=single(exp(1i*pi*(Kx(:)*(XMoff(:)+1*GradNonLinOff(:)).'+Ky(:)*(YMoff(:)+0*GradNonLinOff(:)).')).*exp(1i*2*pi*time(:)*f0Off(:).'));
ENCODE31i=single(exp(1i*pi*(Kx(:)*(XMctr(:)+1*GradNonLinCtr(:)).'+Ky(:)*(YMctr(:)+0*GradNonLinCtr(:)).')).*exp(1i*2*pi*time(:)*f0Ctr(:).'));
ENCODE30=[]; for ircv=1:nRCV, TMPi=sensOff(:,:,ircv); ENCODE30=[ENCODE30; ENCODE30i.*TMPi(:).']; end
ENCODE31=[]; for ircv=1:nRCV, TMPi=sensCtr(:,:,ircv); ENCODE31=[ENCODE31; ENCODE31i.*TMPi(:).']; end
tic, [U31,S31,V31]=svd((ENCODE31),'econ'); toc, U=U31; S=S31; V=V31; diagS31=diag(S31); 
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON31=V*invS*U'; toc, 
IMG30=reshape(RECON30i*(ENCODE30i*imgOff(:)),[ntx,ntx]);
IMG31=reshape(RECON31*(ENCODE30*imgOff(:)),[ntx,ntx]);

disp('>> 5. + ChemShift...');
% + ChemShift
freqCS1=0; freqCS2=500; TE=1e-3;
imgCS1=imgOff; imgCS1(imgOff<0.9)=0; 
imgCS2=imgOff; imgCS2(imgOff>0.9)=0; 
% off
ENCODE40i=single(exp(1i*pi*(Kx(:)*(XMoff(:)+1*GradNonLinOff(:)).'+Ky(:)*YMoff(:).')));
ENCODE40j=single([
    ENCODE40i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Off(:).'+freqCS1)),...
    ENCODE40i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Off(:).'+freqCS2))]);
ENCODE40=single([
    ENCODE40i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Off(:).'+freqCS1)),...
    ENCODE40i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Off(:).'+freqCS2));...
    ENCODE40i.*exp(1i*2*pi*(time(:)+1*TE)*(f0Off(:).'+freqCS1)),...
    ENCODE40i.*exp(1i*2*pi*(time(:)+1*TE)*(f0Off(:).'+freqCS2))]);
ENCODE40rcv=[]; for ircv=1:nRCV, TMPi=sensOff(:,:,ircv); % sensitivity encoding
    ENCODE40rcv=[ENCODE40rcv; ENCODE40.*[TMPi(:);TMPi(:)].']; end
% ctr
ENCODE41i=single(exp(1i*pi*(Kx(:)*(XMctr(:)+1*GradNonLinCtr(:)).'+Ky(:)*YMctr(:).')));
ENCODE41=single([
    ENCODE41i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Ctr(:).'+freqCS1)),...
    ENCODE41i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Ctr(:).'+freqCS2));...
    ENCODE41i.*exp(1i*2*pi*(time(:)+1*TE)*(f0Ctr(:).'+freqCS1)),...
    ENCODE41i.*exp(1i*2*pi*(time(:)+1*TE)*(f0Ctr(:).'+freqCS2))]);
ENCODE42=[]; for ircv=1:nRCV, TMPi=sensCtr(:,:,ircv); % sensitivity encoding
    ENCODE42=[ENCODE42; ENCODE41.*[TMPi(:);TMPi(:)].']; end

tic, [U42,S42,V42]=svd((ENCODE42),'econ'); toc, U=U42; S=S42; V=V42; diagS42=diag(S42);
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON42=V*invS*U'; toc, 
IMG40=reshape(RECON30i*(ENCODE40j*[imgCS1(:);imgCS2(:)]),[ntx,ntx]);
IMG42=reshape(RECON42* (ENCODE40rcv* [imgCS1(:);imgCS2(:)]),[ntx,2*ntx]);


%==============================================
disp('>> Plotting all results...');
%==============================================
figure(99); set(gcf,'Position',[104,4,1293,1076]); FntSz=12; 
subplot('position',[0.00,0.00,1.0,0.4]);   % IMAGE
imagesc(abs(interp2([IMG00,IMG10,IMG20,IMG30,zeros(ntx,ntx/2),IMG40,zeros(ntx,ntx/2);...
    IMG01,IMG11,IMG21,IMG31,IMG42],2)),[-0.02,1.02]); axis image off, colormap(gca,'gray');
W=1/6;
diagS=gather(diag(S01)); subplot('position',[0*W+0.03,0.43,0.13,0.16]); plot(diagS,'-k',LineWidth=2);
set(gca,'FontSize',FntSz,'yscale','log','xtick',[1,12000,24000],'xticklabel',{'1','12000','24000'},'ytick',10.^(-1:2:3)); grid on, axis([1,length(diagS),1e-1,1e3]), 
diagS=gather(diag(S11)); subplot('position',[1*W+0.03,0.43,0.13,0.16]); plot(diagS,'-k',LineWidth=2);
set(gca,'FontSize',FntSz,'yscale','log','xtick',[1,12000,24000],'xticklabel',{'1','12000','24000'},'ytick',10.^(-1:2:3)); grid on, axis([1,length(diagS),1e-1,1e3]), 
diagS=gather(diag(S21)); subplot('position',[2*W+0.03,0.43,0.13,0.16]); plot(diagS,'-k',LineWidth=2);
set(gca,'FontSize',FntSz,'yscale','log','xtick',[1,12000,24000],'xticklabel',{'1','12000','24000'},'ytick',10.^(-1:2:3)); grid on, axis([1,length(diagS),1e-1,1e3]), 
diagS=gather(diag(S31)); subplot('position',[3*W+0.03,0.43,0.13,0.16]); plot(diagS,'-k',LineWidth=2);
set(gca,'FontSize',FntSz,'yscale','log','xtick',[1,12000,24000],'xticklabel',{'1','12000','24000'},'ytick',10.^(-1:2:3)); grid on, axis([1,length(diagS),1e-1,1e3]), 
diagS=gather(diag(S42)); subplot('position',[4*W+0.03,0.43,0.13+W,0.16]); plot(diagS,'-k',LineWidth=2);
set(gca,'FontSize',FntSz,'yscale','log','xtick',[1,12000,24000],'xticklabel',{'1','12000','24000'},'ytick',10.^(-1:2:3)); grid on, axis([1,length(diagS),1e-1,1e3]), 

subplot('Position',[0*W+0.03,0.63,0.13,0.16]);
plot(Kx1(:),Ky1(:),'-k',LineWidth=0.5); 
set(gca,'XTick',-ntx/2:ntx/4:ntx/2,'YTick',-ntx/2:ntx/4:ntx/2,'FontSize',FntSz), axis image, grid on, axis(72*[-1 1 -1 1]), 
title('Gradient + OffCtr',FontSize=18)
subplot('Position',[1*W+0.03,0.63,0.13,0.16]);
imagesc(GradNonLinCtr); colormap(gca,gray); axis image off, colorbar("east",Color=0.5*[1,1,1],Ticks=[],FontSize=FntSz)
title('+ GradNonLin',FontSize=18)
subplot('Position',[2*W+0.03,0.63,0.13,0.16]);
imagesc(f0Ctr); colormap(gca,gray); axis image off, colorbar("south",Color=0.5*[1,1,1],Ticks=[],FontSize=FntSz)
title('+ OffRes',FontSize=18)
subplot('Position',[3*W+0.03,0.63,0.13,0.16]);
TMP=sensCtr(:,:,[6 7 8 5 5 1 4 3 2]); TMP(:,:,5)=0;
% TMP=circMASK.*sens(:,:,[1:4,4:8]); TMP(:,:,5)=0;
imagesc([reshape(abs(TMP(:,:,1:3)),[ntx,ntx*3]);...
         reshape(abs(TMP(:,:,4:6)),[ntx,ntx*3]);...
         reshape(abs(TMP(:,:,7:9)),[ntx,ntx*3])],[0,0.3]); axis image off, colormap(gca,gray)
title('+RxSens',FontSize=18)
subplot('Position',[4*W+0.03,0.63,0.29,0.16]);
ifreq=linspace(-700,200,901); 
plot(ifreq,6^2./(6^2+(ifreq+500).^2)+6^2./(6^2+(ifreq+0).^2),'-k',LineWidth=2); axis([-700 200 -0.2 1.2])
set(gca,'XTick',[-500,0],'XTickLabel',{'Fat','Water'},'YTick',[],'FontSize',FntSz), grid on, axis on, 
title('+ ChemShift',FontSize=18)
=======
% MRM paper submisson Fig.3
% PinvRecon simulations using the numerical Shepp-Logan phantom
% combining different encoding and distortion mechanisms

CondNumb=30;
ntx=32;%64,96,128

%==============================================
disp('>> Loading and generating relevant maps...');
%==============================================
load("data/vd_spiral.mat")
% Kx0, Ky0 and time0 correspond to Fully-sampled R=1
% Kx1, Ky1 and time1 correspond toUnder-sampled R=2x2

% spatial discretization (centered and shifted)
xiLarge=linspace(-2,2,2*ntx+1); xiLarge=xiLarge(1:end-1); dri=mean(diff(xiLarge)); 
[XMlarge,YMlarge]=ndgrid(xiLarge,xiLarge);
ictr=ntx+(-ntx/2:ntx/2-1);    xiCtr=xiLarge(ictr); [XMctr,YMctr]=ndgrid(xiCtr,xiCtr);
ioff=ntx+(-ntx/2:ntx/2-1)+floor(ntx/10); xiOff=xiLarge(ioff); [XMoff,YMoff]=ndgrid(xiCtr,xiOff);

% Generating phantom
imgPhantom=phantom('Modified Shepp-Logan',ntx); 
imgPhantom(imgPhantom>0.75)=0.75; imgPhantom=single(imgPhantom/max(imgPhantom(:)));
imgLarge=zeros(2*ntx,2*ntx); 
imgLarge(ictr,ictr)=imgPhantom;
imgCtr=interpn(XMlarge,YMlarge,imgLarge,XMctr,YMctr);
imgOff=interpn(XMlarge,YMlarge,imgLarge,XMoff,YMoff);

% Loading Biot-Savart simulated coil sensitivity maps
nRCV=8;
load(sprintf('data/sens_maps%d.mat',ntx))

% Generating B0 maps
f0Large=125*YMlarge.^2-30; 
f0Ctr=interpn(XMlarge,YMlarge,f0Large,XMctr,YMctr);
f0Off=interpn(XMlarge,YMlarge,f0Large,XMoff,YMoff);

% Generating gradient nonlinearity
GradNonLinLarge=-(0.15*(XMlarge.^3));
GradNonLinCtr=interpn(XMlarge,YMlarge,GradNonLinLarge,XMctr,YMctr);
GradNonLinOff=interpn(XMlarge,YMlarge,GradNonLinLarge,XMoff,YMoff);

[k,dcf,t,grad,out,ga_burst] = design_epi(240,ntx,1,45,150,...
    1,0,0,0,[],'1h',[1,1],250d3,4d-6);

Kx0=k(1,:,1)*128; Ky0=k(1,:,2)*128; time0=t;
Kx1=k(1,:,1)*128; Ky1=k(1,:,2)*128; time0=t;
%==============================================
disp('>> 1. Reconstructing with offCtr...');
%==============================================
% + shift
Kx=Kx0; Ky=Ky0; time=time0;
ENCODE00=single(exp(1i*pi*(Kx(:)*XMoff(:).'+Ky(:)*YMoff(:).')));
ENCODE01=single(exp(1i*pi*(Kx(:)*XMctr(:).'+Ky(:)*YMctr(:).')));
tic, [U00,S00,V00]=svd((ENCODE00),'econ'); toc, U=U00; S=S00; V=V00; diagS00=diag(S00);
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON00=V*invS*U'; toc, 
IMG00=reshape(RECON00*(ENCODE00*imgOff(:)),[ntx,ntx]);
tic, [U01,S01,V01]=svd((ENCODE01),'econ'); toc, U=U01; S=S01; V=V01; diagS01=diag(S01); 
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON01=V*invS*U'; toc, 
IMG01=reshape(RECON01*(ENCODE00*imgOff(:)),[ntx,ntx]);

disp('>> 2. + GradNonLin...');
% + non-linear x-gradient
ENCODE10=single(exp(1i*pi*(Kx(:)*(XMoff(:)+1*GradNonLinOff(:)).'+Ky(:)*(YMoff(:)+0*GradNonLinOff(:)).')));
ENCODE11=single(exp(1i*pi*(Kx(:)*(XMctr(:)+1*GradNonLinCtr(:)).'+Ky(:)*(YMctr(:)+0*GradNonLinCtr(:)).')));
tic, [U11,S11,V11]=svd((ENCODE11),'econ'); toc, U=U11; S=S11; V=V11; diagS11=diag(S11); 
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON11=V*invS*U'; toc, 
IMG10=reshape(RECON00*(ENCODE10*imgOff(:)),[ntx,ntx]);
IMG11=reshape(RECON11*(ENCODE10*imgOff(:)),[ntx,ntx]);

disp('>> 3. + OffRes...');
% + B0
ENCODE20=single(ENCODE10.*exp(1i*2*pi*time(:)*f0Off(:).'));
ENCODE21=single(ENCODE11.*exp(1i*2*pi*time(:)*f0Ctr(:).'));
tic, [U21,S21,V21]=svd((ENCODE21),'econ'); toc, U=U21; S=S21; V=V21; diagS21=diag(S21); 
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON21=V*invS*U'; toc, 
IMG20=reshape(RECON00*(ENCODE20*imgOff(:)),[ntx,ntx]);
IMG21=reshape(RECON21*(ENCODE20*imgOff(:)),[ntx,ntx]);

disp('>> 4. + RxSens...');
Kx=Kx0; Ky=Ky0; time=2*t;
% + SENSE
ENCODE30i=single(exp(1i*pi*(Kx(:)*XMoff(:).'+Ky(:)*YMoff(:).')));
tic, [U30i,S30i,V30i]=svd((ENCODE30i),'econ'); toc, U=U30i; S=S30i; V=V30i; diagS30i=diag(S30i); 
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON30i=V*invS*U'; toc, 
ENCODE30i=single(exp(1i*pi*(Kx(:)*(XMoff(:)+1*GradNonLinOff(:)).'+Ky(:)*(YMoff(:)+0*GradNonLinOff(:)).')).*exp(1i*2*pi*time(:)*f0Off(:).'));
ENCODE31i=single(exp(1i*pi*(Kx(:)*(XMctr(:)+1*GradNonLinCtr(:)).'+Ky(:)*(YMctr(:)+0*GradNonLinCtr(:)).')).*exp(1i*2*pi*time(:)*f0Ctr(:).'));
ENCODE30=[]; for ircv=1:nRCV, TMPi=sensOff(:,:,ircv); ENCODE30=[ENCODE30; ENCODE30i.*TMPi(:).']; end
ENCODE31=[]; for ircv=1:nRCV, TMPi=sensCtr(:,:,ircv); ENCODE31=[ENCODE31; ENCODE31i.*TMPi(:).']; end
tic, [U31,S31,V31]=svd((ENCODE31),'econ'); toc, U=U31; S=S31; V=V31; diagS31=diag(S31); 
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON31=V*invS*U'; toc, 
IMG30=reshape(RECON30i*(ENCODE30i*imgOff(:)),[ntx,ntx]);
IMG31=reshape(RECON31*(ENCODE30*imgOff(:)),[ntx,ntx]);

disp('>> 5. + ChemShift...');
% + ChemShift
freqCS1=0; freqCS2=500; TE=1e-3;
imgCS1=imgOff; imgCS1(imgOff<0.9)=0; 
imgCS2=imgOff; imgCS2(imgOff>0.9)=0; 
% off
ENCODE40i=single(exp(1i*pi*(Kx(:)*(XMoff(:)+1*GradNonLinOff(:)).'+Ky(:)*YMoff(:).')));
ENCODE40j=single([
    ENCODE40i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Off(:).'+freqCS1)),...
    ENCODE40i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Off(:).'+freqCS2))]);
ENCODE40=single([
    ENCODE40i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Off(:).'+freqCS1)),...
    ENCODE40i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Off(:).'+freqCS2));...
    ENCODE40i.*exp(1i*2*pi*(time(:)+1*TE)*(f0Off(:).'+freqCS1)),...
    ENCODE40i.*exp(1i*2*pi*(time(:)+1*TE)*(f0Off(:).'+freqCS2))]);
ENCODE40rcv=[]; for ircv=1:nRCV, TMPi=sensOff(:,:,ircv); % sensitivity encoding
    ENCODE40rcv=[ENCODE40rcv; ENCODE40.*[TMPi(:);TMPi(:)].']; end
% ctr
ENCODE41i=single(exp(1i*pi*(Kx(:)*(XMctr(:)+1*GradNonLinCtr(:)).'+Ky(:)*YMctr(:).')));
ENCODE41=single([
    ENCODE41i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Ctr(:).'+freqCS1)),...
    ENCODE41i.*exp(1i*2*pi*(time(:)+0*TE)*(f0Ctr(:).'+freqCS2));...
    ENCODE41i.*exp(1i*2*pi*(time(:)+1*TE)*(f0Ctr(:).'+freqCS1)),...
    ENCODE41i.*exp(1i*2*pi*(time(:)+1*TE)*(f0Ctr(:).'+freqCS2))]);
ENCODE42=[]; for ircv=1:nRCV, TMPi=sensCtr(:,:,ircv); % sensitivity encoding
    ENCODE42=[ENCODE42; ENCODE41.*[TMPi(:);TMPi(:)].']; end

tic, [U42,S42,V42]=svd((ENCODE42),'econ'); toc, U=U42; S=S42; V=V42; diagS42=diag(S42);
imax=find(diag(S)>max(diag(S))/CondNumb,1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
tic, RECON42=V*invS*U'; toc, 
IMG40=reshape(RECON30i*(ENCODE40j*[imgCS1(:);imgCS2(:)]),[ntx,ntx]);
IMG42=reshape(RECON42* (ENCODE40rcv* [imgCS1(:);imgCS2(:)]),[ntx,2*ntx]);


%==============================================
disp('>> Plotting all results...');
%==============================================
figure(99); set(gcf,'Position',[104,4,1293,1076]); FntSz=12; 
subplot('position',[0.00,0.00,1.0,0.4]);   % IMAGE
imagesc(abs(interp2([IMG00,IMG10,IMG20,IMG30,zeros(ntx,ntx/2),IMG40,zeros(ntx,ntx/2);...
    IMG01,IMG11,IMG21,IMG31,IMG42],2)),[-0.02,1.02]); axis image off, colormap(gca,'gray');
W=1/6;
diagS=gather(diag(S01)); subplot('position',[0*W+0.03,0.43,0.13,0.16]); plot(diagS,'-k',LineWidth=2);
set(gca,'FontSize',FntSz,'yscale','log','xtick',[1,12000,24000],'xticklabel',{'1','12000','24000'},'ytick',10.^(-1:2:3)); grid on, axis([1,length(diagS),1e-1,1e3]), 
diagS=gather(diag(S11)); subplot('position',[1*W+0.03,0.43,0.13,0.16]); plot(diagS,'-k',LineWidth=2);
set(gca,'FontSize',FntSz,'yscale','log','xtick',[1,12000,24000],'xticklabel',{'1','12000','24000'},'ytick',10.^(-1:2:3)); grid on, axis([1,length(diagS),1e-1,1e3]), 
diagS=gather(diag(S21)); subplot('position',[2*W+0.03,0.43,0.13,0.16]); plot(diagS,'-k',LineWidth=2);
set(gca,'FontSize',FntSz,'yscale','log','xtick',[1,12000,24000],'xticklabel',{'1','12000','24000'},'ytick',10.^(-1:2:3)); grid on, axis([1,length(diagS),1e-1,1e3]), 
diagS=gather(diag(S31)); subplot('position',[3*W+0.03,0.43,0.13,0.16]); plot(diagS,'-k',LineWidth=2);
set(gca,'FontSize',FntSz,'yscale','log','xtick',[1,12000,24000],'xticklabel',{'1','12000','24000'},'ytick',10.^(-1:2:3)); grid on, axis([1,length(diagS),1e-1,1e3]), 
diagS=gather(diag(S42)); subplot('position',[4*W+0.03,0.43,0.13+W,0.16]); plot(diagS,'-k',LineWidth=2);
set(gca,'FontSize',FntSz,'yscale','log','xtick',[1,12000,24000],'xticklabel',{'1','12000','24000'},'ytick',10.^(-1:2:3)); grid on, axis([1,length(diagS),1e-1,1e3]), 

subplot('Position',[0*W+0.03,0.63,0.13,0.16]);
plot(Kx1(:),Ky1(:),'-k',LineWidth=0.5); 
set(gca,'XTick',-ntx/2:ntx/4:ntx/2,'YTick',-ntx/2:ntx/4:ntx/2,'FontSize',FntSz), axis image, grid on, axis(72*[-1 1 -1 1]), 
title('Gradient + OffCtr',FontSize=18)
subplot('Position',[1*W+0.03,0.63,0.13,0.16]);
imagesc(GradNonLinCtr); colormap(gca,gray); axis image off, colorbar("east",Color=0.5*[1,1,1],Ticks=[],FontSize=FntSz)
title('+ GradNonLin',FontSize=18)
subplot('Position',[2*W+0.03,0.63,0.13,0.16]);
imagesc(f0Ctr); colormap(gca,gray); axis image off, colorbar("south",Color=0.5*[1,1,1],Ticks=[],FontSize=FntSz)
title('+ OffRes',FontSize=18)
subplot('Position',[3*W+0.03,0.63,0.13,0.16]);
TMP=sensCtr(:,:,[6 7 8 5 5 1 4 3 2]); TMP(:,:,5)=0;
% TMP=circMASK.*sens(:,:,[1:4,4:8]); TMP(:,:,5)=0;
imagesc([reshape(abs(TMP(:,:,1:3)),[ntx,ntx*3]);...
         reshape(abs(TMP(:,:,4:6)),[ntx,ntx*3]);...
         reshape(abs(TMP(:,:,7:9)),[ntx,ntx*3])],[0,0.3]); axis image off, colormap(gca,gray)
title('+RxSens',FontSize=18)
subplot('Position',[4*W+0.03,0.63,0.29,0.16]);
ifreq=linspace(-700,200,901); 
plot(ifreq,6^2./(6^2+(ifreq+500).^2)+6^2./(6^2+(ifreq+0).^2),'-k',LineWidth=2); axis([-700 200 -0.2 1.2])
set(gca,'XTick',[-500,0],'XTickLabel',{'Fat','Water'},'YTick',[],'FontSize',FntSz), grid on, axis on, 
title('+ ChemShift',FontSize=18)
>>>>>>> Stashed changes:FIG3_SheppLogan.m
