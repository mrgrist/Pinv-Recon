% Scientific reports paper submisson Fig. 2
% Comparison of regularization using truncated SVD and Tikhonov
% regularization

ntx=160;
TMP=linspace(-0.5,0.5,ntx+1); [Rx,Ry]=ndgrid(TMP(1:end-1)); 
circMASK=sqrt(Rx.^2+Ry.^2)<0.975*0.5;
IMG0=phantom('Modified Shepp-Logan',ntx);   
IMG0(IMG0>0.75)=0.75; IMG0=IMG0/max(IMG0(:));

MSE0=mean(abs(IMG0(circMASK)).^2);
kMASK=sqrt(Rx.^2+Ry.^2)<0.5;
K0=ifftshift(fftn(fftshift(IMG0)));
I1=ifftshift(ifftn(fftshift(kMASK.*K0)));
MSEring=mean(abs(I1(circMASK)-IMG0(circMASK)).^2);

%=============================================
disp('>> 2D Spiral design (Grad & k-traj)');
%=============================================
mtx=ntx;
FOV=0.192;          % [m]
res=FOV/mtx;        % [m]
Gmax=0.080;         % [T/m]
Smax=200;           % [T/(m*s)]
BW=125e3;           % [Hz]
dt=1/(2*BW);        % [s]
Nint=4;             % [1] number of spiral interleaves
Nnex=2;             % [1] (optional) NEXing
[k,g,s,time,r,theta]=vds(Smax*1e2,Gmax*1e2,dt,Nint,[FOV*1e2,0],1/(2*res*1e2));
% interleaving single spiral arm
TMP=(mtx/2*k/abs(k(end)))' * exp(1i*2*pi*(0:(Nnex*Nint-1))/(Nnex*Nint));
TMP=TMP+2e-3*randn(size(TMP)); % add rnd to avoid overlapping samples at k=0
Kx=real(TMP(:)); Ky=imag(TMP(:));

% Encode
Encode=double(exp(1i*2*pi*(Kx(:)*Rx(:).'+Ky(:)*Ry(:).')));
% Data
Data=Encode*IMG0(:); 
% Noise
inoise=10.^(0:2);
Noise=randn(size(Data))+1i*randn(size(Data))/2;

%================================
disp('>> Tikhonov + Cholesky');
%================================
EHE=Encode'*Encode; 
maxEig=maxEig_GPU(gpuArray(EHE));
iscale=10.^(-6:0.25:5); 
for i1=1:length(iscale)
    [i1,length(iscale)]
    lambda=iscale(i1)*maxEig;
    [L,flag]=chol(EHE+lambda*eye(size(EHE)),'lower'); 
    invL=inv(L); clear L; iEHE=invL'*invL; clear invL;
    % SRF=abs(diag(Recon*Encode)); Noise=abs(diag(Recon*Recon'));
    for i2=1:length(inoise)
        IMG1=gather(reshape(iEHE*(Encode'*(1*Data+1*inoise(i2)*Noise)),[ntx,ntx])); 
        IMG2=gather(reshape(iEHE*(Encode'*(1*Data+0*inoise(i2)*Noise)),[ntx,ntx])); 
        IMG3=gather(reshape(iEHE*(Encode'*(0*Data+1*inoise(i2)*Noise)),[ntx,ntx])); 
        MSE1(i1,i2,1)=mean(abs(IMG1(circMASK)-IMG0(circMASK)).^2);
        MSE2(i1,i2,1)=mean(abs(IMG2(circMASK)-IMG0(circMASK)).^2);
        MSE3(i1,i2,1)=mean(abs(IMG3(circMASK)).^2);
    end
end

%==========================
disp('FW>> truncated-SVD');
%==========================
ienergy=1-10.^(-7:0.1:0)+eps; 
% Encode=gpuArray(double(exp(1i*2*pi*(Kx(:)*Rx(:).'+Ky(:)*Ry(:).'))));
% clear EHE iEHE;
tic, [U,S,V]=svd(Encode,'econ'); toc, 
energySVD=cumsum(diag(S).^2);
for i1=1:length(ienergy)
    disp([i1,length(ienergy)])
    imax=find(energySVD<=ienergy(i1)*max(energySVD),1,'last');
    invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
    % tic, RECON=V*invS*U'; toc, 
    for i2=1:length(inoise)
        IMG1=reshape(V*(invS*(U'*(1*Data+1*inoise(i2)*Noise))),[ntx,ntx]);
        IMG2=reshape(V*(invS*(U'*(1*Data+0*inoise(i2)*Noise))),[ntx,ntx]);
        IMG3=reshape(V*(invS*(U'*(0*Data+1*inoise(i2)*Noise))),[ntx,ntx]);
        MSE1(i1,i2,2)=mean(abs(IMG1(circMASK)-IMG0(circMASK)).^2);
        MSE2(i1,i2,2)=mean(abs(IMG2(circMASK)-IMG0(circMASK)).^2);
        MSE3(i1,i2,2)=mean(abs(IMG3(circMASK)).^2);
    end
end

figure(98), subplot('Position',[0.60,0.07,0.35,0.24]); Farbe='ybg'; ii=1:length(iscale);
hold off, plot(log10(iscale),log10(MSE2(ii,2,1)),'-k','LineWidth',7); hold on, axis([-6 2 -4 0]);
for i2=1:3
    plot(log10(iscale),log10(MSE1(ii,i2,1)),'-','LineWidth',3,'Color',Farbe(i2)); 
end
for i2=1:3
    plot(log10(iscale),log10(MSE3(ii,i2,1)),'--','LineWidth',1,'Color',Farbe(i2)); hold on,
end
[~,i1min]=min(MSE1(ii,:,1));
set(gca,'XTick',[-4 -2, -0.1],'YTick',log10([MSEring,MSE0]),'YTicklabel',{'log_{10}\DeltaRes','log_{10}MSE_0'},...
    'FontSize',16,'YTickLabelRotation',0)
% xlabel('$$log_{10} \frac{\alpha}{\lambda_{max}}$$','Interpreter','latex','FontSize',16); grid on,
xlabel('log_{10} \lambda/\sigma_{max}','FontSize',16); grid on,
legend('0*std_{noise}','0.1*std_{noise}','1.0*std_{noise}','10*std_{noise}','Location','southeast','Fontsize',12)
set(gca,'XDir','reverse')

figure(98), subplot('Position',[0.60,0.40,0.35,0.24]); Farbe='ybg'; ii=1:length(ienergy);
hold off, plot(log10(1-ienergy),log10(MSE2(ii,2,2)),'-k','LineWidth',7); hold on, axis([-7 0 -4 0]);
for i2=1:3
    plot(log10(1-ienergy),log10(MSE1(ii,i2,2)),'-','LineWidth',3,'Color',Farbe(i2)); 
end
for i2=1:3
    plot(log10(1-ienergy),log10(MSE3(ii,i2,2)),'--','LineWidth',1,'Color',Farbe(i2)); hold on,
end
[~,i1min]=min(MSE1(ii,:,2));
set(gca,'XTick',[-4 -2 -0.1],'YTick',log10([MSEring,MSE0]),'YTicklabel',{'log_{10}\DeltaRes.','log_{10}MSE_0'},...
    'FontSize',16,'YTickLabelRotation',0)
xlabel('log_{10} \DeltaEnergy_{truncated}','FontSize',16); grid on,
set(gca,'XDir','reverse')


figure(98); subplot('Position',[0.0,0.00,0.5,0.32]); 
% Encode=double(exp(1i*2*pi*(Kx(:)*Rx(:).'+Ky(:)*Ry(:).')));
% EHE=Encode'*Encode; 
jscale=repmat(10.^[-4 -2 -0.1].',[1,3]); 
tIMG=zeros(3*ntx,3*ntx);
for i1=1:3
    for i2=1:3
        disp([i1,i2]);
        [L,flag]=chol(EHE+jscale(i1,i2)*maxEig*eye(size(EHE)),'lower'); invL=inv(L); clear L; iEHE=invL'*invL; clear invL;
        tIMG((i2-1)*ntx+(1:ntx),(i1-1)*ntx+(1:ntx))=...
            gather(reshape(iEHE*(Encode'*(1*Data+1*inoise(i2)*Noise)),[ntx,ntx])); 
    end
end
TMP=gather(reshape((Encode'*(1*Data+0*Noise)),[ntx,ntx]));
tIMG((1-1)*ntx+(1:ntx),(4-1)*ntx+(1:ntx))=TMP/max(abs(TMP(:)));
tIMG_new=reshape(flip(reshape(tIMG,[mtx*3,mtx,4]),3),size(tIMG)); % flip to the direction matching svd
imagesc(abs(tIMG_new),[0,1.2]); axis image off, colormap gray;
% title('Tikhonov Regularization','FontSize',24)

figure(98); subplot('Position',[0.0,0.33,0.5,0.32]); 
jenergy=repmat(1-10.^[-4 -2 -0.1].',[1,3]);
sIMG=zeros(3*ntx,4*ntx);
for i1=1:3
    for i2=1:3
        disp([i1,i2]);
        imax=find(energySVD<=jenergy(i1,i2)*max(energySVD),1,'last'); invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
        sIMG((i2-1)*ntx+(1:ntx),(i1-1)*ntx+(1:ntx))=...
            reshape(V*(invS*(U'*(1*Data+1*inoise(i2)*Noise))),[ntx,ntx]);
    end
end
imax=100; invS=1./diag(S); invS(imax+1:end)=0; invS=diag(invS);
sIMG((1-1)*ntx+(1:ntx),(4-1)*ntx+(1:ntx))=...
    reshape(V*(invS*(U'*(1*Data+1*inoise(i2)*Noise))),[ntx,ntx]);
sIMG_new=reshape(flip(reshape(sIMG,[mtx*3,mtx,4]),3),size(sIMG));
imagesc(abs(sIMG_new),[0,1.2]); axis image off, colormap gray;
% title('Truncated SVD','FontSize',24)

jscale=10.^[-4 -2 -0.1];
e0=eig(EHE+0*maxEig*eye(size(EHE))); toc, 
e1=eig(EHE+jscale(1)*maxEig*eye(size(EHE))); toc, 
e2=eig(EHE+jscale(2)*maxEig*eye(size(EHE))); toc, 
e3=eig(EHE+jscale(3)*maxEig*eye(size(EHE))); toc, 
jenergy=(1-10.^[-4 -2 -0.1].');
imax1=find(energySVD<=jenergy(1)*max(energySVD),1,'last');
imax2=find(energySVD<=jenergy(2)*max(energySVD),1,'last');
imax3=find(energySVD<=jenergy(3)*max(energySVD),1,'last');
figure(98); subplot('Position',[0.6,0.72,0.35,0.25]); 
E=[e0,e1,e2,e3]/max(e0(:)); E(E<eps)=eps;
plot(log10(flipud(E(:,1))),'-k','LineWidth',6); hold on, axis([1 length(e0) -6 1])
plot(log10(flipud(E(:,2:end))),'--k','LineWidth',2); hold on, 
set(gca,'xtick',sort(gather([1,100,imax3,imax2,imax1,length(e0)])),'XTickLabel',{'1','','','','',num2str(length(e0))},...
    'FontSize',16); grid on, ylabel('\sigma/\sigma_{max}')

figure(98); subplot('Position',[0.1,0.72,0.35,0.25]); hold off, 
TMP=Kx+1i*Ky; TMP=TMP(1:length(TMP)/Nint/Nnex);
plot(real(TMP),imag(TMP),'-k'); axis image off, axis(82*[-1 1 -1 1])
axis on, box on, grid on,
set(gca,'XTick',[-64:32:64],'YTick',[-64:32:64],'FontSize',16)
xlabel('k_x','FontSize',16);ylabel('k_y','FontSize',16)