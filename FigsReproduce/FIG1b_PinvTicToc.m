% Scientific reports paper submisson Fig. 1b
% Timing the different matrix decomposition methods on the CPU and GPU
% Note: this only runs up to MTX 64 x 64, change line 8 to run longer
% evaluation

clear all, close all, clc, 

Ni=[32,48,64]; % Ni=[32,64:64:320];

for i1=1:length(Ni)
    N=Ni(i1)^2; 
    E=randn(N,N,'single')+1i*randn(N,N,'single'); 
    % CPU
    if Ni(i1)<257
        tic, [U,S,V]=svd(E,'econ'); cpuSVD(i1)=toc, clear U S V; 
        tic, [Q,R,p1]=qr(E,'econ','vector'); cpuQRD(i1)=toc, clear Q R p1;
    end

    tic, EHE=E'*E+1.0*eye(N,N); cpuEHE(i1)=toc, 
    tic, [V,D]=eig(EHE); cpuEIG(i1)=toc, clear V D; 
    tic, L=chol(EHE); cpuCHOL(i1)=toc, clear L;
    % GPU
    if Ni(i1)<193
        E=gpuArray(E);
        tic, [U,S,V]=svd(E,'econ'); gpuSVD(i1)=toc, clear U S V; 
        tic, [Q,R,p1]=qr(E,'econ','vector'); gpuQRD(i1)=toc, clear Q R p1;
        
        tic, EHE=E'*E+1.0*eye(N,N); gpuEHE(i1)=toc, clear E;
        tic, [V,D]=eig(EHE); gpuEIG(i1)=toc, clear V D; 
        tic, L=chol(EHE); gpuCHOL(i1)=toc, clear L;
    end
end

%% plotting
font_sz=10;lw=2;

figure(99), subplot('Position',[0.1,0.2,0.8,0.7])
[x,ii]=sort(Ni);
y=cpuSVD(ii);  plot(x,y,'-b','LineWidth',lw); hold on,
y=cpuEHE(ii);  plot(x,y,'-k','LineWidth',lw); 
y=cpuCHOL(ii); plot(x,y,'-g','LineWidth',lw); 
y=cpuQRD(ii);  plot(x,y,'-c','LineWidth',2); 
y=cpuEIG(ii);  plot(x,y,'-m','LineWidth',2); 

x=x(1:end-2), ii=ii(1:end-2)
y=gpuSVD(ii);  plot(x,y,':b','LineWidth',lw); 
y=gpuEHE(ii);  plot(x,y,':k','LineWidth',lw); 
[x,ii]=sort(Ni); 
y=gpuCHOL(ii); plot(x,y,':g','LineWidth',lw);
y=gpuQRD(ii);  plot(x,y,':c','LineWidth',2); 
y=gpuEIG(ii);  plot(x,y,':m','LineWidth',2); 
hold off,  

legend('SVD','E^HE','CHOL','QR','EIG','Location','southeast','Orientation','vertical','FontSize',32)
axis([32,320,5e-3,1e5]);
grid on; grid off, grid on, 
set(gca,'XScale','log','YScale','log','XTick',sort([64,128,256]),'FontSize',font_sz*1.4,...
    'xticklabel',{' {\bfN_k x N_r\rm}: 4096 x 4096\newline {\bfMTX_{1D}\rm}: 4096 \newline {\bfMTX_{2D}\rm}: 64^2 \newline {\bfMTX_{3D}\rm}: 16^3 \newline              (~134MB)', ...
    '16384 x 16384\newline 16384\newline 128^2\newline 25.4^3\newline(~2.15GB)', ...
    '65536 x 65536\newline 65536\newline 256^2\newline 10.3^3\newline(~34.4GB)'},...
    'YTick',sort([1,60,60*60]),'YTickLabel',{'1sec','1min','1hour'});
ylabel('\bfTime');