function flattened_image=mat2montage(im,rect,numrows)
% MAT2MONTAGE - a montage funciton that does not interpolate. Automatically
% calculates number of rows and columns to show a square
% INPUT: 3D image (x,y,z)
% 09/2023 Kylie Yeung

[x,y,z]=size(im); 

if nargin<2
    rect=0;
    r=ceil(sqrt(z)); % num rows
    c=ceil(z/r); % num cols
elseif rect
    r=floor(sqrt(z)); % num rows
    c=ceil(z/r); % num cols
elseif nargin>2
    r=numrows; % num rows
    c=ceil(z/r); % num cols
else
    disp('rect must be true or false')
end

flattened_image=zeros(x*r,y*c);

% pad up end of image
pad=zeros(x,y,c*r-z);
im=cat(3,im,pad);

for i=1:r
    flattened_image((i-1)*x+1:i*x,:)=reshape(im(:,:,(i-1)*c+1:i*c),[x,y*c]);
end

if ~isreal(flattened_image)
    warning("Image is complex, only showing abs(image)")
    flattened_image=abs(flattened_image);
end

imagesc(flattened_image);%clim([min(flattened_image(:)) max(flattened_image(:))]);

if rect; axis equal;axis tight;else; axis square;end
axis off
colormap gray

end