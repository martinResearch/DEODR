function [scene,err,combined_image]= render_and_compare(scene,sigma,obs,antialiaseError,mask)

if nargin<4
    antialiaseError=false;
end
if nargin<5
    if antialiaseError
        mask=ones(size(obs,2),size(obs,3));
    else
        mask=ones(size(obs));
    end
end

if antialiaseError
    [image,z_buffer,err_buffer]=render(scene,sigma,antialiaseError,obs);
    
    err_buffer=err_buffer.*mask;
    if nargout>1
        err=sum(err_buffer(:));
    end
    err_buffer_B=double(mask);
    scene.ij_b=zeros(size(scene.ij));
    scene.colors_b=zeros(size(scene.colors));
    scene.uv_b=zeros(size(scene.uv));
    scene.shade_b=zeros(size(scene.shade));
    if nargout==3
        image=cat(3,obs,image,repmat(reshape(sqrt(err_buffer)/sqrt(3),1,size(err_buffer,1),size(err_buffer,2)),3,1,1));
    end
    render_b(scene,image,z_buffer,[],sigma,antialiaseError,obs,err_buffer,err_buffer_B);
    
else
    
    [image,z_buffer]=render(scene,sigma);
    %imwrite(permute(uint8(floor(image*255)),[2,3,1]),'image.png')
    
    diff=(image-obs).*mask;
    if nargout==3
        err_buffer=max(0, sum((diff).^2,1));
        combined_image=cat(3,obs,image,repmat(sqrt(err_buffer)/sqrt(3),3,1,1));
    end
    
    if nargout>1
        err=sum(diff(:).^2);
    end
    image_b=2*diff;
    
    
    scene.ij_b=zeros(size(scene.ij));
    scene.colors_b=zeros(size(scene.colors));
    scene.uv_b=zeros(size(scene.uv));
    scene.texture_b=zeros(size(scene.texture));
    scene.shade_b=zeros(size(scene.shade));
    render_b(scene,image,z_buffer,image_b,sigma);
    
end

