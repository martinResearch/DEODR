function [scene,Err,image]= render_and_compare(scene,sigma,Aobs,antialiaseError,mask)

if nargin<4
    antialiaseError=false;
end
if nargin<5
    if antialiaseError
        mask=ones(size(Aobs,2),size(Aobs,3));
    else  
        mask=ones(size(Aobs));
    end
end

if antialiaseError
    [Abuffer,Zbuffer,ErrBuffer]=render(scene,sigma,antialiaseError,Aobs);
    
    ErrBuffer=ErrBuffer.*mask;
    if nargout>1
       Err=sum(ErrBuffer(:));
    end
    ErrBuffer_B=double(mask);    
    scene.ij_b=zeros(size(scene.ij));
    scene.colors_b=zeros(size(scene.colors));
    scene.uv_b=zeros(size(scene.uv));
    scene.shade_b=zeros(size(scene.shade));
    if nargout==3
    image=cat(3,Aobs,Abuffer,repmat(reshape(sqrt(ErrBuffer)/sqrt(3),1,size(ErrBuffer,1),size(ErrBuffer,2)),3,1,1)); 
    end
    render_b(scene,Abuffer,Zbuffer,[],sigma,antialiaseError,Aobs,ErrBuffer,ErrBuffer_B);

else
    
    [Abuffer,Zbuffer]=render(scene,sigma);
    %imwrite(permute(uint8(floor(Abuffer*255)),[2,3,1]),'Abuffer.png')
   
     diff=(Abuffer-Aobs).*mask;  
       if nargout==3
            ErrBuffer=max(0, sum((diff).^2,1));
    image=cat(3,Aobs,Abuffer,repmat(sqrt(ErrBuffer)/sqrt(3),3,1,1));
       end

    if nargout>1
        Err=sum(diff(:).^2);
    end
        Abuffer_b=2*diff;

    
    scene.ij_b=zeros(size(scene.ij));
    scene.colors_b=zeros(size(scene.colors));
    scene.uv_b=zeros(size(scene.uv));
    scene.shade_b=zeros(size(scene.shade));       
   render_b(scene,Abuffer,Zbuffer,Abuffer_b,sigma);
   
end

