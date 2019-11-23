function test_render

scene=example_scene();
sigma=1;% edge antialising width

tic
[Abufferc2,Zbuffer2]=render(scene,sigma);
toc
figure(1);
subplot(1,2,1)
imshow(permute(Abufferc2,[2,3,1]),[0,1]);

subplot(1,2,2)
imshow(1-Zbuffer2,[0,1]);

function scene=example_scene()

    rng('default');
    rng(10);
    Ntri=50;
    SizeW=200;
    SizeH=200;

    material=double(permute(imread('../data/trefle.jpg'),[3,1,2]))/255;
    Hmaterial=size(material,2);
    Wmaterial=size(material,3);

    scale_matrix=[SizeH,0;0,SizeW];
    scale_material=[Hmaterial-1,0;0,Wmaterial-1];
    triangles=cell(Ntri,1);

    for k=1:Ntri

        tmp=scale_matrix*(rand(2,1)*[1,1,1]+0.5*(-0.5+rand(2,3)));
        while abs(det([tmp;[1,1,1]]))<1000
            tmp=scale_matrix*(rand(2,1)*[1,1,1]+0.5*(-0.5+rand(2,3)));
        end
        if det([tmp;[1,1,1]])<0
            tmp=fliplr(tmp) ;
        end
        triangle=[];
        triangle.ij=tmp;                                               % coordinate of the triangle vertices in the image
        triangle.depths=rand(1)*[1,1,1];                              % depth of the triangle vertices (for occlusion handling)
        textured=0;rand(1)>0.5;
        if textured
            triangle.uv=scale_material*[0,1,0;0,0,1]+1;    % texture coordinate of the vertices
            triangle.shade=rand(1,3);                            % lshade  intensity at each vertex
            triangle.colors= zeros(3,3);
            triangle.textured=1;
            triangle.shaded=1;
        else
            triangle.uv=zeros(2,3);
            triangle.shade=zeros(1,3);
            triangle.colors=rand(3,3);                    % colors of the vertices (can be gray, rgb color,or even other dimension vectors) when using simple linear interpolation across triangles
            triangle.textured=0;
            triangle.shaded=0;
        end
        triangle.edgeflags=[true,true,true]';
        triangles{k}=triangle;

    end
    triangles=cell2mat(triangles);
    nbV=Ntri*3;
    
    scene.faces= uint32(reshape([1:nbV]-1,3,Ntri));
   
    scene.faces_uv= scene.faces;
    scene.uv=cat(2,triangles.uv);
    scene.ij=cat(2,triangles.ij);
    scene.depths=cat(2,triangles.depths);
    scene.shade=cat(2,triangles.shade);
    scene.colors=cat(2,triangles.colors);
    scene.edgeflags=squeeze(cat(2,triangles.edgeflags));
    scene.textured=logical([triangles.textured]);
    scene.shaded=logical([triangles.shaded]);
    scene.SizeH=SizeH;
    scene.SizeW=SizeW;
    scene.texure=material   ;
    backgroundColor=[0.3,0.5,0.7];
    scene.background=repmat(backgroundColor(:),1,scene.SizeH,scene.SizeW);
    for i=1:Ntri
        fprintf("%f\n",0.5*det([scene.ij(:,scene.faces(2,i)+1)-scene.ij(:,scene.faces(1,i)+1),scene.ij(:,scene.faces(3,i)+1)-scene.ij(:,scene.faces(1,i)+1)]))
    end


