
%%
addpath(genpath('..'))

display=1;
save_images=true;

antialiaseError=0;
sigma=1;

scene1=example_scene();

[image1,z_buffer]=render(scene1,sigma);


Ntri=length(scene1.faces);
scene2=scene1;
max_uv= repmat([size(scene1.texture,2),size(scene1.texture,3)]',[1,3*Ntri]);

displacement_magnitude_ij=10;
displacement_magnitude_uv=0;
displacement_magnitude_colors=0;

rng('default');
rng(10);
scene2.ij=scene1.ij+randn([2,3*Ntri])*displacement_magnitude_ij;
scene2.uv=scene1.uv+randn([2,3*Ntri])*displacement_magnitude_uv;
scene2.uv=max( scene2.uv,0);
scene2.uv=min( scene2.uv,max_uv);
scene2.colors=scene1.colors+randn([3,3*Ntri])*displacement_magnitude_colors;

alpha_ij=0.01
beta_ij=0.80;
alpha_uv=0.03;
beta_uv=0.80;
alpha_color=0.001;
beta_color=0.70;


speed_ij=zeros(2,3*Ntri);
speed_uv=zeros(2,3*Ntri);
speed_color=zeros(3,3*Ntri);


nb_max_iter=500;
lisframesGif=ceil((1:2:500.^(1/1.5)).^1.5);

figure(1);
mkdir('./images/')
filename='./images/soup_fitting.gif';
Err=zeros(1,nb_max_iter);
for iter=1:nb_max_iter
    
    [ scene2b,Err(iter),image]= render_and_compare(scene2,sigma,image1,antialiaseError);
    
    if display
        if iter==1
            figure(1)
            p=imshow(permute(image,[2,3,1]));
        else
            set(p,'CData',permute(image,[2,3,1]));
            %drawnow;
        end
        
        %     if save_images && ismember(iter,[1,50,500])
        %         imwrite(permute(image,[2,3,1]),sprintf('images/soup_iter_%d.png',iter))
        %     end
    end
    
    if displacement_magnitude_ij>0
        speed_ij=beta_ij*speed_ij-scene2b.ij_b*alpha_ij;
        scene2.ij=scene2.ij+speed_ij;
    end
    if displacement_magnitude_colors>0
        speed_color=beta_color*speed_color-scene2b.colors_b*alpha_color;
        scene2.colors=scene2.colors+speed_color;
    end
    if displacement_magnitude_uv>0
        speed_uv=beta_uv*speed_uv-scene2b.uv_b*alpha_uv;
        scene2.uv=scene2.uv+speed_uv;
        scene2.uv=max( scene2.uv,0);
        scene2.uv=min( scene2.uv,max_uv);
    end
    
    drawnow
    
    
    if iter == 1
        [imind,cm] = rgb2ind(uint8(permute(image,[2,3,1])*255),256);
        imwrite(imind,cm,filename,'gif', 'DelayTime',0.1,'Loopcount',inf);
    elseif ismember(iter,lisframesGif)
        imind=rgb2ind(uint8(permute(image,[2,3,1])*255),cm);
        imwrite(imind,cm,filename,'gif', 'DelayTime',0.1,'WriteMode','append');
    end
    
end

figure(2)
hold on;
plot(Err,'color',rand(3,1))
