function losses = mesh_fitting(image,mask,vertices,faces,vertices_colors,backgroundColor,lights, CameraMatrix, options)

SizeW=size(image,2);
SizeH=size(image,1);

if options.save_images
    mkdir(options.iter_images_folder)
end

if ~exist('AutoDiff','class')
    error('you need to add the automatic differentiation toolbox dowloaded from https://github.com/martinResearch/MatAutoDiff in you path ')
end

if ~options.antialiaseError
    mask=repmat(reshape(mask,1,size(mask,1),size(mask,2)),3,1,1);% duplicating on the 3 chanels
end

M.F = faces;
M.V = vertices;
M.colors = vertices_colors;
M = mesh_adjacencies(M);
Mref = M;


Aobs = double(permute(image,[3,1,2]));

losses = zeros(1,options.nbMaxIter);
durations = zeros(1,options.nbMaxIter);
filteredGrad = zeros(1,numel(M.V));
speed = zeros(1,numel(M.V))';

Miter = Mref;

Laplacian=mesh_laplacian(M);
cT= kron(Laplacian'*Laplacian,speye(3));
edgeDiff = sparse([(1:M.nbE)';(1:M.nbE)'],[M.Edges_Vertices(:,1);M.Edges_Vertices(:,2)],[ones(M.nbE,1);-ones(M.nbE,1)]);
edgeDiff3 = kron(edgeDiff,speye(3));
edgeDiff3T = edgeDiff3';
approxRigidHessian=  options.cregu * (edgeDiff3' * edgeDiff3) + options.gamma * speye(M.nbV*3);

start = tic;
for iter = 1:options.nbMaxIter
    
    MiterAD = Miter;
    MiterAD.V = AutoDiff(Miter.V);
    
    scene3 = mesh2scene(MiterAD,CameraMatrix,lights.ligthDirectional,lights.ambiantLight,SizeH,SizeW,false);
    
    J_col = getderivs(scene3.colors);
    J_ij = getderivs(scene3.ij);
    
    scene2 = mesh2scene(Miter,CameraMatrix,lights.ligthDirectional,lights.ambiantLight,SizeH,SizeW);
    scene2.background = repmat(backgroundColor(:),1,scene2.SizeH,scene2.SizeW);
    
    if options.display   
        [scene2b,Edata,image]= render_and_compare(scene2, options.sigma, Aobs, options.antialiaseError, mask);
    else
        [scene2b] = render_and_compare(scene2, options.sigma, Aobs, options.antialiaseError);
    end
    
    if options.display || options.saveGif
        if iter == 1
            figure(1)
            p = imshow(permute(image,[2,3,1]));
        else
            set(p,'CData',permute(image,[2,3,1]));
            drawnow;
        end
        
        if options.save_images && ismember(iter,[1,20,40])
            imwrite(permute(image,[2,3,1]),fullfile(options.iter_images_folder,sprintf('iter_%d.png',iter)))
        end
    end
    
    if options.save_gif
        filename=fullfile(options.iter_images_folder,'iterations.gif');
        if iter == 1
            [imind,cm] = rgb2ind(uint8(permute(image,[2,3,1])*255),256);
            imwrite(imind,cm,filename,'gif', 'DelayTime',0.2,'Loopcount',inf,'DisposalMethod','leaveInPlace');
        elseif mod((iter-1),1) == 0
            imind=rgb2ind(uint8(permute(image,[2,3,1])*255),cm);
            imwrite(imind,cm,filename,'gif', 'DelayTime',0.2,'WriteMode','append');
        end
    end
    imwrite(permute(image,[2,3,1]),fullfile(options.iter_images_folder,sprintf('iter_%d.png',iter)))
    
    ijB = scene2b.ij_b;
    colorsB = scene2b.colors_b;
    GradData = ijB(:)'*J_ij+colorsB(:)'*J_col;
    
    %regularization: we add the folowing quadratic term to the energy we
    %are minimising ,
    %0.5*norm(L*(M.V-Mref.V))^2
    % with L the lapalacian matrix of the graph corresponding
    %to the mesh , this term penalise deformation with respect to the
    %original mesh and enforce some kind of rigidity to the mesh
    
    H = options.alpha * (J_col' * J_col) + options.beta * (J_ij' * J_ij) + approxRigidHessian;
   
    %H= cTplusGama;
    
    switch options.method
        case 'filteredGradient'
            filteredGrad = 0.8*filteredGrad+options.coefData*GradData;
            G = filteredGrad'+ options.cregu*cT*(Miter.V(:)-Mref.V(:));
            %step=-H\G;
            [step,~] = pcg(H,-G,1e-1,100);
            Miter.V(:) = Miter.V(:)+step;
        case 'heavyBall'
            diff = edgeDiff3*(Miter.V(:)-Mref.V(:));
            grad_regu = options.cregu*edgeDiff3T*diff;
            %Eregu=cregu*0.5*sum(diff.^2);
            Eregu = options.cregu * 0.5 * sum(diff.^2);
            G = GradData' + grad_regu;

            step = -H\G;
            %[step,~]=pcg(H,-G,0.2,100);
            % step=-bicg(H,G,1e-4,10);
            %step=-linsolve(H,G,struct('POSDEF',true));
            speed = (1-options.damping) * (speed * options.inertia + (1-options.inertia) * step);
            Miter.V(:) = Miter.V(:) + speed;
            E = Edata + Eregu;
            fprintf('Energy=%f : Edata=%f Eregu=%f\n',E,Edata,Eregu);
            losses(iter) = E;
        otherwise
            error('unkown method')        
    end
    durations(iter) = toc(start);   
end


if options.display
    figure(2)
    hold on;
    plot(losses,'color',rand(3,1))
    figure(4)
    hold on;
    plot(durations,losses,'color',rand(3,1))
end


fprint('error=%f',losses(end))

end






