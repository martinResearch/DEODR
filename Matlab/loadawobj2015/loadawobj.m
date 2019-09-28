function [V,F3,F4]=loadawobj(modelname,opts)
% loadawobj load a Wavefront/Alias obj style model.
%    It will only consider polygons with 3 or 4 vertices.
%    [v,f3,f4]=loadawobj(modelname)
%    s=loadawobj(modelname)
%    s=loadawobj(modelname,1) % will load vertex normals and textures
% Programme will normally ignore normal and texture data unless the
% option is given. 
% It will also ignore any part of the obj specification that is not a
% polygon mesh, ie nurbs and the materials file (.mtl). Please let me
% know if these are needed.
%
% Examples:
%   loadawobj('name.obj') will load and display name.obj
%   S=loadawobj('name.obj') will load obj into S
%   patch('Vertices',S.v','Faces',S.f3','FaceColor','g');
%
%
% s=loadawobj(modelname) gives a structure
%  version
%  vertices
%  f3 (triangles)
%  f4 (quadrilaterals)
%  groups
%  g3 faces per group
%  g4 faces per group
%  Vn3 Vertex normals indices for f3
%  Vn4 Vertex normals indices for f4
%  Vertex normals (if option is given)
%  Vertex textures (if option is given)
%
% where the file is grouped the following code fragment can be adapted
% to split out groups
%
% clf; axis('equal'); col='mybgrcmy';
% for jj=3:length(S.g)+2
%     patch('Vertices',S.v','Faces',S.f3(:,S.g3(jj-1):S.g3(jj)-1)','FaceColor',col(jj));
%     pause;
% end; 
%
% See also Anders Sandberg's vertface2obj.m and saveobjmesh.m
% 
% there is still a good chance that obj files will not load. I
% would be grateful for any reports and examples of those that fail.
%
% W.S. Harwin, University Reading, 2006,2010,2015.
% Matlab BSD license
% thanks also to Doug Hackett


version=0.3;
if nargin <1 
  disp('specify model name')
end


fid = fopen(modelname,'r');
if (fid<0)
  error(['can not open file: ' modelname]);
  return ;
end

vnum=1;
f3num=1; % triangle (f3) faces so far
f4num=1; % quad (f4) faces so far
vtnum=1; % vertex textures so far
vnnum=1; % vertex normals so far
gnum=0;  % groups so far
g3num=1; %
g4num=1;
Vtmp=[]; % set to null so that it can be assigned in the structure

% Line by line passing of the obj file

while ~feof(fid)
  Ln=fgets(fid);
  Ln=removespace(Ln);
  objtype=sscanf(Ln,'%s',1);
  l=length(Ln);
  if l==0  % isempty(s) ; 
%    disp(['empty' Ln]);
    continue
  end
%  disp(Ln)
  Lyn=Ln; %temp hack
  switch objtype
    case '#' % comment
      disp(Ln);
    case 'v' % vertex
      v=sscanf(Ln(2:end),'%f');
      Vtmp(:,vnum)=v;
      vnum=vnum+1;
    case 'vt'	% textures
      if nargin>=2; % option to collect 
          vtxtr=sscanf(Ln(3:end),'%f');
          Vtexture(:,vtnum)=vtxtr;
          vtnum=vtnum+1;
      end
    case 'g' % sub mesh
      disp(Ln);
%      if f3num > g3num(end);
      g3num=[g3num f3num];
      g4num=[g4num f4num];
      gnum=gnum+1;
      g{gnum}=Ln(3:end);
    case 'mtllib' % material library
        disp(Ln);
    case 'usemtl' % use this material name
        disp(Ln);
    case 'l' % Line
        disp(Ln);
    case 's' %smooth shading across polygons
        disp(Ln);
    case 'vn' % normals
      if nargin>=2; % option to collect 
          vnorms=sscanf(Ln(3:end),'%f');
          Vnorms(:,vnnum)=vnorms;
          vnnum=vnnum+1;
      end
    case 'f' % faces
      nvrts=length(findstr(Ln,' ')); % spaces as a predictor of n vertices
      slashpat=findstr(Ln,'/');
      nslash=length(slashpat);
      if nslash >1 % dblslash can be 0, 1 or >1
        dblslash=slashpat(2)-slashpat(1); else dblslash=0; 
      end
      Ln=Ln(3:end); % get rid of the f
      if nslash == 0 % Face = vertex
%        disp('-v');
        f1=sscanf(Ln,'%f');
      elseif nslash == nvrts && dblslash>1 % Face = v/tc
%        disp('-v/tc');
        data1=sscanf(Ln,'%f/%f');
        if nvrts == 3 
          f1=data1([1 3 5]);
          tc1=data1([2 4 6]);
        end
        if nvrts == 4;
          f1=data1([1 3 5 7]);
          tc1=data1([2 4 6 8]);
        end
      elseif nslash == 2*nvrts && dblslash==1 % v//n
%        disp('-v//n');
        data1=sscanf(Ln,'%f//%f');
        if nvrts == 3 
          f1=data1([1 3 5]);
          vn1=data1([2 4 6]);
          Vn3(:,f3num)=f1;
        end
        if nvrts == 4;
          f1=data1([1 3 5 7]);
          vn11=data1([2 4 6 8]);
          Vn4(:,f4num)=f1;
        end
      elseif nslash == 2*nvrts && dblslash>1 % v/tc/n
%        disp('-v/tc/n');
        data1=sscanf(Ln,'%f/%f/%f');
        if nvrts == 3
          f1=data1([1 4 7]);
          tc1=data1([2 5 8]);
          vn1=data1([3 6 9]);
          Vn3(:,f3num)=f1;
        end
        if nvrts == 4;
          f1=data1([1 4 7 10]);
          tc1=data1([2 5 8 11]);
          vn1=data1([3 6 9 12]);
          Vn4(:,f4num)=f1;
        end
      end
% Now put the data into the array(s)
      if nvrts == 3
          F3(:,f3num)=f1;
          f3num=f3num+1;
      elseif nvrts ==4
          F4(:,f4num)=f1;
          f4num=f4num+1;
      else
          warning(sprintf('v nvrts=%d %s',nvrts, Ln));
      end       
    otherwise 
%      if ~strcmp(Lyn,char([13 10])) % carriage return
%        if ~ all(Lyn == ' ') % ignore lines with only space
          disp(['unprocessed-' Ln '-']); % see what has not been processed
%          double(Ln)
          %        end
%      end
    end
  
end

fclose(fid);



% plot if no output arguments are given
if nargout ==0
  if exist('F3','var') 
    patch('Vertices',Vtmp','Faces',F3','FaceColor','g');
  end
  if exist('F4','var')
    patch('Vertices',Vtmp','Faces',F4','FaceColor','b');
  end
  axis('equal')
  clear Vtmp F3 F4
end

if nargout >=2 
  V=Vtmp;
  if ~ exist('F3','var') 
    warning('No 3 element faces')
    F3=[];
  end
  if nargout ==3
    if ~ exist('F4','var') 
      warning('No 4 element faces')
      F4=[];
    end
  end
end

if nargout ==1 
  V.version=version;
  V.v=Vtmp;
  if exist('F3','var')
      V.f3=F3;  end
  if exist('F4','var')
    V.f4=F4;  end
  if gnum>0
      V.g=g;  end
  V.g3=[g3num f3num];
  V.g4=[g4num f4num];
  if exist('Vn3','var')
    V.Vn3=Vn3;  end
  if exist('Vn4','var')
    V.Vn4=Vn4;  end
  if exist('Vnorms','var')
    V.Vnorms=Vnorms;  end
  if exist('Vtexture','var')
    V.Vtexture=Vtexture;  end
end


function Lyn=removespace(Lyn)
% A not an elegant way to remove
% surplus space
Lyn=strtrim(Lyn);
Lyn=strrep(Lyn,'       ',' '); % 8-2 .. 12-6  
Lyn=strrep(Lyn,'    ',' '); % 5-2 6-3 4-1
Lyn=strrep(Lyn,'  ',' '); % 3-2 2-1
Lyn=strrep(Lyn,'  ',' '); 
Lyn=strrep(Lyn,char([13 10]),''); % remove cr/lf 
Lyn=strrep(Lyn,char([10]),''); % remove lf 
