load('train_annotations.mat')
n_images=size(train_path,1);
x_i=cell(n_images,1); % This will contain the whole image 
x_h=cell(n_images,1); % This will contain the heads
x_p=cell(n_images,1); % This will contain the eye grids
x_g=cell(n_images,1); % This is the gaze map and will contain the output
%for i=1:n_images
for i=1:1000
  
   siz=230;   
   e=train_eyes{i}; 
    
   disp(i);
   img=imread(train_path{i});
   img=imresize(img,[siz,siz]);
   x_i{i}=img;

   % This is just extracting the face patch using the normlized co-ordinates of the eyes.
   alpha = 0.3;
   w_x = floor(alpha*size(img,2));
   w_y = floor(alpha*size(img,1));
   if(mod(w_x,2)==0)
        w_x = w_x +1;
   end

   if(mod(w_y,2)==0)
        w_y = w_y +1;
   end
   
   im_face = ones(w_y,w_x,3,'uint8');
   im_face(:,:,1) = 123*ones(w_y,w_x,'uint8');
   im_face(:,:,2) = 117*ones(w_y,w_x,'uint8');
   im_face(:,:,3) = 104*ones(w_y,w_x,'uint8');
   center = floor([e(1)*size(img,2) e(2)*size(img,1)]);
   d_x = floor((w_x-1)/2);
   d_y = floor((w_y-1)/2);

   bottom_x = center(1)-d_x;
   delta_b_x = 1;
   if(bottom_x<1)
        delta_b_x =2-bottom_x;
        bottom_x=1;
    end
    top_x = center(1)+d_x;
    delta_t_x = w_x;
    if(top_x>size(img,2))
         delta_t_x = w_x-(top_x-size(img,2));
         top_x = size(img,2);
    end
    bottom_y = center(2)-d_y;
    delta_b_y = 1;
    if(bottom_y<1)
        delta_b_y =2-bottom_y;
        bottom_y=1;
    end
    top_y = center(2)+d_y;
    delta_t_y = w_y;
    if(top_y>size(img,1))
         delta_t_y = w_y-(top_y-size(img,1));
         top_y = size(img,1);
    end
    
    im_face(delta_b_y:delta_t_y,delta_b_x:delta_t_x,:) = img(bottom_y:top_y,bottom_x:top_x,:);
    x_h{i} = im_face;

    % Now this is how the eye is to be located.
    f = zeros(1,1,169,'single');
    z = zeros(13,13,'single');
    x = floor(train_eyes{i}(1)*13)+1;
    y = floor(train_eyes{i}(2)*13)+1;
    z(x,y) =1;
    f(1,1,:) = z(:);
    x_p{i} = f;
    
    %This is for the gaze information
    
    %This shifted grid information is kept as a reference;
     v_x = [0 1 -1 0 0];
     v_y = [0 0 0 -1 1];
    
    
    f1=zeros(5,5);
    f2=zeros(5,5);
    f3=zeros(5,5);
    f4=zeros(5,5);
    f5=zeros(5,5);
    
    xg=ceil(train_gaze{i}(1)*5)
    yg=ceil(train_gaze{i}(2)*5)
    
    % Just change here later because the logic is not correct
    
    % Just create enough to show that it is working
    
    imat=yg; % interchange of coordinates as is done in images
    jmat=xg;
    
    % Consdier the 4 different cases.
    f1(imat, jmat)=1;
    f2(satFn(imat+1), satFn(jmat))=1;
    f3(satFn(imat-1), satFn(jmat))=1;
    f4(satFn(imat), satFn(jmat-1))=1;
    f5(satFn(imat), satFn(jmat+1))=1;
    
    
  
    f1=reshape(f1,[25,1]);
    f2=reshape(f2,[25,1]);
    f3=reshape(f3,[25,1]);
    f4=reshape(f4,[25,1]);
    f5=reshape(f5,[25,1]);
    
    x_g{i}=[f1;f2;f3;f4;f5];
%     end
    
    
end