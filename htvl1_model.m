function [u] = htvl1_model(f0)
% htvl1 model (l1 norm data fidelity, first and second order term)

padNum=5;
f0=padarray(f0,[padNum,padNum],'symmetric');

[m,n]=size(f0);
f0=double(f0);
% mask=create_binary_mask(f0, 0, 255);%[0,255]
mask=create_binary_mask(f0, 0, 1);%[0,1]

u=f0;

%% data fidelity l1 norm
% if use_l1_dataFidelity
z=zeros(m,n);
b01=zeros(m,n);
% end
%% gradient l1 norm
w1=zeros(m,n);
w2=zeros(m,n);
b11=zeros(m,n);%grad_x
b12=zeros(m,n);%grad_y

%% hessian l1 norm
v1=zeros(m,n);
v2=zeros(m,n);
v3=zeros(m,n);
b21=zeros(m,n);%hess_11
b22=zeros(m,n);%hess_12, %hess_21
b23=zeros(m,n);%hess_22

%% weight-parameters 
alfa=ones(m,n)*1.0; %weight for first order variational term
theta1=5;%update rate for first order variational term

beta=ones(m,n)*1.0; %weight for second order variational term
theta2=10;%update rate for second order variational term

l1=5.0;%60.0;%15
lamda=mask*l1; %weight for data fidelity

theta3=ones(m,n)*30;%update rate for data fidelity
%theta3 should be much larger than theta1 and theta2 to avoid oversmoothing

threshold = -0.05; %huber-mask threshold
%% pre-caculate for iterative update
[Y,X]=meshgrid(0:n-1,0:m-1);
G=cos(2*pi*X/m)+cos(2*pi*Y/n)-2;%F(\partical_x^-\partical_x^+)

%% start smoothing
terminate=false;
maxStep=20;

tic
for step=1:maxStep
    if(terminate)
        break;
    end
    temp_u = u;
    step

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update u using FFT
    div_w_b=Bx(b11-w1)+By(b12-w2);
    div_v_b=Fx(Bx(v1-b21))+2*Bx(By(v2-b22))+Fy(By(v3-b23));
    

    z_f_b=(z+f0-b01);
    fft_left=1*theta3-2*theta1*G+4*theta2*G.^2 ;
    g=theta3.*z_f_b+theta1*div_w_b+theta2*div_v_b;

    u=real(ifftn(fftn(g)./fft_left));
    % u(u<0)=0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update w using soft thresholding
    c1=Fx(u)+b11;
    c2=Fy(u)+b12;
    abs_c=sqrt(c1.^2+c2.^2+eps);
    w1=max(abs_c-alfa/theta1,0).*c1./abs_c;
    w2=max(abs_c-alfa/theta1,0).*c2./abs_c;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update v using soft thresholding
    s1=Bx(Fx(u))+b21;
    s2=Fy(Fx(u))+b22;
    s3=By(Fy(u))+b23;
    abs_s=sqrt(s1.^2+2*s2.^2+s3.^2+eps);
    v1=max(abs_s-beta/theta2,0).*s1./abs_s;
    v2=max(abs_s-beta/theta2,0).*s2./abs_s;
    v3=max(abs_s-beta/theta2,0).*s3./abs_s;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update z using soft thresholding
    huber_mask=getHuberMask(u, f0, mask, threshold);%one-side huber loss
 
    z_c=u-f0+b01;
    abs_zc=abs(z_c)+eps;
    lamda(huber_mask)=0.001;
    z=max(abs_zc-lamda./theta3,0).*z_c./abs_zc;  
    % update Bregman iterative parameters
    b01=z_c-z;        
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update Bregman iterative parameters
    b11=c1-w1;
    b12=c2-w2;
    
    b21=s1-v1;
    b22=s2-v2;
    b23=s3-v3;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    terminate=isConvergent(temp_u,u);
end

toc

%% show and save image
u(u<0)=0;
u(u>1)=1;
u=u(padNum+1:m-padNum,padNum+1:n-padNum);
figure; imshow(u, [])
% uint8_image = im2uint8(u/255.0); % Convert to uint8
% imwrite(uint8_image, 'data/hotvl1/tvbhl1-n.png'); % Save the image as a PNG filg2;
end

