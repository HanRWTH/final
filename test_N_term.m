%% This slot generates three infilled depth maps
% compare them with the rendered normal map
% infill depth map using htvl1 model 
f0 = imread('11_PointsDepth.png');
figure; imshow(f0, []);
f0 = double(f0(:,:,1));
f0 = f0/255.0;
u = htvl1_model(f0);
imwrite(u, 'final/htvl1_20.png');


%%
% normal_rendered = imread('12_PointsNormalScreen.png');
% n_x = normal_rendered(:,:,1);
% imwrite(n_x, 'final/7_PointsNormalScreen_x.png');
% n_y = normal_rendered(:,:,2);
% imwrite(n_y, 'final/7_PointsNormalScreen_y.png');
% n_z = normal_rendered(:,:,3);
% imwrite(n_z, 'final/7_PointsNormalScreen_z.png');
%%
u_20 = double(imread('htvl1_20.png'))/255.0;
u_50 = double(imread('htvl1_50.png'))/255.0;
u_100 = double(imread('htvl1_100.png'))/255.0;
[n_x, n_y, n_z] = read_normal_img('11_PointsNormalScreen.png');


% u_Fx = Fx(u_20);
% u_Fy = -Fy(u_20);

gamma = 1; %10
sigma = 0.5; %0.01
mask=create_binary_mask(f0, 0, 1);%[0,1]
% sum(abs(Fx(u_20) - sigma*n_x) + abs(Fy(u_20) - sigma*n_y), "all")

loss_20 = sum((gamma *mask) .* (abs(Fx(u_20) - sigma*n_x) + abs(-Fy(u_20) - sigma*n_y)),"all");
loss_20_tmp = sum((gamma *mask) .* (abs(Fx(u_20) ) + abs(-Fy(u_20) )),"all");

loss_50 = sum((gamma *mask) .* (abs(Fx(u_50) - sigma*n_x) + abs(-Fy(u_50) - sigma*n_y)),"all");
loss_100 = sum((gamma *mask) .* (abs(Fx(u_100) - sigma*n_x) + abs(-Fy(u_100) - sigma*n_y)),"all");
loss_100_tmp = sum((gamma *mask) .* (abs(Fx(u_100) ) + abs(-Fy(u_100) )),"all");

% loss_20 = (gamma *mask) .* (abs(Fx(u_20) - sigma*n_x) + abs(-Fy(u_20) - sigma*n_y));
% loss_50 = (gamma *mask) .* (abs(Fx(u_50) - sigma*n_x) + abs(-Fy(u_50) - sigma*n_y));
% loss_100 = (gamma *mask) .* (abs(Fx(u_100) - sigma*n_x) + abs(-Fy(u_100) - sigma*n_y));
imshow(loss_20,[]);imwrite(loss_20, 'final/loss_20.png');
imshow(loss_50,[]);imwrite(loss_50, 'final/loss_50.png');
imshow(loss_100,[]);imwrite(loss_100, 'final/loss_100.png');

%%
% Assuming Fx_u, Fy_u, n_x, n_y are all matrices of the same size
% and correspond to the gradients and normal components at each pixel.
u_Fx = Fx(u_20);
u_Fy = -Fy(u_20);
% Sample the data to avoid overcrowding the plot.
% You can change the sampling rate depending on your data resolution.
sampling_rate = 10; % For example, every 20th data point.

% Prepare a grid of coordinates where vectors will be plotted.
[X, Y] = meshgrid(1:sampling_rate:size(u_Fx, 2), 1:sampling_rate:size(u_Fx, 1));

% Sample the gradient and normal components based on the sampling rate.
U_depth = u_Fx(1:sampling_rate:end, 1:sampling_rate:end);
V_depth = u_Fy(1:sampling_rate:end, 1:sampling_rate:end);
U_normal = n_x(1:sampling_rate:end, 1:sampling_rate:end);
V_normal = n_y(1:sampling_rate:end, 1:sampling_rate:end);

% Normalize vectors for better visualization.
norm_factor_depth = sqrt(U_depth.^2 + V_depth.^2)+eps;
norm_factor_normal = sqrt(U_normal.^2 + V_normal.^2)+eps;
U_depth = U_depth ./ norm_factor_depth;
V_depth = V_depth ./ norm_factor_depth;
U_normal = U_normal ./ norm_factor_normal;
V_normal = V_normal ./ norm_factor_normal;
back_mask = f0 ~= 1;%[0,1]
back_mask = ~(back_mask(1:sampling_rate:end, 1:sampling_rate:end));
U_normal(back_mask) = 0;
V_normal(back_mask) = 0;

% Now plot the depth map gradients.
figure;
quiver(X, Y, U_depth, V_depth, 'r');
hold on; % Keep the plot for overlaying the normal vectors.
quiver(X, Y, U_normal, V_normal, 'b');
hold off;

% Adjust the plot properties for better visualization.
axis equal tight; % Equal scaling and tight fitting to the vectors.
legend('Depth Gradient', 'Normals');
title('Visual Comparison of Depth Map Gradients and Normals');
xlabel('X Axis');
ylabel('Y Axis');

% Depending on the data, you might want to reverse the axis directions.
% set(gca, 'YDir', 'reverse');
% set(gca, 'XDir', 'reverse');

%%
[n_x, n_y, n_z] = read_normal_img('11_PointsNormalScreen.png');
u_Fx = Fx(u_100);
u_Fy = -Fy(u_100);
% Visualize X components of the gradient and normal map
figure;
subplot(1, 2, 1); % This creates a subplot for the first image
imagesc(2*u_Fx); % Visualize the X component of the depth map gradient
colormap jet; % Use the jet color map for better visibility
colorbar; % Show a color bar for reference
axis image; % Set the axis limits to the image size
title('Gradient of Depth Map (X component)');

back_mask = f0 ~= 1;%[0,1]
n_x(~back_mask) = 0;
subplot(1, 2, 2); % This creates a subplot for the second image
imagesc(n_x); % Visualize the X component of the normal map
colormap jet; % Use the jet color map for better visibility
colorbar; % Show a color bar for reference
axis image; % Set the axis limits to the image size
title('Normal Map (X component)');

% Visualize Y components of the gradient and normal map
figure;
subplot(1, 2, 1); % This creates a subplot for the first image
imagesc(2*u_Fy); % Visualize the Y component of the depth map gradient
colormap jet; % Use the jet color map for better visibility
colorbar; % Show a color bar for reference
axis image; % Set the axis limits to the image size
title('Gradient of Depth Map (Y component)');

subplot(1, 2, 2); % This creates a subplot for the second image
n_y(~back_mask) = 0;
imagesc(n_y); % Visualize the Y component of the normal map
colormap jet; % Use the jet color map for better visibility
colorbar; % Show a color bar for reference
axis image; % Set the axis limits to the image size
title('Normal Map (Y component)');

%% think about other methods to make use of normal map

%% issues by pruning, too much or too little prune, again new parameters

%% This slot tests the parameters
% read paper about how to set the parameter

%% This slot tests using RGB to correct infilled Depth map, or fix 
% the oversmoothing RGB with infilled Depth map

%%
