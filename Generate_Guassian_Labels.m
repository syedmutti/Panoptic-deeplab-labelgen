clc
clear 
close all

%% DIRECTORIES

data_dir = '/mrtstorage/datasets/cityscapes/gtFine/train';
save_dir = '/mrtstorage/users/rehman/datasets/cityscapes/GTcustom/heatmaps/train';


%% Generate Guassian with standard deviation of 8 pixels

mu = [0 0];
stdDev = 8;
sigma = [stdDev*2 0; 0 stdDev*2];
gridStep = 1;
factorSigma = 1;

x1 = mu(1) - sigma(1,1)* factorSigma : gridStep: mu(1) + sigma(1,1)* factorSigma;
x2 = mu(2) - sigma(2,2)* factorSigma : gridStep: mu(2) + sigma(2,2)* factorSigma;
[X1, X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

y = mvnpdf(X, mu, sigma);
y = reshape(y,length(x2), length(x1));

y = (y - (min(min(y))))./(max(max(y)- min(min(y))));


%% Generate Label for each Image

main_dir = dir(data_dir);

for i = 3:length(main_dir)
    fprintf('%s\n',main_dir(i).name)
    
    for j = 3:length(main_dir)
        current_dir = fullfile(main_dir(j).folder, main_dir(j).name);
        file = dir(fullfile(current_dir, '*instanceIds.png'));
        for k= 1:length(file)

                % Reading Image
                current_image_path = fullfile(file(k).folder, file(k).name);
                image = imread(current_image_path);
               
                % Creating Labels
                [height, width] = size(image);
                label = zeros(height, width);
                
                % Looking for instances 
                unique_values = unique(image);

                x = [];
                for l= 1:length(unique_values)
                    if unique_values(l) > 1000
                      x = [x unique_values(l)] ; 
                    end    
                end
                %% Extract Center Points

                for m = 1:length(x)

                    instance = (image == x(m));

                    car1distTrafo = bwdist(~instance);
                    [C,I] = max(car1distTrafo(:));
                    [Y,X] = ind2sub(size(car1distTrafo), I);

                    %% Apply Guassian to independent centers
                    offset = (length(y)-1)/2;
                   
                    if (X-offset < 0)
                        clip = abs(X-offset)
                        label(Y-offset : Y+offset, 1 : X+offset) = label(Y-offset : Y+offset, 1 : X+offset) + y(:, clip+2:end);
                   
                    elseif (X+offset > width)
                        clip = abs(X+offset-width)
                        label(Y-offset : Y+offset, X-offset : X+offset-clip) = label(Y-offset : Y+offset, X-offset : X+offset-clip) + y(:,1:33-clip);
                        
                    elseif(Y-offset < 0)
                        clip = abs(Y-offset)
                        label(1 : Y+offset, X-offset : X+offset) = label(1 : Y+offset, X-offset:X+offset) + y(clip+2:end,:);
                    
                    elseif(Y+offset > height)
                        clip = abs(Y+offset-height)
                        label(Y-offset : Y+offset-clip, X-offset : X+offset) = label(Y-offset : Y+offset-clip, X-offset:X+offset) + y(1:33-clip,:);
                  
                    % Removing O issue in subtraction 
                    elseif (X-offset == 0)
                        clip = abs(X-offset)
                        label(Y-offset : Y+offset, 1 : X+offset) = label(Y-offset : Y+offset, 1 : X+offset) + y(:, 2:end);
                        
                    elseif(Y-offset == 0)
                        clip = abs(Y-offset)
                        label(1 : Y+offset, X-offset : X+offset) = label(1 : Y+offset, X-offset:X+offset) + y(2:end,:);
                    
                    else
                        label(Y-offset : Y+offset, X-offset : X+offset) = label(Y-offset : Y+offset, X-offset:X+offset) + y(:,:);
                    end
                    
                end
              
                
                save_path = fullfile(save_dir, main_dir(j).name, file(k).name);
                imwrite(label, save_path , 'png');
        end
    end
end
