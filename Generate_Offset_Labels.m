clc
clear 
close all

%% DIRECTORIES

data_dir = '/mrtstorage/datasets/cityscapes/gtFine/val';
save_dir = '/mrtstorage/users/rehman/datasets/cityscapes/gt_custom/offsetvectors/val';



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

                image_offsets_x = zeros(height, width, 'uint8');
                image_offsets_y = zeros(height, width, 'uint8');

                
                % Looking for instances 
                unique_values = unique(image);

                uni_inst = [];
                for l= 1:length(unique_values)
                    if unique_values(l) > 1000
                      uni_inst = [uni_inst unique_values(l)] ; 
                    end    
                end

                %% Filter and add mask distance between points
                for x = 1:length(uni_inst)

                    instance = (image == uni_inst(x));

                    %% Calculate center

                    cardistTrafo = bwdist(~instance, 'Euclidean');
                    [C,I] = max(cardistTrafo(:));
                    [Y,X] = ind2sub(size(cardistTrafo), I);
%                     points_x =[points_x X];
%                     points_y =[points_y Y];

                    %% Local Offset Masks 
                    local_mask_x = zeros(height, width, 'uint8');
                    local_mask_y = zeros(height, width, 'uint8');

                    local_mask_x(Y,X)=  127;
                    local_mask_y(Y,X) = 127;    


                    %% Adding Offset from center

                    for h = 1 : height
                        for w = 1 : width

                            if (X-w>0)
                                local_mask_x(h,w) = 127 + (abs(X-w)/2);
                            end

                            if (X-w<=0)
                                local_mask_x(h,w) = 127 - (abs(X-w)/2);
                            end

                            if (Y-h>0)
                                local_mask_y(h,w) = 127 + (abs(Y-h)/2);
                            end

                            if (Y-h<=0)
                                local_mask_y(h,w) = 127 - (abs(Y-h)/2);
                            end

                        end
                    end


                      %% Mask values that are outside single instance mask 
                    local_mask_x = local_mask_x.* uint8(instance);
                    local_mask_y = local_mask_y.* uint8(instance);

                    %% Add to complete image mask
                    image_offsets_x = image_offsets_x + local_mask_x;

                    image_offsets_y = image_offsets_y + local_mask_y;

                    extra_channel = ones(height, width).*logical(image_offsets_x);


                end
                Label = cat(3, image_offsets_x, image_offsets_y, extra_channel);
                
  
                save_path = fullfile(save_dir, main_dir(j).name, file(k).name);
                imwrite(Label, save_path , 'png');
        end
    end
end
