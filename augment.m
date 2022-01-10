function aug = augment(groups, type)
%Augment This function augments the data of the original dataset as
%required in part 2 of the project
%   Stores the augmented data in the following location

if(type==0)
    data_address = './data/wallpapers/train';
    output_address = './data/wallpapers/train_aug/';
else
    data_address = './data/wallpapers/test';
    output_address = './data/wallpapers/test_aug/';
end

move = [];
angle = [];
scale = [];

for i = 1:17
    group = imageDatastore(fullfile(data_address,groups(i)),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
    temp = 0;
    for j = 1:1000
        for k = 1:5
            % Opens the image
            I = readimage(group,j);
            % rescales the image
            r1 = 1.0 + (0.5).*rand(1,1);
            scale = [scale r1];
            I1 = imresize(I, r1);
            % Rotates the image
            r2 = 0 + (360-0).*rand(1,1);
            angle = [angle r2];
            I2 = imrotate(I1, r2, 'crop');
            % translates the image in x and y direction
            r3 = 1 + (40-1).*rand(1,1);
            move = [move r3];
            I3 = imtranslate(I2, [r3,r3]);
            s = size(I);
            x = s(1)/3;
            target = 128-1;
            r = [x,x,target,target]; 
            J = imcrop(I3,r); 
            label = char(groups(i));
            temp = temp+1;
            temp2 = num2str(temp);
            name = strcat(label, '_', temp2);
            path = strcat(output_address, label, '/', name, '.png');
            if ~exist(strcat(output_address, label), 'dir')
                mkdir(strcat(output_address, label))
            end    
            imwrite(J, path);
        end
    end
end
figure
title('Translation - Train')
histogram(move,'BinMethod','integers');
figure
title('Rotation - Train')
histogram(angle,'BinMethod','integers');
figure
title('Scaling - Train')
histogram(scale,'BinWidth',0.1);

aug = 1;
end

