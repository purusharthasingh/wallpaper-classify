function result = alexAugment(input_address, output_address,Symmetry_Groups)
%Takes the input data from the augmented data and converts it to match the
%first layer input of the Alexnet
%   Creates new directory if one does not exist for Alex data
if ~exist(output_address, 'dir')
    mkdir(output_address)
end

for i = 1:17
    group = imageDatastore(fullfile(input_address,Symmetry_Groups(i)),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
    for j = 1:5000
        % For each image, changes size to 227 by 227 and adds color
        I = readimage(group,j);
        I = imresize(I, [227,227]);
        I = ind2rgb(I, hsv(256));
        temp2 = num2str(j);
        label = char(Symmetry_Groups(i));
        name = strcat(label, '_', temp2);
        path = strcat(output_address, label, '/', name, '.png');
        if ~exist(strcat(output_address, label), 'dir')
            mkdir(strcat(output_address, label))
        end    
        imwrite(I, path);
    end
    result = 1;
end
end

