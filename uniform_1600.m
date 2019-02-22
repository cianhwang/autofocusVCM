Model_parameters = zeros(33,5);
Focused = double(rgb2gray(imread('data_6_1600_1650_7848_s00_00000.jpg'  )))/256; %%%%%%%%%%%%%%%%%%%
for i = 500:50:2100
    i
    MXX = zeros(1,3);
    Names = dir(['data_6_1600_',num2str(i),'_*.jpg']);   %%%%%%%%%%%%%%%%%
    names = {Names.name};
    Defocused = double(rgb2gray(imread(names{1})))/256;
    
    if i == 500
        for j = 125:1:145
            h = fspecial('disk',j);
            Modeled = conv2(Focused,h);
            Modeled = Modeled(floor(size(Modeled,1)/2)-600:floor(size(Modeled,1)/2)+600,...
                floor(size(Modeled,2)/2)-1000:floor(size(Modeled,1)/2)+1000);
            for k = 0.89:0.01:0.93
                Modeled_resize = imresize(Modeled,k);
                c = normxcorr2(Modeled_resize,Defocused);
                if max(c(:)) > MXX(1)
                    %MXX(1) = max(c(:))
                    MXX(2) = j;
                    MXX(3) = k;
                    MXX(1) = max(c(:));
                end
            end
        end
    elseif i == 1650                                      %%%%%%%%%%%%%%%%%%%%5
        MXX(2) = 0;
        MXX(3) = 1;
        MXX(1) = 1;
    elseif i<1650                                        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        for j = max(Model_parameters((i-450)/50-1,3)-10,1):1:Model_parameters((i-450)/50-1,3)
            h = fspecial('disk',j);
            Modeled = conv2(Focused,h);
            Modeled = Modeled(floor(size(Modeled,1)/2)-600:floor(size(Modeled,1)/2)+600,...
                floor(size(Modeled,2)/2)-1000:floor(size(Modeled,1)/2)+1000);
            for k = Model_parameters((i-450)/50-1,4)-0.01:0.01:Model_parameters((i-450)/50-1,4)+0.02
                Modeled_resize = imresize(Modeled,k);
                c = normxcorr2(Modeled_resize,Defocused);
                if max(c(:)) > MXX(1)
                    %MXX(1) = max(c(:))
                    MXX(2) = j;
                    MXX(3) = k;
                    MXX(1) = max(c(:));
                end
            end
        end  
    else
        for j = max(Model_parameters((i-450)/50-1,3),1):1:Model_parameters((i-450)/50-1,3)+10
            h = fspecial('disk',j);
            Modeled = conv2(Focused,h);
            Modeled = Modeled(floor(size(Modeled,1)/2)-600:floor(size(Modeled,1)/2)+600,...
                floor(size(Modeled,2)/2)-1000:floor(size(Modeled,1)/2)+1000);
            for k = Model_parameters((i-450)/50-1,4)-0.01:0.01:Model_parameters((i-450)/50-1,4)+0.02
                Modeled_resize = imresize(Modeled,k);
                c = normxcorr2(Modeled_resize,Defocused);
                if max(c(:)) > MXX(1)
                    %MXX(1) = max(c(:))
                    MXX(2) = j;
                    MXX(3) = k;
                    MXX(1) = max(c(:));
                end
            end
        end
    end
    
    

    Model_parameters((i-450)/50,1) = 1650;               %%%%%%%%%%%%%%%%%%%%%
    Model_parameters((i-450)/50,2) = i;
    Model_parameters((i-450)/50,3) = MXX(2);
    Model_parameters((i-450)/50,4) = MXX(3);
    Model_parameters((i-450)/50,5) = MXX(1);
    save('Model_parameters_1650.mat','Model_parameters');  %%%%%%%%%%%%%%%%%%%
end

save('Model_parameters_1650.mat','Model_parameters');          %%%%%%%%%%%%%%%
