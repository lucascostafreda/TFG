function [output] = newnorm(input)
    for i = 1:size(input,2)/24
        temp1 = input(:,(i-1)*24+1:i*24);
        temp2 = rescale(temp1(:));
        output(:,(i-1)*24+1:i*24) = reshape(temp2,size(temp1));
    end
end