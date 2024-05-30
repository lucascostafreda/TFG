function decimal = binaryVectorToDecimal(binaryVector, mode)
    if nargin < 2
        mode = 'LSB'; % Default to least significant bit if mode is not specified
    end
    n = length(binaryVector);
    if strcmpi(mode, 'MSB')
        powersOfTwo = 2 .^ (n-1:-1:0);
    else % Default to 'LSB' mode
        powersOfTwo = 2 .^ (0:n-1);
    end
    decimal = sum(binaryVector .* powersOfTwo);
end
