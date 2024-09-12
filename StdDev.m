function [media, desviacionEstandar,numImag] = StdDev(PI, percentages)

    % Valores de retorno:
    %   media: La media de los valores en el vector
    %   desviacionEstandar: La desviación estándar de los valores en el vector

    % Calcula la media del vector
    media = mean(PI);
    vector2=percentages.*6200;
    % Calcula la desviación estándar del vector
    desviacionEstandar = std(PI);

    numImag = sum(PI.*vector2);
   
end
