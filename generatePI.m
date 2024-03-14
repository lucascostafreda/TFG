function PI_combinations = generatePI(r, total_available_RBs)
    % Define los pasos y la desviación permitida hacia abajo
    step = 0.1;
    deviation_allowed = total_available_RBs * 0.05; % 5% de los RBs disponibles
    lower_bound = total_available_RBs - deviation_allowed; % Límite inferior permitido
    
    % Inicializa la matriz para almacenar combinaciones válidas de PI
    PI_combinations = [];
    
    % Genera todas las combinaciones posibles de PI con pasos de 0.1
    PI_steps = 0:step:1;
    [grid{1:numel(r)}] = ndgrid(PI_steps);
    combinations = reshape(cat(numel(r)+1, grid{:}), [], numel(r));
    
    % Filtra las combinaciones que cumplen con la condición de recursos
    for i = 1:size(combinations, 1)
        PI = combinations(i, :);
        total_RB_usage = sum(PI .* r);
        if total_RB_usage <= total_available_RBs && total_RB_usage >= lower_bound
            PI_combinations = [PI_combinations; PI];
        end
    end
    
    % Ordena las combinaciones por eficiencia
    % (Este ejemplo ordena basado en la suma de PI, que puedes cambiar según tu criterio de "eficiencia")
    [~, idx] = sort(sum(PI_combinations, 2), 'descend');
    PI_combinations = PI_combinations(idx, :);
end