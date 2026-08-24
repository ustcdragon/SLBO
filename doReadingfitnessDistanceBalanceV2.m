%%%%%Student–Learning-Based Optimizer
%%%%%author:Qixiangbo and MaXu
function divDistances = doReadingfitnessDistanceBalanceV2( population, fitness )
[~, bestIndex] = min(fitness);
best = population(bestIndex, :);
[populationSize, dimension] = size(population);
distances = zeros(1, populationSize);
normFitness = zeros(1, populationSize);
normDistances = zeros(1, populationSize);
divDistances = zeros(1, populationSize);
for i = 1 : populationSize
    value = 0;

    value = pdist2(best,population(i,:),'cosine');%%1.3856583537596269e-52
    distances(i) = value;
end
minFitness = min(fitness); maxMinFitness = max(fitness) - minFitness;
minDistance = min(distances); maxMinDistance = max(distances) - minDistance;
for i = 1 : populationSize
    normFitness(i) = 1 - ((fitness(i) - minFitness) / maxMinFitness);
    normDistances(i) = (distances(i) - minDistance) / maxMinDistance;
    divDistances(i) = normFitness(i) + normDistances(i);
end
[~, index] = max(divDistances);
end

