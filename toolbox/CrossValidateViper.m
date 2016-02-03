function [ ds, runs ] = CrossValidateViper(ds, learn_algs, idxa, idxb, params,runs,runs_mode)

for c=1:params.numFolds
    clc; fprintf('VIPeR run %d of %d\n',c,params.numFolds);
    if strcmp(runs_mode,'new')==1        
        perm = randperm(params.N);% draw random permuation        
        idxtrain = perm(1:params.N/2);% split in equal-sized train and test sets
        idxtest  = perm(params.N/2+1:end);
        idxtrain_neg=randperm(params.N/2);
        runs(c).perm = perm;
        runs(c).idxtrain = idxtrain;
        runs(c).idxtest = idxtest;
        runs(c).idxtrain_neg=idxtrain_neg;
    elseif strcmp(runs_mode,'repeat')==1
        idxtrain=runs(c).idxtrain;
        idxtest=runs(c).idxtest;
        idxtrain_neg=runs(c).idxtrain_neg;
    end
    % train on first half
    for aC=1:length(learn_algs)
        cHandle = learn_algs{aC};
        fprintf('    training %s ',upper(cHandle.type));
        X=cHandle.p.ftr(1:cHandle.p.numCoeffs,:);
        s = learnPairwise(cHandle,X,[idxa(idxtrain) idxa(idxtrain)],[idxb(idxtrain) idxb(idxtrain(idxtrain_neg))],logical([ones(1,size(idxtrain,2)) zeros(1,size(idxtrain,2))]));
        if ~isempty(fieldnames(s))
            fprintf('... done in %.4fs\n',s.t);
            ds(c).(cHandle.type) = s;
        else
            fprintf('... not available\n');
        end
    end
       
    % test on second half
    names = fieldnames(ds(c));
    for nameCounter=1:length(names)   
        cHandle = ds(c).(names{nameCounter}).learnAlgo;
        fprintf('    evaluating %s ',upper(names{nameCounter}));           
        X=cHandle.p.ftr(1:cHandle.p.numCoeffs,:);
        ds(c).(names{nameCounter}).cmc = calcMCMC(ds(c).(names{nameCounter}).M, X,idxa,idxb,idxtest);
        fprintf('... done \n');
    end
end

end