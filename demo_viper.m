%% Person Re-Identification on the VIPeR dataset
clc; clear all; close all;
run('./toolbox/init.m');
runs_mode='repeat'; % repeat the same training/testing splits
% runs_mode='new';  % new training/testing splits

%% Get Features
GetFtr_MDF;

%% Set up parameters
params.numCoeffs = 34; % dimensionality reduction by PCA to 34D from 22154D in KISSME
params.N = 632; % number of image pairs, 316 to train 316 to test
params.numFolds = 100; % number of random train/test splits
params.pmetric = 0;
params.ftr=ux;
params2=params;
params2.numCoeffs=44;% ie. d2=44
params2.ftr=ux2;

%% Cross-validate over a number of runs
pair_metric_learn_algs = {...
    LearnAlgoMDF_KISS(params2) ...
    LearnAlgoKISSME(params), ...
    LearnAlgoMLEuclidean(params) ...
    };
if strcmp(runs_mode,'repeat')==1
    load('./data/viper_runs100.mat'); % load the same splits used in our paper
else
    runs=struct();
end
[ ds,runs] = CrossValidateViper(struct(),pair_metric_learn_algs,idxa,idxb,params,runs,runs_mode);

%% Plot Cumulative Matching Characteristic (CMC) Curves
names = fieldnames(ds);
for nameCounter=1:length(names)
    s = [ds.(names{nameCounter})];
    ms.(names{nameCounter}).cmc = cat(1,s.cmc)./(params.N/2);
    ms.(names{nameCounter}).roccolor = s(1).roccolor;
end

h = figure;
names = fieldnames(ms);
CMR_Comp=zeros(length(names),316);
for nameCounter=1:length(names)
    avgcmc=median(ms.(names{nameCounter}).cmc,1);
    CMR_Comp(nameCounter,:)=avgcmc;
    hold on; plot(avgcmc,'LineWidth',2, ...
        'Color',ms.(names{nameCounter}).roccolor);
    fprintf([upper(names{nameCounter}) ':' ...
             'CMC Rank-1: '  num2str(sprintf('%.2f',100*avgcmc(1))) '%% '...
             'CMC Rank-25: ' num2str(sprintf('%.2f',100*avgcmc(25))) '%%\n']);
end

title('Cumulative Matching Characteristic (CMC) Curves - VIPeR dataset');
box('on');
set(gca,'XTick',[0 10 20 30 40 50 100 150 200 250 300 350]);
ylabel('Matches');xlabel('Rank');
ylim([0 1]);
hold off;grid on;
legend(upper(names),'Location','SouthEast');
saveas(h,'./res/all_viper');

