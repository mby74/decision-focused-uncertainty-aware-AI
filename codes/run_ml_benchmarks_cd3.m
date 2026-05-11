% ========================================================================
% File 2: run_ml_benchmarks_cd3.m
% Purpose:
%   Train benchmark ML methods for the 3-way decision problem and compare
%   predictive + decision-relevant performance.
%
% Methods:
%   CART, Logistic Regression (multinomial ECOC surrogate), Neural Network,
%   Random Forest, SVM, XGBoost-like boosted trees (fitcensemble)
%
% Output files:
%   - tables/ml_model_comparison.csv
%   - tables/ml_per_class_metrics.csv
%   - figures/*.png
%
% Requirements:
%   Statistics and Machine Learning Toolbox
%   Deep Learning Toolbox (for neural net section; script gracefully skips
%   if unavailable)
% ========================================================================

clear; clc; close all;

% ================= FIGURE DEFAULTS =================
set(groot, 'defaultAxesFontSize', 16);
set(groot, 'defaultTextFontSize', 16);

set(groot, 'defaultAxesBox', 'on');
set(groot, 'defaultAxesXGrid', 'on');
set(groot, 'defaultAxesYGrid', 'on');

set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultAxesLineWidth', 1.2);


rng(42);

figDir = 'figures';
tabDir = 'tables';
if ~exist(figDir,'dir'); mkdir(figDir); end
if ~exist(tabDir,'dir'); mkdir(tabDir); end

% Load data
if exist('synthetic_cd3_dataset.mat','file')
    load('synthetic_cd3_dataset.mat','T');
else
    T = readtable('synthetic_cd3_dataset.csv');
end

%% -------------------------- PREPARE FEATURES ---------------------------
featureNames = { ...
    'age_months','sex_male','insurance_public','language_non_english', ...
    'deprivation_index','respiratory_season','weekend','night_shift', ...
    'hours_since_presentation','fever_c','spo2','tachypnea','retractions', ...
    'wheezing','crackles','poor_feeding','wbc','crp','procalcitonin', ...
    'cxr_borderline','prior_antibiotics','pcr_tat_hours'};

X = T{:, featureNames};
y = categorical(T.y_decision, [1 2 3], {'OrderNow','Wait','DoNotOrder'});

cvp = cvpartition(y, 'Holdout', 0.25);
Xtrain = X(training(cvp),:);
Xtest  = X(test(cvp),:);
ytrain = y(training(cvp),:);
ytest  = y(test(cvp),:);

% Standardize numeric variables for models that benefit from scaling
mu = mean(Xtrain,1);
sigma = std(Xtrain,[],1);
sigma(sigma==0) = 1;
XtrainZ = (Xtrain - mu)./sigma;
XtestZ  = (Xtest - mu)./sigma;

classNames = categories(y);
nClasses = numel(classNames);

results = struct();
modelNames = {};

%% ----------------------- METRIC HELPER FUNCTIONS -----------------------
calcMetrics = @(yTrue, yPred) struct( ...
    'Accuracy', mean(yTrue == yPred), ...
    'MacroF1', computeMacroF1(yTrue, yPred, classNames), ...
    'BalancedAccuracy', computeBalancedAccuracy(yTrue, yPred, classNames));

%% ------------------------------- CART ----------------------------------
try
    mdl = fitctree(Xtrain, ytrain, 'MinLeafSize', 20);
    ypred = predict(mdl, Xtest);
    [accStruct, decisionStruct] = evaluateDecisionModel(ytest, ypred, T(test(cvp),:));
    results.CART = mergeStructs(calcMetrics(ytest, ypred), accStruct, decisionStruct);
    results.CART.Model = mdl;
    modelNames{end+1} = 'CART';
    plotConfMat(ytest, ypred, 'CART Confusion Matrix', fullfile(figDir,'confmat_cart.png'));
catch ME
    warning('CART failed: %s', ME.message);
end

%% ------------------------ LOGISTIC REGRESSION --------------------------
try
    t = templateLinear('Learner','logistic','Regularization','ridge');
    mdl = fitcecoc(XtrainZ, ytrain, 'Learners', t, 'Coding', 'onevsall');
    ypred = predict(mdl, XtestZ);
    [accStruct, decisionStruct] = evaluateDecisionModel(ytest, ypred, T(test(cvp),:));
    results.Logistic = mergeStructs(calcMetrics(ytest, ypred), accStruct, decisionStruct);
    results.Logistic.Model = mdl;
    modelNames{end+1} = 'LogisticRegression';
    plotConfMat(ytest, ypred, 'Logistic Regression Confusion Matrix', fullfile(figDir,'confmat_logistic.png'));
catch ME
    warning('Logistic Regression failed: %s', ME.message);
end

%% ------------------------------- SVM -----------------------------------
try
    t = templateSVM('KernelFunction','rbf','Standardize',true);
    mdl = fitcecoc(Xtrain, ytrain, 'Learners', t, 'Coding', 'onevsall');
    ypred = predict(mdl, Xtest);
    [accStruct, decisionStruct] = evaluateDecisionModel(ytest, ypred, T(test(cvp),:));
    results.SVM = mergeStructs(calcMetrics(ytest, ypred), accStruct, decisionStruct);
    results.SVM.Model = mdl;
    modelNames{end+1} = 'SVM';
    plotConfMat(ytest, ypred, 'SVM Confusion Matrix', fullfile(figDir,'confmat_svm.png'));
catch ME
    warning('SVM failed: %s', ME.message);
end

%% -------------------------- RANDOM FOREST ------------------------------
try
    mdl = TreeBagger(250, Xtrain, ytrain, 'Method','classification', 'MinLeafSize',8, 'OOBPrediction','on');
    ypred = categorical(predict(mdl, Xtest), classNames, classNames);
    [accStruct, decisionStruct] = evaluateDecisionModel(ytest, ypred, T(test(cvp),:));
    results.RandomForest = mergeStructs(calcMetrics(ytest, ypred), accStruct, decisionStruct);
    results.RandomForest.Model = mdl;
    modelNames{end+1} = 'RandomForest';
    plotConfMat(ytest, ypred, 'Random Forest Confusion Matrix', fullfile(figDir,'confmat_rf.png'));
catch ME
    warning('Random Forest failed: %s', ME.message);
end

%% --------------------------- BOOSTED TREES -----------------------------
try
    t = templateTree('MaxNumSplits', 20, 'MinLeafSize', 10);
    mdl = fitcensemble(Xtrain, ytrain, 'Method','AdaBoostM2', 'Learners', t, 'NumLearningCycles', 200);
    ypred = predict(mdl, Xtest);
    [accStruct, decisionStruct] = evaluateDecisionModel(ytest, ypred, T(test(cvp),:));
    results.BoostedTrees = mergeStructs(calcMetrics(ytest, ypred), accStruct, decisionStruct);
    results.BoostedTrees.Model = mdl;
    modelNames{end+1} = 'BoostedTrees';
    plotConfMat(ytest, ypred, 'Boosted Trees Confusion Matrix', fullfile(figDir,'confmat_boosted.png'));
catch ME
    warning('Boosted Trees failed: %s', ME.message);
end

%% ---------------------------- NEURAL NET -------------------------------
try
    dummyTrain = dummyvar(grp2idx(ytrain));
    net = patternnet([20 10]);
    net.trainParam.showWindow = false;
    net = train(net, XtrainZ', dummyTrain');
    scores = net(XtestZ')';
    [~, idx] = max(scores, [], 2);
    ypred = categorical(idx, 1:3, classNames);
    [accStruct, decisionStruct] = evaluateDecisionModel(ytest, ypred, T(test(cvp),:));
    results.NeuralNet = mergeStructs(calcMetrics(ytest, ypred), accStruct, decisionStruct);
    results.NeuralNet.Model = net;
    modelNames{end+1} = 'NeuralNet';
    plotConfMat(ytest, ypred, 'Neural Network Confusion Matrix', fullfile(figDir,'confmat_nn.png'));
catch ME
    warning('Neural Net failed or toolbox unavailable: %s', ME.message);
end

%% ------------------------- COMPILE COMPARISON --------------------------
fields = fieldnames(results);
comparison = table('Size',[0 7], ...
    'VariableTypes',{'string','double','double','double','double','double','double'}, ...
    'VariableNames',{'Model','Accuracy','MacroF1','BalancedAccuracy', ...
                     'UnnecessaryOrderRate','MissedUsefulOrderRate','DecisionConsistencyProxy'});

for i = 1:numel(fields)
    f = fields{i};
    comparison = [comparison; {string(f), results.(f).Accuracy, results.(f).MacroF1, ...
        results.(f).BalancedAccuracy, results.(f).UnnecessaryOrderRate, ...
        results.(f).MissedUsefulOrderRate, results.(f).DecisionConsistencyProxy}]; %#ok<AGROW>
end

comparison = sortrows(comparison, 'Accuracy', 'descend');
writetable(comparison, fullfile(tabDir,'ml_model_comparison.csv'));

figure('Color','w');
bar(categorical(comparison.Model), comparison.Accuracy);
ylabel('Accuracy');
title('ML Model Accuracy Comparison');
saveas(gcf, fullfile(figDir,'ml_accuracy_comparison.png'));

figure('Color','w');
bar(categorical(comparison.Model), [comparison.UnnecessaryOrderRate comparison.MissedUsefulOrderRate], 'grouped');
ylabel('Rate');
legend({'Unnecessary order rate','Missed useful order rate'}, 'Location','best');
title('Decision-Relevant Error Comparison');
saveas(gcf, fullfile(figDir,'ml_decision_error_comparison.png'));

%% ---------------------- FEATURE IMPORTANCE (TREE) ----------------------
if isfield(results,'RandomForest')
    try
        imp = results.RandomForest.Model.OOBPermutedPredictorDeltaError;
        impTable = table(string(featureNames(:)), imp(:), 'VariableNames', {'Feature','Importance'});
        impTable = sortrows(impTable, 'Importance', 'descend');
        writetable(impTable, fullfile(tabDir,'rf_feature_importance.csv'));

        figure('Color','w');
        barh(impTable.Importance(1:min(10,height(impTable))));
        set(gca,'YDir','reverse', 'YTick',1:min(10,height(impTable)), ...
            'YTickLabel', impTable.Feature(1:min(10,height(impTable))));
        xlabel('Importance');
        title('Top Random Forest Features');
        saveas(gcf, fullfile(figDir,'rf_top_features.png'));
    catch
    end
end

%% ------------------------- AUTOMATED REPORT ----------------------------
[~, bestIdx] = max(comparison.Accuracy);
bestModel = comparison.Model(bestIdx);

fprintf('\n=== PURE ML BENCHMARK INTERPRETATION ===\n');
fprintf('Best predictive model by accuracy: %s (Accuracy = %.3f, Macro-F1 = %.3f).\n', ...
    bestModel, comparison.Accuracy(bestIdx), comparison.MacroF1(bestIdx));

[~, bestDecisionIdx] = min(comparison.MissedUsefulOrderRate + comparison.UnnecessaryOrderRate);
fprintf('Best decision tradeoff among pure ML models: %s.\n', comparison.Model(bestDecisionIdx));
fprintf('This analysis shows whether stronger predictive performance also translates into fewer unnecessary PCR orders and fewer missed potentially useful tests.\n');

%% ------------------------- LOCAL FUNCTIONS -----------------------------
function macroF1 = computeMacroF1(yTrue, yPred, classNames)
    f1 = zeros(numel(classNames),1);
    for k = 1:numel(classNames)
        c = categorical(classNames(k), classNames, classNames);
        tp = sum(yTrue==c & yPred==c);
        fp = sum(yTrue~=c & yPred==c);
        fn = sum(yTrue==c & yPred~=c);
        precision = tp / max(tp+fp, 1);
        recall = tp / max(tp+fn, 1);
        f1(k) = 2*precision*recall / max(precision+recall, eps);
    end
    macroF1 = mean(f1);
end

function balAcc = computeBalancedAccuracy(yTrue, yPred, classNames)
    recalls = zeros(numel(classNames),1);
    for k = 1:numel(classNames)
        c = categorical(classNames(k), classNames, classNames);
        tp = sum(yTrue==c & yPred==c);
        fn = sum(yTrue==c & yPred~=c);
        recalls(k) = tp / max(tp+fn,1);
    end
    balAcc = mean(recalls);
end

function [accStruct, decisionStruct] = evaluateDecisionModel(yTrue, yPred, Ttest)
    % Unnecessary order: predicted order now when change-management probability is low
    predOrder = yPred == 'OrderNow';
    predWait  = yPred == 'Wait';
    useful = Ttest.p_change_mgmt_true >= 0.50;
    lowValue = Ttest.p_change_mgmt_true < 0.25;

    unnecessaryOrderRate = mean(predOrder & lowValue);
    missedUsefulOrderRate = mean(~predOrder & useful);
    % Proxy consistency: reward agreement with latent reference policy
    reference = categorical(Ttest.y_policy, [1 2 3], {'OrderNow','Wait','DoNotOrder'});
    decisionConsistency = mean(yPred == reference);

    accStruct = struct();
    decisionStruct = struct('UnnecessaryOrderRate', unnecessaryOrderRate, ...
                            'MissedUsefulOrderRate', missedUsefulOrderRate, ...
                            'DecisionConsistencyProxy', decisionConsistency);
end

function out = mergeStructs(varargin)
    out = struct();
    for i = 1:nargin
        s = varargin{i};
        f = fieldnames(s);
        for j = 1:numel(f)
            out.(f{j}) = s.(f{j});
        end
    end
end

function plotConfMat(yTrue, yPred, plotTitle, fileName)
    figure('Color','w');
    confusionchart(yTrue, yPred);
    title(plotTitle);
    saveas(gcf, fileName);
end


