% ========================================================================
% File 3: run_hybrid_ml_mechanistic_cd3.m
% Purpose:
%   Implement a hybrid ML + mechanistic decision model for:
%       Order PCR now vs wait vs do not order
%
% Strategy:
%   1) Use ML submodels to estimate components needed for decision support:
%      - p(viral)
%      - p(change in management)
%      - risk of 72h return
%      - risk of unnecessary antibiotics
%   2) Apply a mechanistic / decision-science layer that computes expected
%      utility of each action using value-of-information and turnaround-time.
%   3) Generate tables, graphs, and plain-language interpretations.
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

cvp = cvpartition(height(T), 'Holdout', 0.25);
trainIdx = training(cvp);
testIdx = test(cvp);

Xtrain = X(trainIdx,:);
Xtest  = X(testIdx,:);
Ttrain = T(trainIdx,:);
Ttest  = T(testIdx,:);

%% -------------------------- SUBMODEL 1: p(viral) -----------------------
% Predict whether infection is viral-like
YviralTrain = Ttrain.true_infection_state == 1;
mdlViral = fitclinear(Xtrain, YviralTrain, 'Learner', 'logistic');
[~, scoreViral] = predict(mdlViral, Xtest);
pViral = scoreViral(:,2);

%% ------------------ SUBMODEL 2: p(change management) -------------------
YchangeTrain = Ttrain.will_change_management;
mdlChange = fitrensemble(Xtrain, YchangeTrain, 'Method','LSBoost', 'NumLearningCycles',150);
pChange = predict(mdlChange, Xtest);
pChange = max(0, min(1, pChange));

%% ---------------------- SUBMODEL 3: risk return ------------------------
YreturnTrain = Ttrain.return_72h;
mdlReturn = fitclinear(Xtrain, YreturnTrain, 'Learner','logistic');
[~, scoreReturn] = predict(mdlReturn, Xtest);
pReturn = scoreReturn(:,2);

%% --------------- SUBMODEL 4: unnecessary antibiotics risk --------------
YabxTrain = Ttrain.unnecessary_antibiotics;
mdlAbx = fitclinear(Xtrain, YabxTrain, 'Learner','logistic');
[~, scoreAbx] = predict(mdlAbx, Xtest);
pUnnecessaryAbx = scoreAbx(:,2);

%% --------------------- MECHANISTIC DECISION LAYER ----------------------
% Utilities are interpretable weighted combinations.
% The form can later be refined with clinician input.
TAT = Ttest.pcr_tat_hours;
season = Ttest.respiratory_season;
borderline = double((Ttest.spo2 >= 91 & Ttest.spo2 <= 95) | (Ttest.crp >= 15 & Ttest.crp <= 65) | Ttest.cxr_borderline==1);

% Value of information falls as TAT increases
VOI = pChange .* exp(-0.12*TAT);

U_order_now = ...
      1.60*VOI ...
    + 0.35*pViral ...
    + 0.20*season ...
    + 0.25*borderline ...
    - 0.22*pUnnecessaryAbx ...
    - 0.05*TAT;

U_wait = ...
      1.00*VOI ...
    + 0.28*borderline ...
    + 0.15*pReturn ...
    - 0.03*TAT;

U_no_order = ...
      0.90*(1-pChange) ...
    + 0.22*(1-pViral) ...
    + 0.15*(1-borderline) ...
    - 0.20*pReturn;

utilities = [U_order_now, U_wait, U_no_order];
[~, yHybridNum] = max(utilities, [], 2);
yHybrid = categorical(yHybridNum, [1 2 3], {'OrderNow','Wait','DoNotOrder'});

yTrue = categorical(Ttest.y_decision, [1 2 3], {'OrderNow','Wait','DoNotOrder'});
yRef  = categorical(Ttest.y_policy, [1 2 3], {'OrderNow','Wait','DoNotOrder'});

%% -------------------------- EVALUATION ---------------------------------
accuracy = mean(yHybrid == yTrue);
macroF1 = computeMacroF1(yTrue, yHybrid, categories(yTrue));
balAcc = computeBalancedAccuracy(yTrue, yHybrid, categories(yTrue));
decisionConsistency = mean(yHybrid == yRef);

unnecessaryOrderRate = mean(yHybrid=='OrderNow' & Ttest.p_change_mgmt_true < 0.25);
missedUsefulOrderRate = mean(yHybrid~='OrderNow' & Ttest.p_change_mgmt_true >= 0.50);

comparisonHybrid = table(string('HybridMLMechanistic'), accuracy, macroF1, balAcc, ...
    unnecessaryOrderRate, missedUsefulOrderRate, decisionConsistency, ...
    'VariableNames', {'Model','Accuracy','MacroF1','BalancedAccuracy', ...
                      'UnnecessaryOrderRate','MissedUsefulOrderRate','DecisionConsistencyProxy'});
writetable(comparisonHybrid, fullfile(tabDir,'hybrid_model_comparison.csv'));

%% -------------------- CASE-LEVEL EXPLANATION TABLE ---------------------
[~, utilRankIdx] = sort(max(utilities,[],2) - median(utilities,2), 'descend');
sel = utilRankIdx(1:min(12, numel(utilRankIdx)));

explainTable = table(Ttest.patient_id(sel), yHybrid(sel), U_order_now(sel), U_wait(sel), U_no_order(sel), ...
    pViral(sel), pChange(sel), pReturn(sel), pUnnecessaryAbx(sel), TAT(sel), ...
    'VariableNames', {'patient_id','RecommendedAction','U_OrderNow','U_Wait','U_DoNotOrder', ...
                      'pViral','pChangeMgmt','pReturn72h','pUnnecessaryAbx','TAT_hours'});
writetable(explainTable, fullfile(tabDir,'hybrid_case_explanations.csv'));

%% ------------------------------ FIGURES --------------------------------
figure('Color','w');
confusionchart(yTrue, yHybrid);
title('Hybrid ML + Mechanistic Model Confusion Matrix');
saveas(gcf, fullfile(figDir,'confmat_hybrid.png'));

figure('Color','w');
bar(categorical({'OrderNow','Wait','DoNotOrder'}), [mean(U_order_now), mean(U_wait), mean(U_no_order)]);
ylabel('Mean utility');
title('Average Hybrid Utility by Action');
saveas(gcf, fullfile(figDir,'hybrid_mean_utilities.png'));

figure('Color','w');
scatter(TAT, VOI, 18, 'filled');
xlabel('PCR turnaround time (hours)');
ylabel('Estimated value of information');
title('Value of Information vs Turnaround Time');
saveas(gcf, fullfile(figDir,'voi_vs_tat.png'));

figure('Color','w');
subplot(1,1,1);
scatter(pChange, U_order_now - U_no_order, 18, 'filled');
xlabel('Predicted probability PCR changes management');
ylabel('Utility difference: Order now - Do not order');
title('How Change-in-Management Drives Order Decisions');
saveas(gcf, fullfile(figDir,'utility_difference_vs_pchange.png'));

%% ------------ OPTIONAL COMBINED COMPARISON WITH PURE ML TABLE ----------
mlFile = fullfile(tabDir,'ml_model_comparison.csv');
if exist(mlFile, 'file')
    mlComp = readtable(mlFile);
    combined = [mlComp; comparisonHybrid];
    writetable(combined, fullfile(tabDir,'combined_ml_vs_hybrid_comparison.csv'));

    figure('Color','w');
    bar(categorical(combined.Model), combined.Accuracy);
    ylabel('Accuracy');
    title('Pure ML vs Hybrid Accuracy Comparison');
    saveas(gcf, fullfile(figDir,'combined_accuracy_comparison.png'));

    figure('Color','w');
    bar(categorical(combined.Model), [combined.UnnecessaryOrderRate combined.MissedUsefulOrderRate], 'grouped');
    ylabel('Rate');
    legend({'Unnecessary order rate','Missed useful order rate'}, 'Location','best');
    title('Pure ML vs Hybrid Decision-Relevant Error Comparison');
    saveas(gcf, fullfile(figDir,'combined_decision_error_comparison.png'));
end

%% ----------------------- AUTOMATED INTERPRETATION ----------------------
fprintf('\n=== HYBRID MODEL INTERPRETATION ===\n');
fprintf('Hybrid model accuracy = %.3f, Macro-F1 = %.3f, Balanced Accuracy = %.3f.\n', accuracy, macroF1, balAcc);
fprintf('Hybrid unnecessary order rate = %.3f.\n', unnecessaryOrderRate);
fprintf('Hybrid missed useful order rate = %.3f.\n', missedUsefulOrderRate);
fprintf('Hybrid decision consistency proxy = %.3f.\n', decisionConsistency);

if exist(mlFile, 'file')
    bestML = mlComp(find(mlComp.Accuracy == max(mlComp.Accuracy),1),:);
    fprintf('Best pure ML accuracy = %.3f (%s).\n', bestML.Accuracy, bestML.Model{1});
    if accuracy >= bestML.Accuracy - 0.03
        fprintf('Interpretation: the hybrid model achieved comparable predictive performance while making decisions using explicit value-of-information and turnaround-time logic.\n');
    else
        fprintf('Interpretation: the hybrid model traded some predictive accuracy for a more explicit decision-support structure.\n');
    end
end
fprintf('This model is not only predicting a label; it is comparing the expected utility of ordering now, waiting, or not ordering.\n');

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
