% ========================================================================
% File: compare_patient_outcomes_cd3_revised.m
% Purpose:
%   Compare estimated patient outcomes under:
%       (1) Pure ML decision policies
%       (2) Tuned Hybrid ML + mechanistic decision policy
%
% Main improvements over prior version:
%   - Adds explicit ICU-risk submodel to hybrid decision logic
%   - Tunes hybrid utility weights and threshold on validation data
%   - Selects hybrid settings that minimize estimated composite harm
%
% Binary action:
%   1 = Order PCR now
%   0 = Do not order now
%
% Output files:
%   - tables/policy_outcome_comparison_revised.csv
%   - tables/hybrid_tuning_results.csv
%   - figures/policy_outcome_comparison_revised.png
%   - figures/policy_composite_harm_revised.png
%   - figures/policy_pcr_order_rate_revised.png
% ========================================================================

clear; clc; close all;
rng(42);

%% --------------------------- STYLE DEFAULTS ----------------------------
set(groot, 'defaultAxesFontSize', 16);
set(groot, 'defaultTextFontSize', 16);
set(groot, 'defaultAxesBox', 'on');
set(groot, 'defaultAxesXGrid', 'on');
set(groot, 'defaultAxesYGrid', 'on');
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultAxesLineWidth', 1.2);

%% ----------------------------- PATH SETUP ------------------------------
baseDir = pwd;
figDir = fullfile(baseDir, 'figures');
tabDir = fullfile(baseDir, 'tables');

if ~exist(figDir, 'dir'); mkdir(figDir); end
if ~exist(tabDir, 'dir'); mkdir(tabDir); end

%% ----------------------------- LOAD DATA -------------------------------
if exist('synthetic_cd3_dataset.mat', 'file')
    load('synthetic_cd3_dataset.mat', 'T');
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

% Binary observed action:
%   1 = Order PCR now
%   0 = Do not order now
y_obs_binary = double(T.y_decision == 1);

% Binary reference policy:
%   1 = Order PCR now
%   0 = Wait or do not order
y_ref_binary = double(T.y_policy == 1);

%% ---------------------- TRAIN / VALID / TEST SPLIT ---------------------
n = height(T);
cvOuter = cvpartition(n, 'Holdout', 0.25);
trainValidIdx = training(cvOuter);
testIdx = test(cvOuter);

T_trainValid = T(trainValidIdx,:);
T_test = T(testIdx,:);

X_trainValid = X(trainValidIdx,:);
X_test = X(testIdx,:);

y_trainValid = y_obs_binary(trainValidIdx);
y_test = y_obs_binary(testIdx);
y_ref_test = y_ref_binary(testIdx);

% Split trainValid into train / validation
cvInner = cvpartition(height(T_trainValid), 'Holdout', 0.25);
trainIdx_local = training(cvInner);
validIdx_local = test(cvInner);

T_train = T_trainValid(trainIdx_local,:);
T_valid = T_trainValid(validIdx_local,:);

X_train = X_trainValid(trainIdx_local,:);
X_valid = X_trainValid(validIdx_local,:);

y_train = y_trainValid(trainIdx_local);
y_valid = y_trainValid(validIdx_local);

%% -------------------- STANDARDIZATION HELPERS --------------------------
muTrain = mean(X_train,1);
sigmaTrain = std(X_train,[],1);
sigmaTrain(sigmaTrain == 0) = 1;

X_trainZ = (X_train - muTrain) ./ sigmaTrain;
X_validZ = (X_valid - muTrain) ./ sigmaTrain;
X_testZ  = (X_test  - muTrain) ./ sigmaTrain;

%% ======================================================================
% PART 1. PURE ML DECISION POLICIES
% ======================================================================
fprintf('\nTraining pure ML decision policies...\n');

policyNames = {};
policyDecisions_test = {};

% ------------------------- Logistic Regression --------------------------
try
    mdlLog = fitclinear(X_trainValid, y_trainValid, 'Learner', 'logistic');
    [~, scoreLog] = predict(mdlLog, X_test);
    predLog = double(scoreLog(:,2) >= 0.5);
    policyNames{end+1} = 'PureML_Logistic';
    policyDecisions_test{end+1} = predLog(:);
catch ME
    warning('Logistic model failed: %s', ME.message);
end

% -------------------------------- SVM ----------------------------------
try
    muTV = mean(X_trainValid,1);
    sigmaTV = std(X_trainValid,[],1);
    sigmaTV(sigmaTV == 0) = 1;
    X_trainValidZ = (X_trainValid - muTV) ./ sigmaTV;
    X_testZ_tv = (X_test - muTV) ./ sigmaTV;

    mdlSVM = fitcsvm(X_trainValidZ, y_trainValid, ...
        'KernelFunction', 'rbf', ...
        'Standardize', true, ...
        'ClassNames', [0 1]);
    mdlSVM = fitPosterior(mdlSVM);
    [~, scoreSVM] = predict(mdlSVM, X_testZ_tv);
    predSVM = double(scoreSVM(:,2) >= 0.5);
    policyNames{end+1} = 'PureML_SVM';
    policyDecisions_test{end+1} = predSVM(:);
catch ME
    warning('SVM model failed: %s', ME.message);
end

% ---------------------------- CART / Tree -------------------------------
try
    mdlTree = fitctree(X_trainValid, y_trainValid, 'MinLeafSize', 20);
    predTree = predict(mdlTree, X_test);
    policyNames{end+1} = 'PureML_CART';
    policyDecisions_test{end+1} = double(predTree(:));
catch ME
    warning('CART model failed: %s', ME.message);
end

% -------------------------- Random Forest -------------------------------
try
    mdlRF = TreeBagger(250, X_trainValid, y_trainValid, ...
        'Method','classification', 'MinLeafSize',8, 'OOBPrediction','on');
    predRFcell = predict(mdlRF, X_test);
    predRF = str2double(predRFcell);
    policyNames{end+1} = 'PureML_RandomForest';
    policyDecisions_test{end+1} = double(predRF(:));
catch ME
    warning('Random forest failed: %s', ME.message);
end

% --------------------------- Boosted Trees ------------------------------
try
    tBoost = templateTree('MaxNumSplits', 20, 'MinLeafSize', 10);
    mdlBoost = fitcensemble(X_trainValid, y_trainValid, ...
        'Method','AdaBoostM1', 'Learners', tBoost, ...
        'NumLearningCycles', 200);
    predBoost = predict(mdlBoost, X_test);
    policyNames{end+1} = 'PureML_BoostedTrees';
    policyDecisions_test{end+1} = double(predBoost(:));
catch ME
    warning('Boosted trees failed: %s', ME.message);
end

%% ======================================================================
% PART 2. OUTCOME MODELS FOR OFF-POLICY EVALUATION
% ======================================================================
%
% We train outcome models on TRAIN data only:
%   P(return_72h | x, action)
%   P(icu_transfer | x, action)
%   P(unnecessary_antibiotics | x, action)
%
% These are later used to evaluate candidate hybrid policies on VALID and
% final policies on TEST.
% ======================================================================
fprintf('Training outcome models for off-policy evaluation...\n');

A_train = y_train(:);
Xout_train = [X_train, A_train];

muOut = mean(Xout_train,1);
sigmaOut = std(Xout_train,[],1);
sigmaOut(sigmaOut == 0) = 1;
Xout_trainZ = (Xout_train - muOut) ./ sigmaOut;

mdlOutcomeReturn = fitclinear(Xout_trainZ, T_train.return_72h, 'Learner', 'logistic');
mdlOutcomeICU    = fitclinear(Xout_trainZ, T_train.icu_transfer, 'Learner', 'logistic');
mdlOutcomeAbx    = fitclinear(Xout_trainZ, T_train.unnecessary_antibiotics, 'Learner', 'logistic');

%% ======================================================================
% PART 3. HYBRID SUBMODELS TRAINED ON TRAIN DATA
% ======================================================================
fprintf('Training hybrid submodels...\n');

% pViral
Yviral_train = double(T_train.true_infection_state == 1);
mdlViral = fitclinear(X_trainZ, Yviral_train, 'Learner', 'logistic');

% pChange
mdlChange = fitrensemble(X_train, T_train.will_change_management, ...
    'Method', 'LSBoost', 'NumLearningCycles', 150);

% pReturn
mdlReturn = fitclinear(X_trainZ, T_train.return_72h, 'Learner', 'logistic');

% pICU
mdlICU = fitclinear(X_trainZ, T_train.icu_transfer, 'Learner', 'logistic');

% pUnnecessaryAbx
mdlAbx = fitclinear(X_trainZ, T_train.unnecessary_antibiotics, 'Learner', 'logistic');

%% ======================================================================
% PART 4. TUNE HYBRID MODEL ON VALIDATION SET
% ======================================================================
fprintf('Tuning hybrid utility weights on validation data...\n');

% Get hybrid submodel predictions on VALIDATION set
[~, scoreViral_valid] = predict(mdlViral, X_validZ);
pViral_valid = scoreViral_valid(:,2);

pChange_valid = predict(mdlChange, X_valid);
pChange_valid = max(0, min(1, pChange_valid));

[~, scoreReturn_valid] = predict(mdlReturn, X_validZ);
pReturn_valid = scoreReturn_valid(:,2);

[~, scoreICU_valid] = predict(mdlICU, X_validZ);
pICU_valid = scoreICU_valid(:,2);

[~, scoreAbx_valid] = predict(mdlAbx, X_validZ);
pAbx_valid = scoreAbx_valid(:,2);

TAT_valid = T_valid.pcr_tat_hours;
season_valid = T_valid.respiratory_season;
borderline_valid = double((T_valid.spo2 >= 91 & T_valid.spo2 <= 95) | ...
                          (T_valid.crp >= 15 & T_valid.crp <= 65) | ...
                          (T_valid.cxr_borderline == 1));

% Grid search candidates
wVOI_list        = [0.8 1.0 1.2 1.4 1.6];
wViral_list      = [0.1 0.2 0.3];
wReturn_list     = [0.1 0.2 0.3 0.4];
wICU_list        = [0.3 0.5 0.7 0.9];
wBorderline_list = [0.1 0.2 0.3];
wAbx_list        = [0.1 0.2 0.3];
lambda_list      = [0.05 0.08 0.10 0.12 0.15];
tau_list         = [-0.05 0.00 0.05 0.10 0.15];

% For not-order utility
vNoChange_list   = [0.6 0.8 1.0];
vNoViral_list    = [0.1 0.2 0.3];
vNoBorder_list   = [0.05 0.10 0.15];
vReturnPenalty_list = [0.2 0.3 0.4];
vICUPenalty_list    = [0.3 0.5 0.7];

tuningRows = {};
rowCount = 0;

bestComposite = inf;
bestParams = struct();

for wVOI = wVOI_list
    for wViral = wViral_list
        for wReturn = wReturn_list
            for wICU = wICU_list
                for wBorder = wBorderline_list
                    for wAbx = wAbx_list
                        for lambda = lambda_list
                            for tau = tau_list
                                for vNoChange = vNoChange_list
                                    for vNoViral = vNoViral_list
                                        for vNoBorder = vNoBorder_list
                                            for vRetPen = vReturnPenalty_list
                                                for vICUPen = vICUPenalty_list

                                                    VOI_valid = pChange_valid .* exp(-lambda .* TAT_valid);

                                                    U_order = ...
                                                          wVOI   .* VOI_valid ...
                                                        + wViral .* pViral_valid ...
                                                        + wReturn.* pReturn_valid ...
                                                        + wICU   .* pICU_valid ...
                                                        + 0.20   .* season_valid ...
                                                        + wBorder.* borderline_valid ...
                                                        - wAbx   .* pAbx_valid ...
                                                        - 0.05   .* TAT_valid;

                                                    U_wait = ...
                                                          0.70   .* VOI_valid ...
                                                        + 0.20   .* borderline_valid ...
                                                        + 0.10   .* pReturn_valid ...
                                                        - 0.03   .* TAT_valid;

                                                    U_no_order = ...
                                                          vNoChange .* (1 - pChange_valid) ...
                                                        + vNoViral  .* (1 - pViral_valid) ...
                                                        + vNoBorder .* (1 - borderline_valid) ...
                                                        - vRetPen   .* pReturn_valid ...
                                                        - vICUPen   .* pICU_valid;

                                                    U_not_now = max(U_wait, U_no_order);
                                                    deltaU = U_order - U_not_now;
                                                    A_valid_policy = double(deltaU > tau);

                                                    % Estimate outcomes under this policy
                                                    metrics_valid = evaluate_policy_outcomes( ...
                                                        X_valid, A_valid_policy, muOut, sigmaOut, ...
                                                        mdlOutcomeReturn, mdlOutcomeICU, mdlOutcomeAbx);

                                                    rowCount = rowCount + 1;
                                                    tuningRows(rowCount,:) = { ...
                                                        wVOI, wViral, wReturn, wICU, wBorder, wAbx, ...
                                                        lambda, tau, vNoChange, vNoViral, vNoBorder, ...
                                                        vRetPen, vICUPen, ...
                                                        metrics_valid.PCR_Order_Rate, ...
                                                        metrics_valid.Expected_Return72h, ...
                                                        metrics_valid.Expected_ICUTransfer, ...
                                                        metrics_valid.Expected_UnnecessaryAbx, ...
                                                        metrics_valid.Expected_CompositeHarm}; %#ok<AGROW>

                                                    if metrics_valid.Expected_CompositeHarm < bestComposite
                                                        bestComposite = metrics_valid.Expected_CompositeHarm;
                                                        bestParams.wVOI = wVOI;
                                                        bestParams.wViral = wViral;
                                                        bestParams.wReturn = wReturn;
                                                        bestParams.wICU = wICU;
                                                        bestParams.wBorder = wBorder;
                                                        bestParams.wAbx = wAbx;
                                                        bestParams.lambda = lambda;
                                                        bestParams.tau = tau;
                                                        bestParams.vNoChange = vNoChange;
                                                        bestParams.vNoViral = vNoViral;
                                                        bestParams.vNoBorder = vNoBorder;
                                                        bestParams.vRetPen = vRetPen;
                                                        bestParams.vICUPen = vICUPen;
                                                    end

                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

hybridTuningTable = cell2table(tuningRows, 'VariableNames', { ...
    'wVOI','wViral','wReturn','wICU','wBorder','wAbx', ...
    'lambda','tau','vNoChange','vNoViral','vNoBorder', ...
    'vRetPen','vICUPen', ...
    'PCR_Order_Rate','Expected_Return72h','Expected_ICUTransfer', ...
    'Expected_UnnecessaryAbx','Expected_CompositeHarm'});

hybridTuningTable = sortrows(hybridTuningTable, 'Expected_CompositeHarm', 'ascend');
writetable(hybridTuningTable, fullfile(tabDir, 'hybrid_tuning_results.csv'));

fprintf('Best hybrid validation composite harm = %.4f\n', bestComposite);
disp(bestParams);

%% ======================================================================
% PART 5. REFIT HYBRID SUBMODELS ON TRAIN+VALID, APPLY BEST PARAMS TO TEST
% ======================================================================
fprintf('Refitting hybrid submodels on train+validation data...\n');

% Recompute standardized trainValid
muTV = mean(X_trainValid,1);
sigmaTV = std(X_trainValid,[],1);
sigmaTV(sigmaTV == 0) = 1;

X_trainValidZ = (X_trainValid - muTV) ./ sigmaTV;
X_testZ_tv = (X_test - muTV) ./ sigmaTV;

mdlViral_TV = fitclinear(X_trainValidZ, double(T_trainValid.true_infection_state == 1), 'Learner', 'logistic');
mdlChange_TV = fitrensemble(X_trainValid, T_trainValid.will_change_management, ...
    'Method', 'LSBoost', 'NumLearningCycles', 150);
mdlReturn_TV = fitclinear(X_trainValidZ, T_trainValid.return_72h, 'Learner', 'logistic');
mdlICU_TV = fitclinear(X_trainValidZ, T_trainValid.icu_transfer, 'Learner', 'logistic');
mdlAbx_TV = fitclinear(X_trainValidZ, T_trainValid.unnecessary_antibiotics, 'Learner', 'logistic');

[~, scoreViral_test] = predict(mdlViral_TV, X_testZ_tv);
pViral_test = scoreViral_test(:,2);

pChange_test = predict(mdlChange_TV, X_test);
pChange_test = max(0, min(1, pChange_test));

[~, scoreReturn_test] = predict(mdlReturn_TV, X_testZ_tv);
pReturn_test = scoreReturn_test(:,2);

[~, scoreICU_test] = predict(mdlICU_TV, X_testZ_tv);
pICU_test = scoreICU_test(:,2);

[~, scoreAbx_test] = predict(mdlAbx_TV, X_testZ_tv);
pAbx_test = scoreAbx_test(:,2);

TAT_test = T_test.pcr_tat_hours;
season_test = T_test.respiratory_season;
borderline_test = double((T_test.spo2 >= 91 & T_test.spo2 <= 95) | ...
                         (T_test.crp >= 15 & T_test.crp <= 65) | ...
                         (T_test.cxr_borderline == 1));

VOI_test = pChange_test .* exp(-bestParams.lambda .* TAT_test);

U_order_test = ...
      bestParams.wVOI    .* VOI_test ...
    + bestParams.wViral  .* pViral_test ...
    + bestParams.wReturn .* pReturn_test ...
    + bestParams.wICU    .* pICU_test ...
    + 0.20               .* season_test ...
    + bestParams.wBorder .* borderline_test ...
    - bestParams.wAbx    .* pAbx_test ...
    - 0.05               .* TAT_test;

U_wait_test = ...
      0.70 .* VOI_test ...
    + 0.20 .* borderline_test ...
    + 0.10 .* pReturn_test ...
    - 0.03 .* TAT_test;

U_no_order_test = ...
      bestParams.vNoChange .* (1 - pChange_test) ...
    + bestParams.vNoViral  .* (1 - pViral_test) ...
    + bestParams.vNoBorder .* (1 - borderline_test) ...
    - bestParams.vRetPen   .* pReturn_test ...
    - bestParams.vICUPen   .* pICU_test;

U_not_now_test = max(U_wait_test, U_no_order_test);
deltaU_test = U_order_test - U_not_now_test;
hybridDecision_test = double(deltaU_test > bestParams.tau);

policyNames{end+1} = 'Hybrid_ML_Mechanistic_Tuned';
policyDecisions_test{end+1} = hybridDecision_test(:);

%% ======================================================================
% PART 6. REFIT OUTCOME MODELS ON TRAIN+VALID AND EVALUATE POLICIES ON TEST
% ======================================================================
fprintf('Evaluating all policies on held-out test set...\n');

A_trainValid = y_trainValid(:);
Xout_trainValid = [X_trainValid, A_trainValid];

muOut_TV = mean(Xout_trainValid,1);
sigmaOut_TV = std(Xout_trainValid,[],1);
sigmaOut_TV(sigmaOut_TV == 0) = 1;
Xout_trainValidZ = (Xout_trainValid - muOut_TV) ./ sigmaOut_TV;

mdlOutcomeReturn_TV = fitclinear(Xout_trainValidZ, T_trainValid.return_72h, 'Learner', 'logistic');
mdlOutcomeICU_TV    = fitclinear(Xout_trainValidZ, T_trainValid.icu_transfer, 'Learner', 'logistic');
mdlOutcomeAbx_TV    = fitclinear(Xout_trainValidZ, T_trainValid.unnecessary_antibiotics, 'Learner', 'logistic');

nPolicies = numel(policyNames);
results = table('Size', [nPolicies+1, 8], ...
    'VariableTypes', {'string','double','double','double','double','double','double','double'}, ...
    'VariableNames', {'Policy','PCR_Order_Rate', ...
                      'Expected_Return72h','Expected_ICUTransfer', ...
                      'Expected_UnnecessaryAbx','Expected_CompositeHarm', ...
                      'Decision_Agreement_With_Observed', ...
                      'Decision_Agreement_With_PolicyRef'});

% Observed care row
metrics_obs = evaluate_policy_outcomes(X_test, y_test, muOut_TV, sigmaOut_TV, ...
    mdlOutcomeReturn_TV, mdlOutcomeICU_TV, mdlOutcomeAbx_TV);

results.Policy(1) = "Observed_Care";
results.PCR_Order_Rate(1) = metrics_obs.PCR_Order_Rate;
results.Expected_Return72h(1) = metrics_obs.Expected_Return72h;
results.Expected_ICUTransfer(1) = metrics_obs.Expected_ICUTransfer;
results.Expected_UnnecessaryAbx(1) = metrics_obs.Expected_UnnecessaryAbx;
results.Expected_CompositeHarm(1) = metrics_obs.Expected_CompositeHarm;
results.Decision_Agreement_With_Observed(1) = 1.0;
results.Decision_Agreement_With_PolicyRef(1) = mean(y_test == y_ref_test);

% Policy rows
for i = 1:nPolicies
    A_test_policy = double(policyDecisions_test{i}(:));

    metrics = evaluate_policy_outcomes(X_test, A_test_policy, muOut_TV, sigmaOut_TV, ...
        mdlOutcomeReturn_TV, mdlOutcomeICU_TV, mdlOutcomeAbx_TV);

    results.Policy(i+1) = string(policyNames{i});
    results.PCR_Order_Rate(i+1) = metrics.PCR_Order_Rate;
    results.Expected_Return72h(i+1) = metrics.Expected_Return72h;
    results.Expected_ICUTransfer(i+1) = metrics.Expected_ICUTransfer;
    results.Expected_UnnecessaryAbx(i+1) = metrics.Expected_UnnecessaryAbx;
    results.Expected_CompositeHarm(i+1) = metrics.Expected_CompositeHarm;
    results.Decision_Agreement_With_Observed(i+1) = mean(A_test_policy == y_test);
    results.Decision_Agreement_With_PolicyRef(i+1) = mean(A_test_policy == y_ref_test);
end

results = sortrows(results, 'Expected_CompositeHarm', 'ascend');
writetable(results, fullfile(tabDir, 'policy_outcome_comparison_revised.csv'));

%% ======================================================================
% PART 7. FIGURES
% ======================================================================
figure('Color','w', 'Position', [80 80 1350 650]);
bar(categorical(results.Policy), ...
    [results.Expected_Return72h, results.Expected_ICUTransfer, ...
     results.Expected_UnnecessaryAbx, results.Expected_CompositeHarm], ...
    'grouped');
ylabel('Estimated rate / score');
title('Estimated Patient Outcomes Under Each Decision Policy (Revised)');
legend({'72h Return','ICU Transfer','Unnecessary Antibiotics','Composite Harm'}, ...
    'Location', 'bestoutside');
xtickangle(30);
saveas(gcf, fullfile(figDir, 'policy_outcome_comparison_revised.png'));

figure('Color','w', 'Position', [100 100 1200 550]);
bar(categorical(results.Policy), results.Expected_CompositeHarm);
ylabel('Expected composite harm');
title('Expected Composite Harm by Policy (Lower is Better)');
xtickangle(30);
saveas(gcf, fullfile(figDir, 'policy_composite_harm_revised.png'));

figure('Color','w', 'Position', [100 100 1200 550]);
bar(categorical(results.Policy), results.PCR_Order_Rate);
ylabel('PCR order rate');
title('PCR Ordering Rate by Policy (Revised)');
xtickangle(30);
saveas(gcf, fullfile(figDir, 'policy_pcr_order_rate_revised.png'));

%% ======================================================================
% PART 8. INTERPRETATION
% ======================================================================
fprintf('\n=== REVISED PATIENT OUTCOME COMPARISON INTERPRETATION ===\n');

bestIdx = find(results.Expected_CompositeHarm == min(results.Expected_CompositeHarm), 1);
fprintf('Best policy by estimated composite harm: %s\n', results.Policy(bestIdx));

hybridIdx = find(results.Policy == "Hybrid_ML_Mechanistic_Tuned", 1);
if ~isempty(hybridIdx)
    fprintf('\nTuned hybrid estimated outcomes:\n');
    fprintf('  PCR order rate           = %.3f\n', results.PCR_Order_Rate(hybridIdx));
    fprintf('  Expected 72h return      = %.3f\n', results.Expected_Return72h(hybridIdx));
    fprintf('  Expected ICU transfer    = %.3f\n', results.Expected_ICUTransfer(hybridIdx));
    fprintf('  Expected unnecessary abx = %.3f\n', results.Expected_UnnecessaryAbx(hybridIdx));
    fprintf('  Expected composite harm  = %.3f\n', results.Expected_CompositeHarm(hybridIdx));
end

pureMask = startsWith(results.Policy, "PureML_");
pureResults = results(pureMask, :);

if ~isempty(pureResults) && ~isempty(hybridIdx)
    [~, bestPureLocal] = min(pureResults.Expected_CompositeHarm);
    bestPure = pureResults(bestPureLocal, :);

    fprintf('\nBest pure ML policy by estimated composite harm: %s\n', bestPure.Policy);
    fprintf('  Composite harm = %.3f\n', bestPure.Expected_CompositeHarm);

    diffHarm = bestPure.Expected_CompositeHarm - results.Expected_CompositeHarm(hybridIdx);
    diffRet  = bestPure.Expected_Return72h - results.Expected_Return72h(hybridIdx);
    diffICU  = bestPure.Expected_ICUTransfer - results.Expected_ICUTransfer(hybridIdx);
    diffAbx  = bestPure.Expected_UnnecessaryAbx - results.Expected_UnnecessaryAbx(hybridIdx);

    fprintf('\nTuned hybrid minus best pure ML comparison:\n');
    fprintf('  Improvement in composite harm      = %.4f (positive favors hybrid)\n', diffHarm);
    fprintf('  Improvement in expected 72h return = %.4f\n', diffRet);
    fprintf('  Improvement in expected ICU        = %.4f\n', diffICU);
    fprintf('  Improvement in unnecessary abx     = %.4f\n', diffAbx);

    if diffHarm > 0
        fprintf('\nInterpretation: after tuning, the hybrid model is estimated to outperform the best pure ML policy on composite harm.\n');
    elseif abs(diffHarm) <= 0.01
        fprintf('\nInterpretation: after tuning, the hybrid model is outcome-competitive with the best pure ML policy while retaining decision-theoretic interpretability.\n');
    else
        fprintf('\nInterpretation: even after tuning, the best pure ML policy remains better on estimated composite harm; consider revising simulation assumptions, hybrid features, or outcome models.\n');
    end
end

fprintf('\nBest hybrid parameters selected on validation data:\n');
disp(bestParams);

fprintf('\nImportant note: these are estimated counterfactual outcomes from outcome models, not direct proof from a prospective interventional trial.\n');

%% ======================================================================
% LOCAL FUNCTION
% ======================================================================
function metrics = evaluate_policy_outcomes(Xeval, Aeval, muOut, sigmaOut, ...
    mdlOutcomeReturn, mdlOutcomeICU, mdlOutcomeAbx)

    Xout_eval = [Xeval, Aeval(:)];
    Xout_evalZ = (Xout_eval - muOut) ./ sigmaOut;

    [~, scoreR] = predict(mdlOutcomeReturn, Xout_evalZ);
    pRet = scoreR(:,2);

    [~, scoreI] = predict(mdlOutcomeICU, Xout_evalZ);
    pICU = scoreI(:,2);

    [~, scoreA] = predict(mdlOutcomeAbx, Xout_evalZ);
    pAbx = scoreA(:,2);

    % Composite harm weights can be adjusted
    compositeHarm = 0.45*pRet + 0.35*pICU + 0.20*pAbx;

    metrics = struct();
    metrics.PCR_Order_Rate = mean(Aeval);
    metrics.Expected_Return72h = mean(pRet);
    metrics.Expected_ICUTransfer = mean(pICU);
    metrics.Expected_UnnecessaryAbx = mean(pAbx);
    metrics.Expected_CompositeHarm = mean(compositeHarm);
end