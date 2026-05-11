% ========================================================================
% File: generate_synthetic_ehr_cd3.m
% Purpose:
%   Generate a synthetic, time-stamped pediatric respiratory EHR dataset
%   for Clinical Decision 3:
%       Order multiplex PCR now vs wait vs do not order
%
% Output files:
%   - synthetic_cd3_dataset.csv
%   - synthetic_cd3_dataset.mat
%   - figures/*.png
%   - tables/*.csv
%
% Notes:
%   - This script creates one shared synthetic cohort so the pure ML and
%     hybrid ML + mechanistic approaches are evaluated on the same patients.
%   - The simulation includes latent variables (true infection state,
%     change-in-management potential, latent severity) and observed EHR-like
%     variables.
%   - This revised version explicitly forces key vectors to be column vectors
%     to avoid logical-indexing and implicit-expansion bugs.
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

% Use absolute paths based on current folder
baseDir = pwd;

figDir = fullfile(baseDir, 'figures');
tabDir = fullfile(baseDir, 'tables');

% Create directories if they don't exist
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

if ~exist(tabDir, 'dir')
    mkdir(tabDir);
end

rng(42);

%% ----------------------------- PARAMETERS ------------------------------
N = 2500;
outputDataCSV = 'synthetic_cd3_dataset.csv';
outputDataMAT = 'synthetic_cd3_dataset.mat';
figDir = 'figures';
tabDir = 'tables';

if ~exist(figDir, 'dir')
    mkdir(figDir);
end
if ~exist(tabDir, 'dir')
    mkdir(tabDir);
end

%% -------------------------- HELPER FUNCTIONS ---------------------------
sigmoid = @(x) 1 ./ (1 + exp(-x));
clip01  = @(x) max(0, min(1, x));
tocol   = @(x) reshape(x, [], 1);

%% -------------------------- DEMOGRAPHICS -------------------------------
patient_id = tocol((1:N)');

age_months = tocol(round(12 + 48*randn(N,1) + 18*rand(N,1)));
age_months(age_months < 0) = 0;
age_months(age_months > 216) = 216;

sex_male = tocol(double(rand(N,1) < 0.52));
insurance_public = tocol(double(rand(N,1) < 0.47));
language_non_english = tocol(double(rand(N,1) < 0.18));
deprivation_index = tocol(clip01(0.15 + 0.7*rand(N,1)));
respiratory_season = tocol(double(rand(N,1) < 0.62));
weekend = tocol(double(rand(N,1) < 0.29));
night_shift = tocol(double(rand(N,1) < 0.34));

%% --------------------------- CLINICAL STATE ----------------------------
% Latent illness severity and infection state
latent_severity = tocol(randn(N,1));

p_viral = sigmoid(1.1 + 1.0*respiratory_season - 0.25*latent_severity + 0.12*(age_months < 24));
p_bacterial = sigmoid(-1.2 + 0.95*latent_severity - 0.45*respiratory_season + 0.18*(age_months < 6));

p_viral = tocol(clip01(p_viral));
p_bacterial = tocol(clip01(p_bacterial));

u = tocol(rand(N,1));

true_infection_state = zeros(N,1); % 1 = viral, 2 = bacterial, 3 = mixed/other

mask_viral = u < p_viral .* (1 - p_bacterial);
mask_bacterial = (u >= p_viral .* (1 - p_bacterial)) & ...
                 (u < p_viral .* (1 - p_bacterial) + p_bacterial .* (1 - p_viral));

true_infection_state(mask_viral) = 1;
true_infection_state(mask_bacterial) = 2;
true_infection_state(true_infection_state == 0) = 3;

true_viral = double(true_infection_state == 1);
true_bacterial = double(true_infection_state == 2);
true_mixed = double(true_infection_state == 3);

%% ------------------------ OBSERVED EHR FEATURES ------------------------
fever_c = 37.0 + 0.5*randn(N,1) + 0.8*true_bacterial + 0.45*true_viral + 0.3*latent_severity;
fever_c = tocol(max(36.0, min(40.5, fever_c)));

spo2 = 98 - 1.2*randn(N,1) - 2.2*(latent_severity > 0.8) - 1.1*true_bacterial - 0.8*true_viral;
spo2 = tocol(max(84, min(100, round(spo2))));

tachypnea = tocol(double(sigmoid(-0.2 + 1.1*latent_severity + 0.45*true_viral + 0.35*true_bacterial + 0.25*(age_months < 24)) > rand(N,1)));
retractions = tocol(double(sigmoid(-0.5 + 1.2*latent_severity + 0.55*true_viral) > rand(N,1)));
wheezing = tocol(double(sigmoid(-0.7 + 0.75*true_viral + 0.20*latent_severity + 0.25*(age_months < 36)) > rand(N,1)));
crackles = tocol(double(sigmoid(-0.7 + 0.7*true_bacterial + 0.35*latent_severity) > rand(N,1)));
poor_feeding = tocol(double(sigmoid(-1.1 + 0.65*(age_months < 12) + 0.55*latent_severity) > rand(N,1)));

wbc = 8 + 2.2*randn(N,1) + 3.0*true_bacterial + 0.8*true_mixed + 0.3*latent_severity;
wbc = tocol(max(2.5, min(28, wbc)));

crp = exp(1.2 + 0.6*randn(N,1) + 0.65*true_bacterial + 0.18*true_viral + 0.25*latent_severity);
crp = tocol(min(crp, 220));

procalcitonin = exp(-0.3 + 0.8*randn(N,1) + 0.95*true_bacterial + 0.18*true_mixed + 0.2*latent_severity);
procalcitonin = tocol(min(procalcitonin, 18));

cxr_borderline = tocol(double(sigmoid(-1.0 + 0.7*true_bacterial + 0.35*true_mixed + 0.4*latent_severity) > rand(N,1)));
prior_antibiotics = tocol(double(rand(N,1) < sigmoid(-1.4 + 0.55*true_bacterial + 0.35*weekend)));

hours_since_presentation = tocol(0.5 + 10*rand(N,1));

%% ------------------- PCR TURNAROUND / VALUE OF INFO --------------------
% TAT depends on workflow/time; longer at night/weekend
pcr_tat_hours = 2.5 + 1.2*randn(N,1) + 2.2*night_shift + 1.6*weekend;
pcr_tat_hours = tocol(max(1, min(14, pcr_tat_hours)));

% Probability PCR result changes management:
% highest when viral suspicion exists, bacterial uncertainty exists,
% and antibiotics/admission decisions are borderline.
borderline_clinical_zone = tocol(double( ...
    (spo2 >= 91 & spo2 <= 95) | ...
    (crp >= 15 & crp <= 65) | ...
    (cxr_borderline == 1)));

p_change_mgmt_true = sigmoid( ...
    -1.1 ...
    + 1.1*true_viral ...
    + 0.55*borderline_clinical_zone ...
    + 0.35*prior_antibiotics ...
    + 0.25*respiratory_season ...
    - 0.11*pcr_tat_hours ...
    + 0.18*latent_severity);

p_change_mgmt_true = tocol(clip01(p_change_mgmt_true));
will_change_management = tocol(double(rand(N,1) < p_change_mgmt_true));

%% ---------------------- MANAGEMENT / OUTCOMES --------------------------
% Latent downstream outcomes, affected by infection, severity, and testing value
p_ed_return_72h = sigmoid(-2.0 + 0.8*latent_severity + 0.45*true_bacterial + 0.30*(1 - will_change_management));
p_icu_transfer = sigmoid(-3.1 + 1.3*latent_severity + 0.9*true_bacterial + 0.35*(spo2 < 90));
p_unnecessary_antibiotics = sigmoid(-1.2 + 1.1*true_viral + 0.3*prior_antibiotics - 0.5*will_change_management);

p_ed_return_72h = tocol(clip01(p_ed_return_72h));
p_icu_transfer = tocol(clip01(p_icu_transfer));
p_unnecessary_antibiotics = tocol(clip01(p_unnecessary_antibiotics));

return_72h = tocol(double(rand(N,1) < p_ed_return_72h));
icu_transfer = tocol(double(rand(N,1) < p_icu_transfer));
unnecessary_antibiotics = tocol(double(rand(N,1) < p_unnecessary_antibiotics));

%% ------------------- REFERENCE POLICY FOR 3-WAY LABEL ------------------
% 1 = order_now, 2 = wait, 3 = do_not_order
benefit_order_now = ...
    1.4*p_change_mgmt_true ...
    + 0.35*respiratory_season ...
    + 0.25*prior_antibiotics ...
    + 0.18*borderline_clinical_zone ...
    - 0.10*pcr_tat_hours;

benefit_wait = ...
    0.85*p_change_mgmt_true ...
    + 0.22*borderline_clinical_zone ...
    + 0.12*latent_severity ...
    - 0.04*pcr_tat_hours;

benefit_no_order = ...
    0.50*(1 - p_change_mgmt_true) ...
    + 0.15*(crp < 12) ...
    + 0.12*(procalcitonin < 0.15) ...
    + 0.08*(spo2 > 95);

score_order_now = tocol(benefit_order_now + 0.10*randn(N,1));
score_wait = tocol(benefit_wait + 0.10*randn(N,1));
score_no = tocol(benefit_no_order + 0.10*randn(N,1));

all_scores = [score_order_now, score_wait, score_no];
[~, y_policy] = max(all_scores, [], 2);
y_policy = tocol(y_policy);

%% ---------------- CLINICIAN THRESHOLD VARIATION / NOISE ----------------
% IMPORTANT: force randsample output to column vector to avoid implicit
% expansion against column variables like night_shift.
clinician_style = randsample([-1, 0, 1], N, true, [0.2 0.55 0.25])';
clinician_style = tocol(clinician_style);

% conservative = -1, balanced = 0, test-prone = 1
flip_prob = 0.05 ...
          + 0.06*(clinician_style == 1) ...
          + 0.04*(night_shift == 1);

flip_prob = tocol(flip_prob);
flip_prob = clip01(flip_prob);

flip_mask = tocol(rand(N,1) < flip_prob);
random_alt = tocol(randi(3, N, 1));

y_decision = y_policy;
y_decision(flip_mask) = random_alt(flip_mask);
y_decision = tocol(y_decision);

%% ------------------------ OBSERVED PCR RESULT --------------------------
% PCR result only meaningful if ordered; still stored in synthetic data.
p_positive_viral_panel = 0.08 + 0.82*true_viral + 0.32*true_mixed - 0.08*true_bacterial;
p_positive_viral_panel = tocol(clip01(p_positive_viral_panel));
pcr_result_positive = tocol(double(rand(N,1) < p_positive_viral_panel));

%% ----------------------- FINAL SAFETY CHECKS ---------------------------
varsToCheck = { ...
    patient_id, age_months, sex_male, insurance_public, language_non_english, ...
    deprivation_index, respiratory_season, weekend, night_shift, hours_since_presentation, ...
    fever_c, spo2, tachypnea, retractions, wheezing, crackles, poor_feeding, ...
    wbc, crp, procalcitonin, cxr_borderline, prior_antibiotics, pcr_tat_hours, ...
    pcr_result_positive, return_72h, icu_transfer, unnecessary_antibiotics, ...
    true_infection_state, latent_severity, p_change_mgmt_true, will_change_management, ...
    y_policy, y_decision};

for i = 1:numel(varsToCheck)
    if ~isequal(size(varsToCheck{i}), [N,1])
        error('A generated variable is not N x 1. Check vector dimensions before building the table.');
    end
end

%% ---------------------------- FINAL TABLE ------------------------------
T = table( ...
    patient_id, age_months, sex_male, insurance_public, language_non_english, ...
    deprivation_index, respiratory_season, weekend, night_shift, hours_since_presentation, ...
    fever_c, spo2, tachypnea, retractions, wheezing, crackles, poor_feeding, ...
    wbc, crp, procalcitonin, cxr_borderline, prior_antibiotics, pcr_tat_hours, ...
    pcr_result_positive, return_72h, icu_transfer, unnecessary_antibiotics, ...
    true_infection_state, latent_severity, p_change_mgmt_true, will_change_management, ...
    y_policy, y_decision, ...
    'VariableNames', { ...
    'patient_id','age_months','sex_male','insurance_public','language_non_english', ...
    'deprivation_index','respiratory_season','weekend','night_shift','hours_since_presentation', ...
    'fever_c','spo2','tachypnea','retractions','wheezing','crackles','poor_feeding', ...
    'wbc','crp','procalcitonin','cxr_borderline','prior_antibiotics','pcr_tat_hours', ...
    'pcr_result_positive','return_72h','icu_transfer','unnecessary_antibiotics', ...
    'true_infection_state','latent_severity','p_change_mgmt_true','will_change_management', ...
    'y_policy','y_decision'});

writetable(T, outputDataCSV);
save(outputDataMAT, 'T');

%% ----------------------------- SUMMARIES -------------------------------
summaryTable = table;
summaryTable.Metric = { ...
    'Number of patients'; ...
    'Order now (%)'; ...
    'Wait (%)'; ...
    'Do not order (%)'; ...
    'Mean PCR TAT (h)'; ...
    'Mean p(change management)'; ...
    '72h return (%)'; ...
    'ICU transfer (%)'; ...
    'Unnecessary antibiotics (%)'};
summaryTable.Value = [ ...
    N; ...
    mean(y_decision == 1)*100; ...
    mean(y_decision == 2)*100; ...
    mean(y_decision == 3)*100; ...
    mean(pcr_tat_hours); ...
    mean(p_change_mgmt_true); ...
    mean(return_72h)*100; ...
    mean(icu_transfer)*100; ...
    mean(unnecessary_antibiotics)*100];

writetable(summaryTable, fullfile(tabDir, 'dataset_summary.csv'));

%% ------------------------------ FIGURES --------------------------------
figure('Color','w');
histogram(pcr_tat_hours, 25);
xlabel('PCR turnaround time (hours)');
ylabel('Count');
title('Synthetic PCR Turnaround Time Distribution');
saveas(gcf, fullfile(figDir, 'tat_distribution.png'));

figure('Color','w');
histogram(p_change_mgmt_true, 25);
xlabel('Probability PCR changes management');
ylabel('Count');
title('Synthetic Distribution of Change-in-Management Probability');
saveas(gcf, fullfile(figDir, 'p_change_management_distribution.png'));

figure('Color','w');
bar([mean(y_decision == 1), mean(y_decision == 2), mean(y_decision == 3)] * 100);
set(gca, 'XTickLabel', {'Order now','Wait','Do not order'});
ylabel('Percent of cohort');
title('Reference Decision Distribution');
saveas(gcf, fullfile(figDir, 'reference_decision_distribution.png'));

figure('Color','w');
scatter(pcr_tat_hours, p_change_mgmt_true, 18, 'filled');
xlabel('PCR turnaround time (hours)');
ylabel('Probability PCR changes management');
title('Change-in-Management Probability vs PCR Turnaround Time');
saveas(gcf, fullfile(figDir, 'pchange_vs_tat.png'));

%% ------------------------- AUTOMATED INTERPRETATION --------------------
fprintf('\n=== SYNTHETIC DATASET INTERPRETATION ===\n');
fprintf('Generated %d synthetic pediatric respiratory cases.\n', N);
fprintf('Decision distribution: Order now %.1f%%, Wait %.1f%%, Do not order %.1f%%.\n', ...
    mean(y_decision == 1)*100, mean(y_decision == 2)*100, mean(y_decision == 3)*100);
fprintf('Mean PCR turnaround time is %.2f hours.\n', mean(pcr_tat_hours));
fprintf('Mean probability that PCR changes management is %.2f.\n', mean(p_change_mgmt_true));
fprintf('72-hour return rate is %.1f%% and ICU transfer rate is %.1f%%.\n', ...
    mean(return_72h)*100, mean(icu_transfer)*100);
fprintf('This dataset includes time, uncertainty, and downstream outcome structure suitable for both pure ML and hybrid decision modeling.\n');

%% -------------------------- OPTIONAL DEBUG INFO ------------------------
fprintf('\nDimension checks:\n');
fprintf('size(y_policy)      = [%d %d]\n', size(y_policy,1), size(y_policy,2));
fprintf('size(flip_prob)     = [%d %d]\n', size(flip_prob,1), size(flip_prob,2));
fprintf('size(flip_mask)     = [%d %d]\n', size(flip_mask,1), size(flip_mask,2));
fprintf('size(random_alt)    = [%d %d]\n', size(random_alt,1), size(random_alt,2));
fprintf('size(y_decision)    = [%d %d]\n', size(y_decision,1), size(y_decision,2));