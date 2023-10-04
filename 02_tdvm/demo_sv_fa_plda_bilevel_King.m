%% ------------------------------------------------loading King-ASR-010 dataset
fprintf('loading data\n');

% % 用GMM SV作为输入
% M_dev = importdata('/data/chenchen/matlab/program/Speaker_Verification_Toolbox_v1.c/data/King_ASR_010/King_new_test_plan/Mix_60_1024/M_dev.mat');
% M_enroll = importdata('/data/chenchen/matlab/program/Speaker_Verification_Toolbox_v1.c/data/King_ASR_010/King_new_test_plan/Mix_60_1024/M_enroll.mat');
% M_test = importdata('/data/chenchen/matlab/program/Speaker_Verification_Toolbox_v1.c/data/King_ASR_010/King_new_test_plan/Mix_60_1024/M_test.mat');
% ubm = importdata('/data/chenchen/matlab/program/Speaker_Verification_Toolbox_v1.c/data/King_ASR_010/King_new_test_plan/Mix_60_1024/ubm/ubm.mat');

num.enroll = 96; 
num.test = 24;

[num.SVdim, num.samples] = size(M_dev);
num.IVdim = 400; % dimension of i-vector
num.Zdim = 200;  % dimension of latent vector for PLDA
 
num.label_dev = repmat([1:1:150],120,1); % for PLDA

% ---------------------------------------------迭代次数
num.nIters = 1;


%% ---------------------------------------------train TVS using FA
for iTers = 11 : 15
    if iTers == 1
        [patameters_FA, pLDA] = fa_plda_bilevel_em(M_dev, ubm, num);
    else
        [patameters_FA, pLDA] = fa_plda_bilevel_em(M_dev, ubm, num, patameters_FA, pLDA);

    end
        
%% evaluation
Ex_dev = pinv(patameters_FA.L)*patameters_FA.B*(M_dev-patameters_FA.m);
Ex_enroll = pinv(patameters_FA.L)*patameters_FA.B*(M_enroll-patameters_FA.m);
Ex_test = pinv(patameters_FA.L)*patameters_FA.B*(M_test-patameters_FA.m);

Ex_model = zeros(num.IVdim,50);
for i = 1 : 50
    Ex_model(:,i) = mean(Ex_enroll(:,(i-1)*num.enroll+1:i*num.enroll),2); % i-vector models
end
clear i

answer_eva = [ones(1,50*num.test) zeros(1,50*num.test*49)];

fprintf('================= cosine ====================\n');
scores_CDS = [];
scores_CDS.all =1 - pdist2(Ex_test',Ex_model','cosine');

scores_CDS.true = [];
scores_CDS.impostor = [];

for a =  1 : 50
    for b = 1 : 50
        A = scores_CDS.all(num.test*(a-1)+1 : num.test*(a-1)+num.test , b);
        if a == b
            scores_CDS.true = [scores_CDS.true ; A(:)];
        elseif a~=b
            scores_CDS.impostor = [scores_CDS.impostor ; A(:)];
        end
    end
end  
clear a b A
scores_CDS = [scores_CDS.true;scores_CDS.impostor];
[eer_TDVL400_PLDA200_CDS(iTers),~,~,~,dcf_king_TDVL400_PLDA200_CDS(iTers)]=compute_eer(scores_CDS,answer_eva,false);


% fprintf('================= PLDA ======================\n');
% % pLDA_baseline = gplda_em(Ex_dev, num.label_dev(:), num.Zdim, 30);
% scores_PLDA = [];
% scores_PLDA.all = (score_gplda_trials(pLDA, Ex_model, Ex_test))';
% 
% scores_PLDA.true = [];
% scores_PLDA.impostor = [];
% 
% for a =  1 : 50
%     for b = 1 : 50
%         A = scores_PLDA.all(num.test*(a-1)+1 : num.test*(a-1)+num.test , b);
%         if a == b
%             scores_PLDA.true = [scores_PLDA.true ; A(:)];
%         elseif a~=b
%             scores_PLDA.impostor  = [scores_PLDA.impostor  ; A(:)];
%         end
%     end
% end
% clear a b A
% scores_PLDA = [scores_PLDA.true;scores_PLDA.impostor];
% [eer_TDVL400_PLDA200(iTers),~,~,~,dcf_king_TDVL400_PLDA200(iTers)]=compute_eer(scores_PLDA,answer_eva,false);

fprintf('===== Summaray ======= Bilevel =====%d=========\n',iTers);
fprintf('+-------------------------------------------+\n');
fprintf('|    Method     |    EER(%%)   | Min DCF_vox |\n');
fprintf('+---------------+-------------+-------------+\n');
fprintf('|      CDS      |    %2.2f    |   %2.4f    |\n', eer_TDVL400_PLDA200_CDS(iTers),dcf_king_TDVL400_PLDA200_CDS(iTers));
% fprintf('|FA(%d)-PLDA(%d)|    %2.2f    |   %2.4f    |\n', num.IVdim,num.Zdim,eer_TDVL400_PLDA200(iTers),dcf_king_TDVL400_PLDA200(iTers));
fprintf('+---------------+-------------+-------------+\n');

end

% [eer_CDS,nMin_CDS] = min(eer_TDVL400_PLDA200_CDS);
% dcf_vox_CDS = dcf_vox_FA400_PLDA200_CDS(nMin_CDS);
% 
% [eer_PLDA, nMin_PLDA] = min(eer_TDVL400_PLDA200);
% dcf_vox_PLDA = dcf_vox_FA400_PLDA200(nMin_PLDA);
% 
% fprintf('======================================================================\n');
% fprintf('===== Summaray ======= Bilevel =====%d=========\n',nMin_PLDA);
% fprintf('+---------------------------------------------+\n');
% fprintf('|      Method     |    EER(%%)   | Min DCF_vox |\n');
% fprintf('+-----------------+-------------+-------------+\n');
% fprintf('|      CDS      |    %2.2f    |   %2.4f    |\n', eer_CDS,dcf_vox_CDS);
% fprintf('|FA(%d)-PLDA(%d)|     %2.2f    |   %2.4f    |\n', num.IVdim,num.Zdim,eer_PLDA,dcf_vox_PLDA);
% fprintf('+-----------------+-------------+-------------+\n');
% fprintf('Finished :D\n');
% 
% clear eer_CDS dcf_vox_CDS eer_PLDA dcf_vox_PLDA


%% results
%       results of FA-PLDA_Bilevel methods             results of FA(sigma renew)+PLDA methods(iter=10)    results of FA(sigma renew)+PLDA methods(iter=100)
% =================== Summaray ====================    ===================Summaray ====================    =================== Summaray ====================
% +-------------------+-------------+-------------+    +-------------------+-------------+-------------+
% |       Method      |    EER(%)   |   Min DCF   |    |       Method      |    EER(%)   |   Min DCF   |
% +-------------------+-------------+-------------+    +-------------------+-------------+-------------+
% |        CDS        |    4.90     |   0.0241    |    |        CDS        |    4.12     |   0.0212    |   |        CDS        |    4.54     |   0.0271    |
% | FA(800)-PLDA(800) |    3.64     |   0.0253    |    | FA(800)-PLDA(800) |    5.00     |   0.0162    |   | FA(800)-PLDA(800) |    10.50    |   0.0272    |
% +-------------------+-------------+-------------+    +-------------------+-------------+-------------+
% |        CDS        |    4.94     |   0.0247    |    |        CDS        |    4.04     |   0.0211    |
% | FA(600)-PLDA(600) |    3.84     |   0.0280    |    | FA(600)-PLDA(600) |    4.36     |   0.0153    |
% +-------------------+-------------+-------------+    +-------------------+-------------+-------------+
% |        CDS        |    5.08     |   0.0251    |    |        CDS        |    4.07     |   0.0220    |
% | FA(400)-PLDA(400) |    4.34     |   0.0318    |    | FA(400)-PLDA(400) |    3.86     |   0.0160    |
% +-------------------+-------------+-------------+    +-------------------+-------------+-------------+
% |        CDS        |    5.56     |   0.0265    |    |        CDS        |    4.38     |   0.0236    |
% | FA(200)-PLDA(200) |    5.31     |   0.0377    |    | FA(200)-PLDA(200) |    3.52     |   0.0191    |
% +-------------------+-------------+-------------+    +-------------------+-------------+-------------+
% |        CDS        |    5.44     |   0.0262    |    |        CDS        |    4.38     |   0.0236    |   |       CDS         |    4.19     |   0.0230    |
% | FA(200)-PLDA(100) |    5.25     |   0.0371    |    | FA(200)-PLDA(100) |    3.55     |   0.0197    |   | FA(200)-PLDA(100) |    4.79     |   0.0175    |
% +-------------------+-------------+-------------+    +-------------------+-------------+-------------+
% |        CDS        |    4.92     |   0.0226    |    |        CDS        |    4.38     |   0.0236    |
% | FA(200)-PLDA(50)  |    5.44     |   0.0383    |    | FA(200)-PLDA(50)  |    3.79     |   0.0230    |
% +-------------------+-------------+-------------+    +-------------------+-------------+-------------+














