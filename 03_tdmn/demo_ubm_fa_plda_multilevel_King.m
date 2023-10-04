%% ------------------------------------------------loading King-ASR-010 dataset
% fprintf('loading data\n');
% % 用GMM SV作为输入
% % m_train=importdata('data/King_ASR_010/Mix_39_1024/X_train.mat'); % GMM supervectors for training
% % m_test=importdata('data/King_ASR_010/Mix_39_1024/X_test.mat');   % GMM supervectors for test
ubm = importdata('data/King_ASR_010/Mix_39_1024/gmm/ubm.mat');    % UBM
% 
% % dim = 39*1024;
% % M_train = m_train(1:dim,:);
% % M_test = m_test(1:dim,:);
% % ubm.mu = ubm0.mu(:);
% % ubm.mu = ubm.mu(1:dim);
% % ubm.sigma = ubm0.sigma(:);
% % ubm.sigma = ubm.sigma(1:dim);
% 
% % ubm.mu = mean(M_train,2);
% % ubm.sigma = diag(cov(M_train'));
% 
% num.spk = 200; % speaker numbers: 200
% num.dev = 80; % 80 samples for training
% num.rest = 40;  % 40 samples for every speaker
% num.train = 32; % 32 samples for inroment
% num.test = num.rest - num.train; % 8 samples for test
% 
% num.nMFCCdim = 39;
% num.nmix = 1024;
% num.ds_factor = 1;
% num.tau = 10;
% num.SVdim = num.nMFCCdim*num.nmix;
% num.samples = num.spk*num.train;

num.IVdim = 400; % dimension of i-vector
num.Zdim = 200; % dimension of latent vector for PLDA
 
num.label_dev = repmat([1:1:num.spk],num.train,1); % for PLDA
nnn = 6400;
num.label_dev = num.label_dev(1:nnn);
num.samples = nnn;

% ---------------------------------------------迭代次数
num.nIters = 1;
num.nworkers = 8;

%% data preparing

% % MFCC as the input
% AllFiles = importdata('data/King_ASR_010/AllFiles.ndx');
% 
% % files_dev =cell(80*200,1);
% files_train = cell(32*200,1);
% files_test = cell(8*200,1);
% for nSpk = 1 : num.spk
% %     files_dev((nSpk-1)*80+1:nSpk*80,1) = AllFiles((nSpk-1)*120+1:(nSpk-1)*120+80);
%     files_train((nSpk-1)*32+1:nSpk*32,1) = AllFiles((nSpk-1)*120+81:(nSpk-1)*120+112);
%     files_test((nSpk-1)*8+1:nSpk*8,1) = AllFiles((nSpk-1)*120+113:nSpk*120);
% end
% 
% % mfcc_dev_cell = cell(1,80*200);
% mfcc_train_cell = cell(1,32*200);
% mfcc_test_cell = cell(1,8*200);
% 
% 
% for nSpk = 1 : num.spk
%     
% %     mfcc_dev = [];
% %     for nFile_dev = 1 : num.dev
% %         mfccFilePaths = fullfile('data/King_ASR_010/Mix_39_1024/mfcc',char(files_dev((nSpk-1)*num.dev+nFile_dev)));
% %         mfcc_dev_cell((nSpk-1)*num.dev+nFile_dev) = importdata(mfccFilePaths);
% % %         mfcc_dev = [mfcc_dev mfcc_dev_cell{(nSpk-1)*num.dev+nFile_dev}];
% %     end
% % %     mfcc_dev_cell_sum{nSpk} = mfcc_dev;
%     
%     for nFile_train = 1 : num.train
%         mfccFilePaths = fullfile('data/King_ASR_010/Mix_39_1024/mfcc',char(files_train((nSpk-1)*num.train+nFile_train)));
%         mfcc_train_cell((nSpk-1)*num.train+nFile_train) = importdata(mfccFilePaths);
%     end
%     
%     for nFile_test = 1 : num.test
%         mfccFilePaths = fullfile('data/King_ASR_010/Mix_39_1024/mfcc',char(files_test((nSpk-1)*num.test+nFile_test)));
%         mfcc_test_cell((nSpk-1)*num.test+nFile_test) = importdata(mfccFilePaths);
%     end
% end
% clear nSpk nFile_dev nFile_train nFile_test mfccFilePaths files_dev files_train files_test


%% ---------------------------------------------train UBM-FA-PLDA multilevel framwork
for iTers = 1 : 15
    if iTers == 1
        [ubm, patameters_FA, pLDA, states_train, states_test] = ubm_fa_plda_multilevel_em_babackup(mfcc_train_cell(1:nnn), mfcc_test_cell, num, ubm);
    else
        [ubm, patameters_FA, pLDA, states_train, states_test] = ubm_fa_plda_multilevel_em_babackup(mfcc_train_cell(1:nnn), mfcc_test_cell, num, ubm, patameters_FA, pLDA);
    end

%% ---------------------------------------------evaluation
N_m_train = patameters_FA.m.*reshape(repmat(states_train.N(:)',num.nMFCCdim,1),num.nmix*num.nMFCCdim,size(mfcc_train_cell,2));
centeredM_train = bsxfun(@minus, states_train.F, N_m_train);
Ex_train = pinv(patameters_FA.L)*patameters_FA.B*centeredM_train;
Ex_model = zeros(num.IVdim,num.spk);
for i = 1 : num.spk
    Ex_model(:,i) = mean(Ex_train(:,(i-1)*num.train+1:i*num.train),2); % i-vector models
end
clear i

N_m_test = patameters_FA.m.*reshape(repmat(states_test.N(:)',num.nMFCCdim,1),num.nmix*num.nMFCCdim,size(mfcc_test_cell,2));
centeredM_test = bsxfun(@minus, states_test.F, N_m_test);
Ex_test = pinv(patameters_FA.L)*patameters_FA.B*centeredM_test;

answer_eva = [ones(1,200*num.test) zeros(1,200*num.test*199)];

fprintf('================= cosine ====================\n');
scores_CDS = [];
scores_CDS.all =1 - pdist2(Ex_test',Ex_model','cosine');

scores_CDS.true = [];
scores_CDS.impostor = [];

for a =  1 : num.spk
    for b = 1 : num.spk
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
[eer_CDS400,~,~,dcf_vox_CDS400]=compute_eer(scores_CDS,answer_eva,false);


fprintf('================= PLDA ======================\n');
% pLDA = gplda_em(Ex, num.label_dev(:), num.Zdim, num.nIters);

scores_PLDA = [];
scores_PLDA.all = (score_gplda_trials(pLDA, Ex_model, Ex_test))';

scores_PLDA.true = [];
scores_PLDA.impostor = [];

for a =  1 : num.spk
    for b = 1 : num.spk
        A = scores_PLDA.all(num.test*(a-1)+1 : num.test*(a-1)+num.test , b);
        if a == b
            scores_PLDA.true = [scores_PLDA.true ; A(:)];
        elseif a~=b
            scores_PLDA.impostor  = [scores_PLDA.impostor  ; A(:)];
        end
    end
end
clear a b A
scores_PLDA = [scores_PLDA.true;scores_PLDA.impostor];
[eer_FA400_PLDA200,~,~,dcf_vox_FA400_PLDA200]=compute_eer(scores_PLDA,answer_eva,false);


% % %% EER and Min DCF
% % [eer_CDS,dcf_CDS] = plotdeteer(scores_CDS.true, scores_CDS.impostor,':m');
% % [eer_PLDA,dcf_PLDA] =plotdeteer(scores_PLDA.true, scores_PLDA.impostor ,'-.b');

% h = legend('CDS','PLDA');
% set(h,'Fontsize',12); 
% clear h

fprintf('================= Summaray =========%d=======\n',iTers);
fprintf('+-------------------------------------------+\n');
fprintf('|    Method     |    EER(%%)   |   Min DCF   |\n');
fprintf('+---------------+-------------+-------------+\n');
fprintf('|      CDS      |    %2.2f     |   %2.4f    |\n', eer_CDS400,dcf_vox_CDS400);
fprintf('|  UBM-FA-PLDA  |    %2.2f     |   %2.4f    |\n', eer_FA400_PLDA200,dcf_vox_FA400_PLDA200);
fprintf('+---------------+-------------+-------------+\n');
fprintf('Finished :D\n');

end

% ================= Summaray ==================
% +-------------------------------------------+
% |    Method     |    EER(%)   |   Min DCF   |
% +---------------+-------------+-------------+
% |      CDS      |    30.06    |   0.0950    |
% |    FA-PLDA    |    4.81     |   0.0320    | UBM
% +---------------+-------------+-------------+

% iter = 65;
% ================= Summaray ==================
% +-------------------------------------------+
% |    Method     |    EER(%)   |   Min DCF   |
% +---------------+-------------+-------------+
% |      CDS      |    5.56     |   0.0300    |
% |  UBM-FA-PLDA  |    4.66     |   0.0351    | UBM = 39*1024; FA = 200; PLDA = 100;
% +---------------+-------------+-------------+
