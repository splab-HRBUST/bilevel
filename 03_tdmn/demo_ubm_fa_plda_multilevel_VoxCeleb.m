%% ------------------------------------------------loading voxCeleb dataset
% fprintf('loading data from Yu Shoujie\n');
% 
% % MFCC as the input
% files_dev = importdata('/data/chenchen/data/voxceleb/voxceleb1_mfcc/train_files.txt');
% files_eva = importdata('/data/chenchen/data/voxceleb/voxceleb1_mfcc/test_files.txt');
% 
% spk_cell_dev = importdata('/data/chenchen/data/voxceleb/IVs_Verification/train_speaker_v_cell.mat');
% spk_cell_eva = importdata('/data/chenchen/data/voxceleb/IVs_Verification/test_speaker_v_cell.mat');
%  
% mfcc_dev_cell = cell(size(files_dev))';
% mfcc_eva_cell = cell(size(files_eva))';
% 
% for nfile = 1 : size(files_dev,1)
% % parfor nSpk = 1 : size(files_dev,1)
%     mfccFilePaths = fullfile('/data/chenchen/data',char(files_dev{nfile}));
%     mfcc_dev_cell(nfile) = importdata(mfccFilePaths);
% end
% 
% for nfile = 1 : size(files_eva,1)
% % parfor nSpk = 1 : size(files_eva,1)
%     mfccFilePaths = fullfile('/data/chenchen/data',char(files_eva{nfile}));
%     mfcc_eva_cell(nfile) = importdata(mfccFilePaths);
% end
% 
% label_dev = [];
% spk_id = 1;
% for i = 1 : size(spk_cell_dev,1)
%     spk_dev_num = size(spk_cell_dev{i},2);   
%     label_dev = [label_dev spk_id*ones(1,spk_dev_num)];
%     spk_id = spk_id + 1;
% end
% 
% label_eva = [];
% spk_id = 1;
% for i = 1 : size(spk_cell_eva,1)
%     spk_eva_num = size(spk_cell_eva{i},2);   
%     label_eva = [label_eva spk_id*ones(1,spk_eva_num)];
%     spk_id = spk_id + 1;
% end
% 
% ubm0 = importdata('/data/chenchen/data/voxceleb/ubm/ubm_train.mat');
% % 
% clear nfile i spk_id mfccFilePaths spk_dev_num spk_eva_num
% clear files_dev files_eva spk_cell_dev spk_cell_eva

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('loading data mine\n');

% spk_cell_dev = importdata('/data/chenchen/data/voxceleb/IVs_Verification/train_speaker_v_cell.mat');
% spk_cell_eva = importdata('/data/chenchen/data/voxceleb/IVs_Verification/test_speaker_v_cell.mat');
% 
% label_dev = [];
% spk_id = 1;
% for i = 1 : size(spk_cell_dev,1)
%     spk_dev_num = size(spk_cell_dev{i},2);   
%     label_dev = [label_dev spk_id*ones(1,spk_dev_num)];
%     spk_id = spk_id + 1;
% end
% 
% label_eva = [];
% spk_id = 1;
% for i = 1 : size(spk_cell_eva,1)
%     spk_eva_num = size(spk_cell_eva{i},2);   
%     label_eva = [label_eva spk_id*ones(1,spk_eva_num)];
%     spk_id = spk_id + 1;
% end
% 
% clear spk_cell_dev spk_cell_eva spk_id spk_dev_num spk_eva_num i
% 
% mfcc_dev_cell = importdata('/data/chenchen/matlab/program/Speaker_Verification_Toolbox_v1.c/data/VoxCeleb/mfcc_dev_cell.mat');
% mfcc_eva_cell = importdata('/data/chenchen/matlab/program/Speaker_Verification_Toolbox_v1.c/data/VoxCeleb/mfcc_eva_cell.mat');

%% parameter setting

num.nMFCCdim = 60;
num.nmix = 1024;
num.ds_factor = 1;
% num.tau = 10;
num.SVdim = num.nMFCCdim*num.nmix;

num.IVdim = 400; % dimension of i-vector
num.Zdim = 200; % dimension of latent vector for PLDA
 
num.label_dev = label_dev(:,1:end); % for PLDA

% ---------------------------------------------迭代次数
num.nIters = 1;
num.nworkers = 8;

%% ---------------------------------------------train UBM-FA-PLDA multilevel framwork
% ubm0 = importdata('/data/chenchen/matlab/program/Speaker_Verification_Toolbox_v1.c/data/VoxCeleb/ubm.mat');


for iTers = 17 : 30
    if iTers == 1
        [ubm, patameters_FA, pLDA, states_dev, states_eva] = ubm_fa_plda_multilevel_em_backup(mfcc_dev_cell(1:end), mfcc_eva_cell, num,ubm0);
    else
        [ubm, patameters_FA, pLDA, states_dev, states_eva] = ubm_fa_plda_multilevel_em_backup(states_dev, states_eva, num, ubm, patameters_FA, pLDA);
    end

%% ---------------------------------------------evaluation
N_m_eva = patameters_FA.m.*reshape(repmat(states_eva.N(:)',num.nMFCCdim,1),num.nmix*num.nMFCCdim,size(mfcc_eva_cell,2));
centeredM_eva = bsxfun(@minus, states_eva.F, N_m_eva);
Ex_eva = pinv(patameters_FA.L)*patameters_FA.B*centeredM_eva;

% % % Ex_eva = [];
% % % for n_eva = 1 : size(N_m_eva,2)
% % %     B = bsxfun(@rdivide, patameters_FA.T', N_m_eva(:,n_eva)');
% % %     L = patameters_FA.I + B*patameters_FA.T;
% % %     Ex_eva(:,n_eva)=pinv(L)*B*centeredM_eva(:,n_eva);
% % % end

answer_eva=[];
target1_IVs=[];
target2_IVs=[];
fverification = fopen('/data/chenchen/data/voxceleb/verification_number_form.txt');
tline = fgetl(fverification);
ver_id=1;
while ischar(tline)
     temp_cell=textscan(tline,'%d %d %d');
     answer_eva(ver_id)=temp_cell{1,1};
     target1_IVs(:,ver_id)=Ex_eva(:,temp_cell{1,2});
     target2_IVs(:,ver_id)=Ex_eva(:,temp_cell{1,3});
     tline = fgetl(fverification);
     ver_id=ver_id+1;
end
fclose(fverification);
clear fverification tline temp_cell ver_id

fprintf('================== CDS ======================\n');
scores_CDS = 1 - pdist2(target2_IVs',target1_IVs','cosine');
scores_CDS = diag(scores_CDS);
[eer_FA400_PLDA200_CDS(iTers),~,~,dcf_vox_FA400_PLDA200_CDS(iTers)]=compute_eer(scores_CDS,answer_eva,false);


fprintf('================= PLDA ======================\n');
% pLDA = gplda_em(Ex, num.label_dev(:), num.Zdim, num.nIters);

scores_PLDA=score_gplda_trials(pLDA,target1_IVs,target2_IVs);
scores_PLDA=diag(scores_PLDA);

[eer_FA400_PLDA200(iTers),~,~,dcf_vox_FA400_PLDA200(iTers)]=compute_eer(scores_PLDA,answer_eva,false);



%% EER and Min DCF

fprintf('===== Summaray ======= Bilevel =====%d=========\n',iTers);
fprintf('+-----------------------------------------------+\n');
fprintf('|        Method     |    EER(%%)   | Min DCF_vox |\n');
fprintf('+-------------------+-------------+-------------+\n');
fprintf('|        CDS        |    %2.2f    |   %2.4f    |\n', eer_FA400_PLDA200_CDS(iTers),dcf_vox_FA400_PLDA200_CDS(iTers));
fprintf('|FA(%d)-PLDA(%d)|    %2.2f    |   %2.4f    |\n', num.IVdim,num.Zdim,eer_FA400_PLDA200(iTers),dcf_vox_FA400_PLDA200(iTers));
fprintf('+-------------------+-------------+-------------+\n');
fprintf('Finished :D\n');

end


% ============== Summaray ======= Bilevel =====1=====
% +-------------------------------------------------+
% |        Method       |    EER(%)   | Min DCF_vox |
% +---------------------+-------------+-------------+
% |         CDS         |    19.27    |   0.8552    |
% |UBM-FA(400)-PLDA(200)|    15.50    |   0.8324    |
% +---------------------+-------------+-------------+

% ================= Summaray ==================
% +-------------------------------------------+
% |    Method     |    EER(%)   | Min DCF 08  |
% +---------------+-------------+-------------+
% |      CDS      |    28.40    |   8.9516    |
% |  UBM-FA-PLDA  |    27.40    |   9.9230    |
% +---------------+-------------+-------------+
