%% ------------------------------------------------loading voxCeleb dataset
fprintf('loading data\n');

% speaker_id_train = importdata('/data/chenchen/data/voxceleb/IVs_Verification/train_speaker_id.mat');
% speaker_id_test = importdata('/data/chenchen/data/voxceleb/IVs_Verification/test_speaker_id.mat');

% for nFloder = 1 : size(speaker_id_train,1)
%     mkdir('/data/chenchen/data/voxceleb/gmm/gmm_train',speaker_id_train{nFloder,1});
% end
% 
% for nFloder = 1 : size(speaker_id_test,1)
%     mkdir('/data/chenchen/data/voxceleb/gmm/gmm_test',speaker_id_test{nFloder,1});
% end

% % MFCC as the input
% files_dev = importdata('/data/chenchen/data/voxceleb/voxceleb1_mfcc/train_files.txt');
% files_eva = importdata('/data/chenchen/data/voxceleb/voxceleb1_mfcc/test_files.txt');
% ubm = importdata('/data/chenchen/data/voxceleb/ubm/ubm_train.mat');
% 
% % adapt GMM
% tic;
% gmm = [];
% map_tau = 10.0;
% config = 'm';
% M_dev = zeros(61440,size(files_dev,1));
% for nFile=1:size(files_dev,1)
% %     path = fullfile('/data/chenchen/data',char(files_dev{nFile}));
% %     load(path);
% %     gmm = mapAdapt(ccc, ubm, map_tau, config);
%     savePath = fullfile('/data/chenchen/data/voxceleb/gmm/gmm_train',char(files_dev{nFile}(31:end)));
% %     save(savePath,'gmm','-v7.3');
%     gmm = importdata(savePath);
%     M_dev(:,nFile) = gmm.mu(:);
% end
% toc; 
% 
% tic;
% gmm = [];
% map_tau = 10.0;
% config = 'm';
% M_eva = zeros(61440,size(files_eva,1));
% for nFile=1:size(files_eva,1)
% %     path = fullfile('/data/chenchen/data',char(files_eva{nFile}));
% %     load(path);
% %     gmm = mapAdapt(ccc, ubm, map_tau, config);
%     savePath = fullfile('/data/chenchen/data/voxceleb/gmm/gmm_test',char(files_eva{nFile}(30:end)));
% %     save(savePath,'gmm','-v7.3'); 
%     gmm = importdata(savePath);
%     M_eva(:,nFile) = gmm.mu(:);    
% end
% toc; 

% M_dev = importdata('/data/chenchen/matlab/program/Speaker_Verification_Toolbox_v1.c/data/VoxCeleb/M_dev.mat');
% M_eva = importdata('/data/chenchen/matlab/program/Speaker_Verification_Toolbox_v1.c/data/VoxCeleb/M_eva.mat');
% ubm = importdata('/data/chenchen/matlab/program/Speaker_Verification_Toolbox_v1.c/data/VoxCeleb/ubm.mat');


%% parameter setting
% % num.label_dev = importdata('/data/chenchen/matlab/program/Speaker_Verification_Toolbox_v1.c/data/VoxCeleb/label_dev.mat'); % for PLDA
% [num.SVdim, num.samples] = size(M_dev);
% num.IVdim = 400; % dimension of i-vector
% num.Zdim = 200; % dimension of latent vector for PLDA

 
num.nIters = 1;

%% training TDVM
for iTers = 1 : 25
%     if iTers == 1
%         [patameters_FA, pLDA] = fa_plda_bilevel_em(M_dev, ubm, num);
%     else
%         [patameters_FA, pLDA] = fa_plda_bilevel_em(M_dev, ubm, num, patameters_FA, pLDA);
%     end

%% evaluation
% % Ex_dev = pinv(patameters_FA.L)*patameters_FA.B*(M_dev-patameters_FA.m);
% Ex_eva = pinv(patameters_FA.L)*patameters_FA.B*(M_eva-patameters_FA.m);

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
clear ver_id tline fverification temp_cell

fprintf('================= cosine ====================\n');
scores_CDS_bilevel = 1 - pdist2(target2_IVs',target1_IVs','cosine');
scores_CDS_bilevel = diag(scores_CDS_bilevel);
[eer_FA400_CDS(iTers),~,~,dcf_vox_FA400_CDS(iTers)]=compute_eer(scores_CDS_bilevel,answer_eva,false);
result_TDVM.eer_FA400_CDS = eer_FA400_CDS;
result_TDVM.dcf_vox_FA400_CDS = dcf_vox_FA400_CDS;

fprintf('================= PLDA ======================\n');
nIters = 1;
% for iTers = 31 : 50
%     
%     if iTers == 1
%         pLDA = gplda_em(Ex_dev, num.label_dev(:), num.Zdim, nIters);
%     else       
%         pLDA = gplda_em(Ex_dev, num.label_dev(:), num.Zdim, nIters, pLDA);
%     end
    
scores_TDVM=score_gplda_trials(pLDA,target1_IVs,target2_IVs);
scores_TDVM=diag(scores_TDVM);
[eer_FA400_PLDA200_30(iTers),~,~,dcf_vox_FA400_PLDA200_30(iTers)]=compute_eer(scores_TDVM,answer_eva,false);
result_TDVM.eer_FA400_PLDA200_30 = eer_FA400_PLDA200_30;
result_TDVM.dcf_vox_FA400_PLDA200_30 = dcf_vox_FA400_PLDA200_30;

fprintf('===== Summaray ======= Bilevel =====%d=========\n',iTers*nIters);
fprintf('+-------------------------------------------+\n');
fprintf('|    Method     |    EER(%%)   |   Min DCF   |\n');
fprintf('+---------------+-------------+-------------+\n');
fprintf('|      CDS      |    %2.2f    |   %2.4f    |\n', eer_FA400_CDS(iTers),dcf_vox_FA400_CDS(iTers));
fprintf('|FA(%d)-PLDA(%d)|    %2.2f    |   %2.4f    |\n', num.IVdim,num.Zdim,eer_FA400_PLDA200_30(iTers),dcf_vox_FA400_PLDA200_30(iTers));
fprintf('+---------------+-------------+-------------+\n');

end
% save('result_TDVM.mat','result_TDVM','-v7.3');

%% EER and Min DCF
% [eer_CDS,dcf_CDS] = plotdeteer(scores_CDS.true, scores_CDS.impostor,':m');
% [eer_PLDA,dcf_PLDA] =plotdeteer(scores_PLDA.true, scores_PLDA.impostor ,'-.b');

% h = legend('CDS','PLDA');
% set(h,'Fontsize',12); 
% clear h

% eer_CDS = min(eer_CDS_ind);
% dcf10_CDS = min(dcf10_CDS_ind);

% [eer_PLDA, nMin] = min(eer_FA400_PLDA400_20);
% dcf_vox_PLDA = dcf_vox_FA400_PLDA400_20(nMin);
% 
% fprintf('======================================================================\n');
% fprintf('===== Summaray ======= Bilevel =====%d=========\n',nMin*nIters);
% fprintf('+---------------------------------------------+\n');
% fprintf('|      Method     |    EER(%%)   | Min DCF_vox |\n');
% fprintf('+-----------------+-------------+-------------+\n');
% % fprintf('|      CDS      |    %2.2f    |   %2.4f    |\n', eer_CDS,dcf10_CDS);
% fprintf('|FA(%d)-PLDA(%d)|    %2.2f    |   %2.4f    |\n', num.IVdim,num.Zdim,eer_PLDA,dcf_vox_PLDA);
% fprintf('+-----------------+-------------+-------------+\n');
% fprintf('Finished :D\n');
% 
% clear eer_PLDA dcf10_PLDA



%% ============================================================================= new MFCC
% ================= Summaray =======24===========
% +---------------------------------------------+
% |      Method     |    EER(%)   |   Min DCF   |
% +-----------------+-------------+-------------+
% |FA(400)-PLDA(200)|    7.49     |   0.0856    |
% +-----------------+-------------+-------------+

%% =============================================================================

% ================= Summaray ================== Baseline,independent
% +-------------------------------------------+
% |    Method     |    EER(%)   |   Min DCF   |
% +---------------+-------------+-------------+
% |      CDS      |    23.29    |   0.0911    |
% |    FA+PLDA    |    13.46    |   0.0982    | iter = 10;no LN, no whitening
% +---------------+-------------+-------------+
% |    FA+PLDA    |    11.40    |   0.0899    | iter = 10;with LN, with whitening
% +---------------+-------------+-------------+
% |    FA+PLDA    |    11.40    |   0.0924    | iter = 10;with LN, no whitening
% +---------------+-------------+-------------+
% |    FA+PLDA    |    13.46    |   0.0980    | iter = 10;no LN, with whitening
% +---------------+-------------+-------------+

% FA: trained iters = 5; renew: T,Sigma 
% PLDA: with LN, with Whitening
% ================= Summaray =======20========= independent
% +-------------------------------------------+
% |    Method     |    EER(%)   |   Min DCF   |
% +---------------+-------------+-------------+
% |FA(400)+PLDA(200)|  11.01    |   0.0884    |
% +---------------+-------------+-------------+


% FA: trained iters = 10;
% PLDA: with LN, with Whitening
% +-------------------------------------------+,independent
% |    Method     |    EER(%)   |   Min DCF   |
% +---------------+-------------+-------------+
% |    FA+PLDA    |    11.40    |   0.0899    |  10
% +---------------+-------------+-------------+
% |    FA+PLDA    |    11.20    |   0.0892    |  20
% +---------------+-------------+-------------+
% |    FA+PLDA    |    11.11    |   0.0890    |  30
% +---------------+-------------+-------------+
% |    FA+PLDA    |    11.12    |   0.0888    |  40
% +---------------+-------------+-------------+
% |    FA+PLDA    |    11.21    |   0.0886    |  50
% +---------------+-------------+-------------+


% FA: trained iters = 10;
% PLDA: no LN, no Whitening
% +-------------------------------------------+,independent
% |    Method     |    EER(%)   |   Min DCF   |
% +---------------+-------------+-------------+
% |    FA+PLDA    |    32.86    |   0.0993    |  1
% +---------------+-------------+-------------+
% |    FA+PLDA    |    15.97    |   0.0982    |  2
% +---------------+-------------+-------------+
% |    FA+PLDA    |    13.30    |   0.0981    |  3
% +---------------+-------------+-------------+
% |    FA+PLDA    |    13.00    |   0.0979    |  4
% +---------------+-------------+-------------+
% |    FA+PLDA    |    12.91    |   0.0979    |  5
% +---------------+-------------+-------------+
% |    FA+PLDA    |    13.46    |   0.0982    |  10
% +---------------+-------------+-------------+
% |    FA+PLDA    |    15.10    |   0.0983    |  20
% +---------------+-------------+-------------+
% |    FA+PLDA    |    16.17    |   0.0977    |  30
% +---------------+-------------+-------------+
% |    FA+PLDA    |    16.96    |   0.0937    |  40
% +---------------+-------------+-------------+
% |    FA+PLDA    |    17.48    |   0.0917    |  50
% +---------------+-------------+-------------+
% |    FA+PLDA    |    17.86    |   0.0910    |  60
% +---------------+-------------+-------------+
% |    FA+PLDA    |    18.17    |   0.0904    |  70
% +---------------+-------------+-------------+
% |    FA+PLDA    |    18.39    |   0.0899    |  80
% +---------------+-------------+-------------+
% |    FA+PLDA    |    18.55    |   0.0896    |  90
% +---------------+-------------+-------------+
% |    FA+PLDA    |    18.67    |   0.0893    |  100
% +---------------+-------------+-------------+

%% ====================================================================================

% ================= Summaray ================== independent
% +-------------------------------------------+
% |    Method     |    EER(%)   |   Min DCF   |
% +---------------+-------------+-------------+
% |      CDS      |    23.29    |   0.0911    |
% |    FA+PLDA    |    13.46    |   0.0982    | iter = 10;
% +---------------+-------------+-------------+
% |      CDS      |    22.84    |   0.0888    |
% |    FA+PLDA    |    15.07    |   0.0983    | iter = 20;
% +---------------+-------------+-------------+


%% ==================================================================================
% UBM is replaced by mean and variance of M_train
% FA 1 iter, PLDA 20 iters
% FA: renew T
% PLDA: with LN, without whitening 
% ================== Summaray =======46==========
% +---------------------------------------------+
% |     Method      |    EER(%)   |   Min DCF   |
% +-----------------+-------------+-------------+
% |FA(400)-PLDA(200)|    10.83    |   0.0859    |
% +-----------------+-------------+-------------+


% without LN, without Whitening
% ================= Summaray ==================
% +-------------------------------------------+
% |    Method     |    EER(%)   |   Min DCF   |
% +---------------+-------------+-------------+
% |      CDS      |    19.81    |   0.0888    |  10
% |    FA-PLDA    |    14.62    |   0.0982    |
% +---------------+-------------+-------------+
% |      CDS      |    19.86    |   0.0885    |  20
% |    FA-PLDA    |    15.28    |   0.0978    |
% +---------------+-------------+-------------+

% with LN, without Witening
% ================= Summaray ==================
% +-------------------------------------------+
% |    Method     |    EER(%)   |   Min DCF   |
% +---------------+-------------+-------------+
% |      CDS      |    24.53    |   0.0949    |  1
% |    FA-PLDA    |    36.97    |   0.1000    |
% +---------------+-------------+-------------+
% |      CDS      |    20.60    |   0.0909    |  2
% |    FA-PLDA    |    17.87    |   0.0985    |
% +---------------+-------------+-------------+
% |      CDS      |    20.23    |   0.0882    |  3
% |    FA-PLDA    |    13.97    |   0.0932    |
% +---------------+-------------+-------------+
% |      CDS      |    20.40    |   0.0878    |  4
% |    FA-PLDA    |    35.62    |   0.0997    |
% +---------------+-------------+-------------+
% |      CDS      |    34.13    |   0.1000    |  5
% |    FA-PLDA    |    35.96    |   0.1000    |
% +---------------+-------------+-------------+


