%% Data preparing
% M_all = importdata('../00_Data/King-ASR-010/Mix_60_1024/M_all.mat');
% patameters_FA_baseline = importdata('../00_Data/King-ASR-010/Mix_60_1024/model/patameters_FA_baseline.mat');
% Ex_all = pinv(patameters_FA_baseline.L)*patameters_FA_baseline.B*(M_all-patameters_FA_baseline.m);
% save('../00_Data/King-ASR-010/Mix_60_1024/SID/Ex_all.mat','Ex_all','-v7.3');
% 
% num.enroll = 96;
% num.test = 24;
% 
% for nSpk = 1 : 200
%     Ex_enroll(:,(nSpk-1)*num.enroll+1:nSpk*num.enroll) =  Ex_all(:,(nSpk-1)*120+1:(nSpk-1)*120+96);
%     Ex_test(:,(nSpk-1)*num.test+1  :nSpk*num.test)     =  Ex_all(:,(nSpk-1)*120+97:nSpk*120);
% end
% clear nSpk
% 
% X_enroll = zscore(Ex_enroll')';
% %X_test =  zscore(Ex_test')';
% mu = mean(Ex_enroll,2);
% sigma0 = std(Ex_enroll,0,2);
% sigma = sigma0;
% sigma(sigma0==0) = 1;
% X_test = bsxfun(@minus, Ex_test, mu);
% X_test = bsxfun(@rdivide, X_test, sigma);
% clear mu sigma sigma0
% 
% num.label_enroll = repmat([1:1:200],96,1); % for PLDA
% num.label_test   = repmat([1:1:200],24,1); % for PLDA
% num.label_enroll = num.label_enroll(:); 
% num.label_test   = num.label_test(:); 

%% Paramether detection
fprintf('================= train the Sup Dictionary ======================\n');
dic_size = [800 1000 1200 1400 1600 1800 2000];
opts.lambda  = 1; 
opts.lambda2 = 0.1; 
opts.iterSupDic = 5; 
opts.computeCost = 0; 
opts.batchSize = 256; %size(X_train,2);
opts.ro = 0.01; % learning rate
opts.ro_W = opts.ro; % learning rate for multiSVM
opts.nu = 1e-8;

Y.outputVectorTrain = ind2vec(num.label_enroll');
% % Y.outputVectorTrain = Y.outputVectorTrain(:,1:256);
Y.trls = num.label_enroll;

for nD = 1 : 7
               
    [~,~,result] = SupODL_adam_multiSVM_heatMap(X_enroll(:,1:end), Y, X_test, num.label_test, dic_size(nD), opts); 

    result_dicSize_temp{nD} = result; 
    acc_dicSize_temp{nD} = result.acc_pre_top5;        
    acc_multiSVM_dicSize(nD) = min(result.acc_pre_top5);

    fprintf('dicSize=%d\n',dic_size(nD));
    acc_dicSize_temp{nD}

        
end

% answer_eva = [ones(1,200*num.test) zeros(1,200*num.test*199)];
% scores.all = result_dicSize_temp{end,end}.scores';
% scores.true = [];
% scores.impostor = [];
% 
% for a =  1 : 200
%     for b = 1 : 200
%         A = scores.all(num.test*(a-1)+1 : num.test*(a-1)+num.test , b);
%         if a == b
%             scores.true = [scores.true ; A(:)];
%         elseif a~=b
%             scores.impostor = [scores.impostor ; A(:)];
%         end
%     end
% end  
% clear a b A
% scores = [scores.true;scores.impostor];
% [eer,~,~,~,dcf]=compute_eer(scores,answer_eva,1)