%% Data preparing
% M_all = importdata('../00_Data/King-ASR-010/Mix_60_1024/M_all.mat');
% patameters_FA_baseline = importdata('../00_Data/King-ASR-010/Mix_60_1024/model/patameters_FA_baseline.mat');
% Ex_all = pinv(patameters_FA_baseline.L)*patameters_FA_baseline.B*(M_all-patameters_FA_baseline.m);
% save('../00_Data/King-ASR-010/Mix_60_1024/Ex_all.mat','Ex_all','-v7.3');
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
dic_size = 1000;
L = [0.001 0.01 0.1 1];
opts.iterSupDic = 5; 
opts.computeCost = 0; 
opts.batchSize = 256; %size(X_train,2);
opts.ro = 0.1; % learning rate
opts.nu = 1e-8;

Y.outputVectorTrain = ind2vec(num.label_enroll');
% % % Y.outputVectorTrain = Y.outputVectorTrain(:,1:256);
Y.trls = num.label_enroll;

acc_softmax_bar3 = zeros(4,4);
for nL1 = 1 : 4
    for nL2 = 1 : 4
        
        opts.lambda  = L(nL1); 
        opts.lambda2 = L(nL2); 
        
        [~,~,result] = SupODL_adam_softmax_heatMap(X_enroll(:,1:end), Y, X_test, num.label_test, dic_size, opts); 
        
        result_temp{nL2,nL1} = result; 
        acc_temp{nL2,nL1} = result.acc_pre_top5;        
        acc_softmax_bar3(nL2,nL1) = min(result.acc_pre_top5);
        
        fprintf('L1=%2.3f,L2=%2.3f\n',L(nL1),L(nL2));
        acc_temp{nL2,nL1}

        
    end
end


