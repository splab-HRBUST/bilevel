% Codes written by Soheil Bahrampour
% Feb 14, 2014

% To perform online supervised task driven dictionary Learning on the train
% XArr with lable trls. The classifier is the multiclass qudratic classifier.

% Inputs:
%       XArr onsisting of train samples where different modalities are concatinated.
%       trls is the train lables
%       n is a vector consiting of feature dimension for each modality
%       d is the number of columns in dictionary
%       opts contains the parameters for multi-task optimization
%       nu is the regularization parameter for the classifier
%       ro is the constant for computing learnning rate
%       InitW consists of parameters for initializing W

% Output:
%       D is the Learned Dictionary which is cell array with same size as X
%           consiting of n*d dictionaries learned from different sensors
%       modelQuad is a Quadratic classifier consisting of the linear coeficients W
%       and bias term b

%%% sparce codes from different modalities are averaged to form the final
%%% feature vector which will be used for classification

function [D, W, result] = SupODL_adam_softmax_everyNIters(XArr, label, ValArr, label_val, dic_size, opts, varargin)
    Y = label.outputVectorTrain;
    trls = label.trls;

    iter = opts.iterSupDic;
    computeCost = opts.computeCost; % flag for computing and ploting cost
    batchSize = opts.batchSize;
    nu = opts.nu;

    n = size(XArr,1); 
    N = size(XArr,2); % number of train samples
    number_classes = size(Y,1);
    

    % adam
    mW = 0;
    mD = 0;
    vW = 0;
    vD = 0;
    learnRate = opts.ro;
%     cost_after = 0.54;

    Y_val = zeros(number_classes,size(ValArr,2)); % each column is a binary vector and is 1 at the row correspoding to the lable of the datapoint
    for j= 1: size(ValArr,2)
        Y_val(label_val(1,j),j) = 1;
    end

    % if nargin > 6 % initial D is provided
    %     D  = varargin{1};
    % else   
    %     % Initialize D using (randomy taken) train samples + at least NumPos samples the positive lables  
    %     D = zeros(sum_n,dic_size);
    %     numAtomPerClass = floor(dic_size/number_classes); % number of atoms per class
    %     %=====================================
    %     temp = 1;
    %     for i = 1: number_classes
    %         tempIndex = find(trls==uniqtrls(1,i));
    %         D(:,temp:temp+numAtomPerClass-1) = XArr(:, tempIndex(1,1:numAtomPerClass)); % at least one sample from the  positive class but not selected randomly
    %         temp = temp + numAtomPerClass;
    %     end
    %     if dic_size > number_classes*numAtomPerClass % fill the rest with random train samples
    %         permut = randperm(N); %randomly shuffle data
    %         D(:,temp:end) = XArr(:,permut(1:dic_size-temp+1)); %initialize to first d train samples
    %     end
    %     %=====================================
    % end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if nargin > 6 % initial D is provided
        D  = varargin{1};
    else   
        % Initialize D using (randomy taken) train samples + at least NumPos samples the positive lables  
        D = zeros(n,dic_size);
        if dic_size > size(XArr,2)
            D(:,1:size(XArr,2)) = XArr(:, randperm(size(XArr,2)));
            D(:,size(XArr,2)+1:end) = XArr(:,randperm(dic_size-size(XArr,2)));
        else
            D = XArr(:, randperm(dic_size));
        end 
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if nargin > 7 % % initial model parameters are provided
       W =  varargin{2};
    else
        % intialize W using unsisupervised learning
       W = 0.01*randn(dic_size, number_classes); % initialize W to small random numbers rather than setting all to zeros
    end

    % learning rate of the stochastic gradient descent algorithm
    % t0 = floor(N/batchSize)*iter/10; % after finalizing N % for setting the learning rate according to the task-driven dic learning paper: we set t0=T/10 where T is total number of updates


    step = 0;
    Atemp2 = zeros(dic_size, N);

    iter_num = (N-rem(N,batchSize))/batchSize;
    t = 1;
    for n_iter = 1 : iter_num
        batch{n_iter} = t:t+batchSize-1;
        t = t + batchSize;
    end
    if t < N
        batch{n_iter+1} = t : N;
    end
    
        % optimization
    if computeCost
        costStep = size(batch,2); % Compute cost every costStep over the last costStep batch of train samples. For each batch of train samples, the cost will be computed before updating the dic using that trian samples
        costTemp = zeros(costStep,1); % to store cost over lasr costStep samples
        costTempCount = 1; % to count how many train sample are passed
        cost = [];
        costIter = 1;
        costAll = [];
    end
    
    niters_Acc = opts.niters_Acc;
    Num_res = floor((n_iter+1)/niters_Acc)*iter;
    acc_pre_val = zeros(1,Num_res);
    acc_pre_val_top5 = zeros(1,Num_res);
    num_res = 1;
    
    iteration = 1;
    % for iteration = 1: iter % For each epoch
    while iteration < iter + 1

        tic;
        fprintf('iter:%d ',iteration);
        permut = randperm(N);
        XArr = XArr(:,permut);
        Y = Y(:,permut);

        for n = 1: size(batch,2) % to loop over all train samples        
            step = step + 1; 
     % ===================================================================
            alpha = full(mexLasso(XArr(:,batch{n}),D,opts));         
            Atemp = alpha;
            minibatch = size(batch{n},2);
            Atemp2(:,batch{n}) = alpha;

            Beta = zeros(dic_size,size(alpha,2));
%             gradLs_actAlphaVec = zeros(size(alpha,1),size(alpha,2));
            for nMinibatch = 1 : size(alpha,2)
                [r,~] = find(alpha(:,nMinibatch));
                
                % softmax regression
                gradLs_actAlpha = - sum(W,2) +  W*exp(W'*alpha(:,nMinibatch))/sum(exp(W'*alpha(:,nMinibatch)));
    %             gradLs_actAlphaVec(:,nMinibatch) = - sum(W,2) +  W*exp(W'*alpha(:,nMinibatch))/sum(exp(W'*alpha(:,nMinibatch)));

                D_lambda = zeros(size(D,1),size(D,2));
                D_lambda(:,r) = D(:,r);
                Beta(:,nMinibatch) = (D_lambda'*D_lambda + opts.lambda2*eye(dic_size))\gradLs_actAlpha;
            end 
 
            temp3 = Beta*squeeze(Atemp2(:,batch{n}))';
            gradD = - D*temp3/minibatch + (XArr(:,batch{n})*Beta' - D*temp3')/minibatch;    

%            gradLs_actAlphaVec = - sum(W,2) +  W*exp(W'*alpha)./sum(exp(W'*alpha));
%            Beta = (D'*D + opts.lambda2*eye(dic_size))\gradLs_actAlphaVec;

%            temp3 = Beta*squeeze(Atemp2(:,batch{n}))';
%            gradD = - D*temp3/minibatch + (XArr(:,batch{n})*Beta' - D*temp3')/minibatch;

            [cost_pre,gradW,~] = softmax_grad(alpha', Y(:,batch{n})', nu, W);

    % ===================================================================
            % compute cost
            if computeCost
                fprintf('%d/%d - %d cost of this batch:%2.4f\n',costTempCount,costStep,iteration,cost_pre);

                costTemp(costTempCount,1) = cost_pre;
                costTempCount = costTempCount + 1;


                if costTempCount == costStep+1
                    cost = [cost;mean(costTemp)];
                    fprintf('===========================mean cost of this iter:%2.4f\n',mean(costTemp));
                    costTempCount = 1;
                    costAll = [costAll;costTemp];
                    costTemp = zeros(costStep,1);
                    costIter = costIter + 1;
                end
            end

    %         % update W, b, D
    %         learnRate = ro;%min(ro, ro*t0/step);
    %         W = W - learnRate*gradW;
    % %         if intercept
    % %             b = b - learnRate*gradb;
    % %         end
    %         D = D - learnRate*gradD;

            % adam
            [W,mW,vW] = adam(W,learnRate,gradW,mW,vW,step);
            [D,mD,vD] = adam(D,learnRate,gradD,mD,vD,step);
            
            if mod(costTempCount-1,niters_Acc) == 0
            
                Aval=full(mexLasso(ValArr,D,opts));
    
                costVal(num_res) = softmax_grad(Aval', Y_val', nu, W);

                aaa_val_re = exp(Aval'*W)./repmat(sum(exp(Aval'*W),2),1,number_classes);
                aaa_val_re = aaa_val_re';
                [~,prelabel_val] = max(aaa_val_re,[],1);
                acc_pre_val(num_res) = sum(prelabel_val == label_val(:)')/size(label_val(:),1);
                acc_pre_val_top5(num_res) = top5Error(aaa_val_re',label_val);
                acc_pre_test_top5 = topNError(aaa_val_re',label_val,5);
                fprintf('=====================================top-1 acc of this 10 iters is %2.4f\n',acc_pre_val(num_res));
                
                if acc_pre_val(num_res) == max(acc_pre_val)
                    result.D=D;
                    result.W=W;
                    result.acc_pre_test_top5=acc_pre_test_top5;
                end
                
                num_res = num_res + 1;
            end
        
        end
        
    
    iteration = iteration+1;
    
     % % saving model
    % save('D_softmax_500.mat','D','-v7.3');
    % save('W_softmax_500.mat','W','-v7.3');
    % save('costAll_softmax_500.mat','costAll','-v7.3');
    % save('cost_softmax_500.mat','cost','-v7.3');
    end
    
    result.cost=cost;
    result.costAll=costAll;
    result.costVal=costVal;
    result.acc=acc_pre_val;
    result.acc_top5=acc_pre_val_top5;
%     result.D=D;
%     result.W=W;
    result.opts=opts;

end

function [theta_new,m,v] = adam(theta,learnRate,g,m,v,step)
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;

m = beta1.*m + (1-beta1).*g(:);
v = beta2.*v + (1-beta2).*(g(:).^2);
m_hat = m./(1-beta1^step);
v_hat = v./(1-beta2^step);
delta_theta_vec = learnRate.*m_hat./(sqrt(v_hat)+epsilon);
delta_theta = reshape(delta_theta_vec,size(theta,1),size(theta,2));
theta_new = theta - delta_theta;
end

function cost_pre = ComputeCost(Atemp,Y,W,b,nu)
    cost_pre = 0.5*sum(sum((Y - W'*Atemp - repmat(b', 1, size(Y,2))).^2));
    cost_pre = cost_pre/size(Y,2) + nu/2*sum(sum(sum(W.^2))); % note that b is not regularized
end

function  [cost,gradW,P] = softmax_grad(alpha,label_extend,nu,W)

    m = size(alpha,1);

    P=exp(alpha*W)./repmat(sum(exp(alpha*W),2),1,size(label_extend,2));

    cost = -1/m*[label_extend(:)]'*log(P(:))+nu/2*sum(W(:).^2);

    gradW = -1/m*alpha'*(label_extend-P)+nu*W;
    
end