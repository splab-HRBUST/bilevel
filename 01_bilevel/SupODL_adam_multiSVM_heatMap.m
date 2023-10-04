% Codes written by Chen Chen

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

function [D, W, result] = SupODL_adam_multiSVM_heatMap(XArr, label, ValArr, label_val, dic_size, opts, varargin)
    Y = label.outputVectorTrain;
    trls = label.trls;
    
    if size(trls,1) ~= 1
        trls = trls';
    end
    


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
    learnRate_W = opts.ro_W;

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

    step = 0;step_W=0;
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
    
    iteration = 1;
    % for iteration = 1: iter % For each epoch
    while iteration < iter + 1

        tic;
        fprintf('iter:%d ',iteration);
        permut = randperm(N);
        XArr = XArr(:,permut);
        Y = Y(:,permut);
        trls = trls(:,permut);

        for n = 1: size(batch,2) % to loop over all train samples        
            step = step + 1; 
            step_W = step_W + 1;
     % ===================================================================
            alpha = full(mexLasso(XArr(:,batch{n}),D,opts));
            trls_batch = trls(batch{n});
            minibatch = size(batch{n},2);
            Atemp2(:,batch{n}) = alpha;

            Beta = zeros(dic_size,size(alpha,2));
            gradLs_actAlpha = 0;
            for nMinibatch = 1 : size(alpha,2)
                [r,~] = find(alpha(:,nMinibatch));
               
                % hinge loss multiSVM
                [v_max,ind_max] = max(W'*alpha(:,nMinibatch));
                v_min = W(:,trls_batch(nMinibatch))'*alpha(:,nMinibatch);
                if v_max - v_min > -1
                    gradLs_actAlpha = W(:,ind_max)-W(:,trls_batch(nMinibatch));
                end
                
                D_lambda = zeros(size(D,1),size(D,2));
                D_lambda(:,r) = D(:,r);
                Beta(:,nMinibatch) = (D_lambda'*D_lambda + opts.lambda2*eye(dic_size))\gradLs_actAlpha;
            end 
 
            temp3 = Beta*squeeze(Atemp2(:,batch{n}))';
            gradD = - D*temp3/minibatch + (XArr(:,batch{n})*Beta' - D*temp3')/minibatch;    
            
            [Q, ~, cost_pre] = multi_hingeloss_cost(W', alpha, Y(:,batch{n}), nu);
            [gradW] = multi_hingeloss_grad(alpha, W', Q, Y(:,batch{n}), nu, trls(:,batch{n}),number_classes);
            gradW = gradW';

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

            % adam
%             for it=1:100
%             [W,mW,vW] = adam(W,learnRate_W,gradW,mW,vW,step_W);
%             step_W=step_W+1;
%             end
            [W,mW,vW] = adam(W,learnRate_W,gradW,mW,vW,step_W);
            [D,mD,vD] = adam(D,learnRate,gradD,mD,vD,step);
                            
        end
        
    
    iteration = iteration+1;

    end
    
    Aval=full(mexLasso(ValArr,D,opts));
    aaa_val_re = W' * Aval;
    acc_pre_val_top5 = top_N_acc(aaa_val_re',label_val,5);
 
    
    if computeCost
        result.cost=cost;
        result.costAll=costAll;
    end

    result.opts=opts;
    result.D=D;
    result.W=W;
    result.scores=aaa_val_re;
    result.acc_pre_top5=acc_pre_val_top5;

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

function [thetagrad] = multi_hingeloss_grad(data, theta, Q, groundTruth, lambda, labels,classNum)
    X = theta*data;
    [~,q] = max(Q);
    % Xq = full(sparse(q, 1:size(X,2), 1));
    % 
    % if size(Xq,1)<200
    %     for i = 1:10-size(Xq,1)
    %         Xq = [Xq;zeros(1, size(Xq,2))];
    %     end
    % end

    Xq = zeros(classNum,size(data,2)); % each column is a binary vector and is 1 at the row correspoding to the lable of the datapoint
    for j= 1: size(data,2)
        Xq(labels(1,j),j) = 1;
    end

    temp = X.*groundTruth;
    temp1 = X.*Xq;
    temp1(find(temp1==0))=[];
    temp(find(temp==0))=[];
    W=(temp - temp1)<1;
    Y = zeros(size(X));

    for i=1:size(X,2)
        Y(labels(i),i) = -W(i);
        Y(q(i),i) = W(i);
    end
    thetagrad = 1/size(X,2)*Y*data' + lambda * theta;
end

function [Q, X, cost] = multi_hingeloss_cost(theta, data, groundTruth,lambda)
    groundTruth1 = groundTruth;
    groundTruth(find(groundTruth==1)) = -inf;  
    groundTruth(find(groundTruth==0)) = 1; 
    X = theta*data;
    Q = X;
    Q = Q.*groundTruth;
    Q(find(Q==inf)) = -inf;
    temp = X.*groundTruth1;
    temp(find(temp==0))=[];
    t = max(0, 1 - temp + max(Q));
    cost = 1/size(data,2)*sum(t)+lambda*sum(theta(:).^2);
end