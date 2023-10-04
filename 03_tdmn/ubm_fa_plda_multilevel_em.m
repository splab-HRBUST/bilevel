% function [ubm, patameters_FA, plda, M_train, M_test] = ubm_fa_plda_bilevel_em(mfcc_ubm, mfcc_test, num, ubm0, patameters_FA0, plda0)
function [ubm, patameters_FA, plda, states_test] = ubm_fa_plda_multilevel_em(mfcc_train, mfcc_test, num, ubm0, patameters_FA0, plda0)
% function patameters_FA = fa_plda_bilevel_em(M_train, ubm, num)
% trains a PLDA model given training data using EM algorithm. Speaker
% labels are given in spk_labs. nphi determines the dimensionality of the
% Eigenvoice subspace, and niter specifies the number of EM iteration.
%
% Technically, assuming a factor analysis (FA) model of the form:
%
%           x = m + Phi . y + e
%
% for i-vectors, the code computes the maximum likelihood estimate (MLE)of
% the factor loading matrix Phi (aka the Eigenvoice subspace). Here, x is
% the i-vector, m is the mean of training i-vectors, and y~N(0,I) is a
% vector of latent factors. The residual noise term e explains the
% variability not captured through the latent variables.
%
% Inputs:
%   data            : input data matrix, one observation per column
%   spk_labs        : class labels for observations in data matrix 
%   nphi            : dimensionality of the Eigenvoice subspace 
%   niter           : number of EM iterations for learning PLDA model
%
% Outputs:
%    plda           : a structure containing the PLDA hyperparameters
%					  as well as the mean of development data and a whitening 
%					  transform:(plda.Phi: Eigenvoice matrix, plda.Sigma: covariance
%					  matrix of the residual noise, plda.M: mean, plda.W: whitening transform)
%

%% data reading and intializing
dataList = mfcc_train(:);

nmix = num.nmix;
nMFCCdim = num.nMFCCdim;
SVdim = num.SVdim;
ds_factor = num.ds_factor;
% tau = num.tau;
ndim = num.IVdim;
spk_labs = num.label_dev;
nphi = num.Zdim;
niter = num.nIters;
nSamples = size(mfcc_train(:),1);
% cost = zeros(niter,1);
nworkers = num.nworkers;

% % make sure the labels are sorted
% [spk_labs, Inx] = sort(spk_labs);
% M_train = M_train(:, Inx);
[~, ia, ic] = unique(spk_labs, 'stable');
spk_counts = histc(ic, 1 : numel(ia)); % # sessions per speaker

if nargin > 3
    % Initial UBM is provided
    ubm =ubm0;
else
    % Initialize UBM parameters
    fprintf('Initializing the UBM hyperparameters ...\n');
    [gm, gv] = comp_gm_gv(dataList);
    ubm = gmm_init(gm, gv);
    
    mix = 1;
    N = 0; F = 0; S = 0; L = 0; nframes = 0;
%    parfor (ix = 1 : nfiles, nworkers)
    for ix = 1 : nSamples
        [n, f, s, l] = expectation(dataList{ix}(:, 1:ds_factor:end), ubm);
        N = N + n; F = F + f; S = S + s; L = L + sum(l);
        nframes = nframes + length(l);
    end
    ubm = maximization(N, F, S);
    while ( mix < nmix )
        ubm = gmm_mixup(ubm);
        mix = mix * 2;
    end

    % Train for front 1/2 UBM components
    % gradually increase the number of iterations per binary split
        % mix = [1 2 4 8 16 32 64 128 256 512 1024];
    niter_ubm = [1 2 2 2  2  2  2  2   1   1   1];
%     niter_ubm = [1 2 4 4  4  4  6  6   10  10  15];
%     niter_ubm(log2(nmix) + 1) = niter;
    
    while ( mix <= nmix )
        if ( mix >= nmix/2 ), ds_factor = 1; end % not for the last two splits!
        fprintf('\nRe-estimating the UBM hyperparameters for %d components ...\n', mix);
        for iter = 1 : niter_ubm(log2(mix) + 1)
            fprintf('EM iter#: %d \t', iter);
            N = 0; F = 0; S = 0; L = 0; nframes = 0;
            tim = tic;
%             parfor (ix = 1 : nSamples, nworkers)
            for ix = 1 : nSamples
                [n, f, s, l] = expectation(dataList{ix}(:, 1:ds_factor:end), ubm);
                N = N + n; F = F + f; S = S + s; L = L + sum(l);
                nframes = nframes + length(l);
            end
            tim = toc(tim);
            fprintf('[llk = %.2f] \t [elaps = %.2f s]\n', L/nframes, tim);
            ubm = maximization(N, F, S);
        end
        if ( mix < nmix )
            ubm = gmm_mixup(ubm); 
        end
        mix = mix * 2;
    end
    
end

if nargin > 4
    patameters_FA = patameters_FA0;
else
    % initializing FA parameters
    fprintf('Initializing the FA hyperparameters ...\n');
    patameters_FA.m = reshape(ubm.mu, num.SVdim, 1);
%     patameters_FA.centeredM = bsxfun(@minus, M_train, patameters_FA.m);
%     patameters_FA.variancesM = sum(patameters_FA.centeredM.^2, 2) / num.samples;
    
    patameters_FA.I = eye(num.IVdim);
    patameters_FA.Sigma = reshape(ubm.sigma, num.SVdim, 1); %ones(1024*39,1);
    patameters_FA.T = randn(nMFCCdim * nmix, ndim) * sum(patameters_FA.Sigma) * 0.001;
%     patameters_FA.T = M_train(:,1:num.IVdim)-patameters_FA.m;   %rand(1024*39, Fac);
end

if nargin > 5
    mu    = plda0.M;
    Phi   = plda0.Phi;
    Sigma = plda0.Sigma;
else
    % Initialize PLDA parameters randomly
    fprintf('Initializing the PLDA hyperparameters ...\n');
    [s1, s2] = RandStream.create('mrg32k3a', 'NumStreams', 2);
    Sigma    = 100 * randn(s1, ndim); % covariance matrix of the residual term
    Phi = randn(s2, ndim, nphi); % factor loading matrix (Eignevoice matrix)
    Phi = bsxfun(@minus, Phi, mean(Phi, 2));
    W2   = calc_white_mat(Phi' * Phi);
    Phi = Phi * W2; % orthogonalize Eigenvoices (columns)
end  

%% train the multilevel GMM-FA-PLDA
fprintf('\nEstimating the UBM-FA-PLDA ..\n');

for iter = 1 : niter
    fprintf('EM iter#: %d \t', iter);
    tim = tic;

    %% Lower level for UBM
%     N = 0; F = 0; S = 0; L = 0; %nframes = 0;
% %     parfor (ix = 1 : nSamples, nworkers)
%     for ix = 1 : nSamples
%         [n, f, s, l] = expectation(dataList{ix}(:, 1:ds_factor:end), ubm);
%         N = N + n; F = F + f; S = S + s; L = L + sum(l);
% %         nframes = nframes + length(l);
%     end
%     ubm.sigma = maximization(N, F, S);
    
    N_train = zeros(nmix,size(mfcc_train,2));
    F_train = zeros(SVdim,size(mfcc_train,2));
    L_train = 0;
    nframes = 0;
%     parfor (n_train = 1 : size(mfcc_train,2), nworkers)
    for n_train = 1 : size(mfcc_train,2)
        [N, F, ~, L] = expectation(mfcc_train{n_train}, ubm);
        N_train(:,n_train) = N(:);
        F_train(:,n_train) = F(:);
        L_train = L_train + sum(L);
        nframes = nframes + length(L);
%         stats_train{n_train} = [N_train; F_train];
%         alpha = N_train ./ (N_train + tau); % tarde-off between ML mean and UBM mean
%         m_ML = bsxfun(@rdivide, F_train, N_train);
%         m = bsxfun(@times, ubm.mu, (1 - alpha)) + bsxfun(@times, m_ML, alpha);
%         M_train(:,n_train) = F_train(:);
    end
    fprintf('[llk = %.2f] \t', L_train/nframes);

    %% Middle level for FA
    patameters_FA.m = reshape(ubm.mu, num.SVdim, 1);
    N_m = patameters_FA.m.*reshape(repmat(N_train(:)',nMFCCdim,1),nmix*nMFCCdim,nSamples);
    patameters_FA.centeredM = bsxfun(@minus, F_train, N_m);
%     patameters_FA.variancesM = sum(patameters_FA.centeredM.^2, 2) / num.samples;
    clear N_m N_train F_train L_train nframes
    

 for iter_FA = 1 : 5

        % E step
        patameters_FA.B = bsxfun(@rdivide, patameters_FA.T', patameters_FA.Sigma');
        patameters_FA.L = patameters_FA.I + patameters_FA.B*patameters_FA.T;
    %     Ex = patameters_FA.L\patameters_FA.B*patameters_FA.centeredM;
        Ex = pinv(patameters_FA.L)*patameters_FA.B*patameters_FA.centeredM;
    %     Exx = Ex*Ex' + nSamples*pinv(patameters_FA.L);

    %     % M step
    %     patameters_FA.T = (patameters_FA.centeredM*Ex')*inv(Exx);
    %     patameters_FA.Sigma = patameters_FA.variancesM - sum(patameters_FA.T .* (patameters_FA.T * Exx), 2) / num.samples;   

        mu = mean(Ex, 2);
        data = bsxfun(@minus, Ex, mu); % centering the data
%         data0 = data;
%         [data, sigma] = length_norm(data); % normalizing the length
        W1   = 1;%calc_white_mat(cov(data'));
        data = W1' * data; % whitening the data

        %% Upper level for PLDA
        for iter_PLDA = 1 : 20
            % E step
            [Ey, Eyy] = expectation_plda(data, Phi, Sigma, spk_counts);
            % M step for PLDA
            [Phi, Sigma] = maximization_plda(data, Ey, Eyy);    
        end

        % M step for FA
    %     delta_w_Q_PLDA = -Sigma\(data-Phi*Ey);
    %     inv_delta_w_Q_PLDA_wT = inv(delta_w_Q_PLDA*data');
        delta_w_Q_PLDA = -pinv(Sigma)*(data-Phi*Ey);
        inv_delta_w_Q_PLDA_wT = pinv(delta_w_Q_PLDA*data');
        M_train = bsxfun(@plus, patameters_FA.centeredM, patameters_FA.m);
        T = M_train*delta_w_Q_PLDA'*inv_delta_w_Q_PLDA_wT*pinv(patameters_FA.I+data*delta_w_Q_PLDA'*inv_delta_w_Q_PLDA_wT);
        patameters_FA.T = T;
    %     patameters_FA.B = bsxfun(@rdivide, patameters_FA.T', patameters_FA.Sigma');
    %     patameters_FA.L = patameters_FA.I + patameters_FA.B*patameters_FA.T;
        delta_w_Q_PLDA_ = delta_w_Q_PLDA;
        clear Ex data Ey Eyy
        

%         delta_w_Q_PLDA = -pinv(Sigma)*(data-Phi*Ey);
%     %     delta_w_Q_PLDA = -Sigma\(data-Phi*Ey);
%         A1 = bsxfun(@rdivide, delta_w_Q_PLDA, sigma);
%         A2_1 = sum(bsxfun(@times, delta_w_Q_PLDA, data0));
%         A2_2 = bsxfun(@times, A2_1, data0);
%         A2 = bsxfun(@rdivide, A2_2, sigma.*sigma);
%         delta_w_Q_PLDA_ =  A1 - A2;
%         Exx_new = Ex*delta_w_Q_PLDA_' + delta_w_Q_PLDA_*Ex';
%         T = patameters_FA.centeredM*delta_w_Q_PLDA_'*pinv(Exx_new);  % the best
%     %     T = patameters_FA.centeredM*delta_w_Q_PLDA_'/(Exx_new);  % the best
%         patameters_FA.T = T;
% %         patameters_FA.B = bsxfun(@rdivide, patameters_FA.T', patameters_FA.Sigma');
% %         patameters_FA.L = patameters_FA.I + patameters_FA.B*patameters_FA.T;
%         clear Ex Exx_new data data0 Ey Eyy


    %     patameters_FA.T = (patameters_FA.centeredM*Ex')*inv(Exx);
    %     patameters_FA.Sigma = patameters_FA.variancesM - sum(patameters_FA.T .* (patameters_FA.T * Exx), 2) / num.samples;

 end
       % M step for UBM   
       R_w = 0; R_mu = 0; R_sigma = 0;
       for n_train = 1 : size(mfcc_train,2)
           [r_w, r_mu, r_sigma] = expectation_ubm(mfcc_train{n_train}, ubm, delta_w_Q_PLDA_(:,n_train), patameters_FA.L, patameters_FA.B);
           R_w     = R_w + r_w;
           R_mu    = R_mu + r_mu;
           R_sigma = R_sigma + r_sigma;
       end
         
       ubm = maximization_ubm(R_w, R_mu, R_sigma);
%       ubm = maximization_ubm(mfcc_train, ubm, delta_w_Q_PLDA_, patameters_FA.L, patameters_FA.B, nworkers);

    tim = toc(tim);
    fprintf('[elaps = %.2f s]\n', tim);
end

% N_train = zeros(nmix,size(mfcc_train,2));
% F_train = zeros(SVdim,size(mfcc_train,2));
% % parfor (n_train = 1 : size(mfcc_train,2), nworkers)
% for n_train = 1 : size(mfcc_train,2)
%     [N, F] = expectation(mfcc_train{n_train}, ubm);
%     N_train(:,n_train) = N(:);
%     F_train(:,n_train) = F(:);
% end
% states_train.N = N_train;
% states_train.F = F_train;

N_test = zeros(nmix,size(mfcc_test,2));
F_test = zeros(SVdim,size(mfcc_test,2));
% parfor (n_test = 1 : size(mfcc_test,2), nworkers)
for n_test = 1 : size(mfcc_test,2)
    [N, F] = expectation(mfcc_test{n_test}, ubm);
    N_test(:,n_test) = N(:);
    F_test(:,n_test) = F(:);
end
states_test.N = N_test;
states_test.F = F_test;
    

plda.Phi   = Phi;
plda.Sigma = Sigma;
plda.W     = 1;
plda.M     = mu;


%% Functions for UBM
function [gm, gv] = comp_gm_gv(data)
% computes the global mean and variance of data
nframes = cellfun(@(x) size(x, 2), data, 'UniformOutput', false);
nframes = sum(cell2mat(nframes));
gm = cellfun(@(x) sum(x, 2), data, 'UniformOutput', false);
gm = sum(cell2mat(gm'), 2)/nframes;
gv = cellfun(@(x) sum(bsxfun(@minus, x, gm).^2, 2), data, 'UniformOutput', false);
gv = sum(cell2mat(gv'), 2)/( nframes - 1 );

function gmm = gmm_init(glob_mu, glob_sigma)
% initialize the GMM hyperparameters (Mu, Sigma, and W)
gmm.mu    = glob_mu;
gmm.sigma = glob_sigma;
gmm.w     = 1;

function [N, F, S, llk, post] = expectation(data, gmm)
% compute the sufficient statistics
[post, llk] = postprob(data, gmm.mu, gmm.sigma, gmm.w(:));
N = sum(post, 2)';
F = data * post';
S = (data .* data) * post';

function [post, llk] = postprob(data, mu, sigma, w)
% compute the posterior probability of mixtures for each frame
% post = lgmmprob(data, mu, sigma, w);
post = lgmmprob(data, mu, abs(sigma), w);
llk  = logsumexp(post, 1);
post = exp(bsxfun(@minus, post, llk));
% post_min = min(post(find(post)));
% post(post==0)=post_min;


function logprob = lgmmprob(data, mu, sigma, w)
% compute the log probability of observations given the GMM
ndim = size(data, 1);
% C = sum(mu.*mu./sigma) + sum(log(sigma));
% D = (1./sigma)' * (data .* data) - 2 * (mu./sigma)' * data  + ndim * log(2 * pi);
% logprob = -0.5 * (bsxfun(@plus, C',  D));
% logprob = bsxfun(@plus, logprob, log(w));

C = sum(mu.*mu./sigma) + sum(log(abs(sigma)));
D = (1./sigma)' * (data .* data) - 2 * (mu./sigma)' * data  + ndim * log(2 * pi);
logprob = -0.5 * (bsxfun(@plus, C',  D));
sign_w = sign(w);
logprob = bsxfun(@plus, logprob, sign_w.*log(abs(w)));

function y = logsumexp(x, dim)
% compute log(sum(exp(x),dim)) while avoiding numerical underflow
xmax = max(x, [], dim);
y    = xmax + log(sum(exp(bsxfun(@minus, x, xmax)), dim));
ind  = find(~isfinite(xmax));
if ~isempty(ind)
    y(ind) = xmax(ind);
end

function [r_w, r_mu, r_sigma] = expectation_ubm(mfcc_train, ubm, delta_w_Q_PLDA, L, B)
dim = size(mfcc_train,1);
mix = size(ubm.w,2);
nSamples = size(mfcc_train,2);
% A = L\B;
A = pinv(L)*B;

[post, ~] = postprob(mfcc_train, ubm.mu, ubm.sigma, ubm.w(:));
D = delta_w_Q_PLDA'*A;
mfcc_repmat = repmat(mfcc_train,mix,1);
mfcc_centered = bsxfun(@minus,mfcc_repmat, ubm.mu(:));

post_repmat = reshape(repmat(post(:)',dim,1),mix*dim,nSamples);
D_reshape = reshape(D,dim,mix);

mfcc_p_reshape = reshape(sum(mfcc_centered.*post_repmat,2),dim,mix);
r_w = sum(D_reshape.*mfcc_p_reshape,1);


% mfcc_m_p_reshape = reshape(sum(mfcc_centered.*(mfcc_repmat.*post_repmat),2),dim,mix);
% r_mu = D_reshape.*mfcc_m_p_reshape;
% 
% mfcc_S_p_reshape = reshape(sum(mfcc_centered.*(mfcc_centered.*mfcc_centered.*post_repmat),2),dim,mix);
% r_sigma = D_reshape.*mfcc_S_p_reshape;


% m_w_p = mfcc_centered.*post_repmat;
m_m_p = mfcc_centered*(mfcc_repmat.*post_repmat)';
m_S_p = mfcc_centered*(mfcc_centered.*mfcc_centered.*post_repmat)';

for nMix = 1 : mix
    
%     % weight
%     r_w(:,(nMix-1)*dim+1:nMix*dim) = D(:,(nMix-1)*dim+1:nMix*dim)*m_w_p((nMix-1)*dim+1:nMix*dim,(nMix-1)*dim+1:nMix*dim)';    
    
    % mu
    r_mu(:,(nMix-1)*dim+1:nMix*dim) = D(:,(nMix-1)*dim+1:nMix*dim)*m_m_p((nMix-1)*dim+1:nMix*dim,(nMix-1)*dim+1:nMix*dim)';    

    % sigma
    r_sigma(:,(nMix-1)*dim+1:nMix*dim) = D(:,(nMix-1)*dim+1:nMix*dim)*m_S_p((nMix-1)*dim+1:nMix*dim,(nMix-1)*dim+1:nMix*dim)';

end
% r_w = reshape(r_w(:),1,mix);
r_mu = reshape(r_mu(:),dim,mix);
r_sigma = reshape(r_sigma(:),dim,mix);


function ubm = maximization_ubm(R_w, R_mu, R_sigma)
w = R_w/sum(R_w);
mu = bsxfun(@rdivide, R_mu, R_w);
sigma = bsxfun(@rdivide, R_sigma, R_w) - (mu .* mu);
sigma = apply_var_floors(w, sigma, 0.1);
ubm.w = w;
ubm.mu= mu;
ubm.sigma = abs(sigma);


function ubm = maximization_ubm0(mfcc_train,ubm,delta_w_Q_PLDA,L,B,nworkers)
% Re-estimation of UBM hyperparameters
ubm.w0 = ubm.w;
ubm.mu0 = ubm.mu;
ubm.sigma0 = ubm.sigma; 

R_w = 0;  P = 0;
R_mu = 0;
R_sigma = 0; %R_sigma_ = 0;
dim = size(mfcc_train{1},1);
mix = size(ubm.w,2);
% A = L\B;
A = pinv(L)*B;

% parfor (n_train = 1 : size(mfcc_train,2), nworkers)
for n_train = 1 : size(mfcc_train,2)
    nSamples = size(mfcc_train{n_train},2);
    [post, ~] = postprob(mfcc_train{n_train}, ubm.mu, ubm.sigma, ubm.w(:));
    D = delta_w_Q_PLDA(:,n_train)'*A;
    mfcc_repmat = repmat(mfcc_train{n_train},mix,1);
    mfcc_centered = bsxfun(@minus,mfcc_repmat, ubm.mu(:));
%     mfcc_variances = mfcc_centered.^2;
    
%     r_w = D*mfcc_centered*post';
%     p = sum(r_w);
%     R_w = R_w + r_w;
%     P = P + p;

% %     post_rep = repmat(post,dim,1);
% %     post_repmat = reshape(repmat(post(:)',dim,1),mix*dim,nSamples);
%     r_mu = D*mfcc_centered*(mfcc_repmat.*post_repmat)';    
%     R_mu = R_mu + r_mu;     

% %     r_sigma_ = D*mfcc_centered*(mfcc_variances.*post_rep)';
% %     R_sigma_ = R_sigma_ + r_sigma_;
%     r_sigma = D*mfcc_centered*(mfcc_repmat.*mfcc_repmat.*post_repmat)';
%     R_sigma = R_sigma + r_sigma;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % weight
%     post_reshape = reshape(post,1,mix*nSamples);
%     post_repmat = repmat(post_reshape,dim,1);
%     post_repmat = reshape(post_repmat,dim*mix,nSamples);
%     mfcc_centered_sum = sum(mfcc_centered.*post_repmat,2);
    post_repmat = reshape(repmat(post(:)',dim,1),mix*dim,nSamples);
    mfcc_centered_sum_reshape = reshape(sum(mfcc_centered.*post_repmat,2),dim,mix);
    D_reshape = reshape(D,dim,mix);
    r_w = sum(D_reshape.*mfcc_centered_sum_reshape,1);
    p = sum(r_w);
    R_w = R_w + r_w;
    P = P + p;
    
    m_m_p = mfcc_centered*(mfcc_repmat.*post_repmat)';
    m_S_p = mfcc_centered*(mfcc_centered.*mfcc_centered.*post_repmat)';
    
    for nMix = 1 : mix   
        % mu
        r_mu(:,(nMix-1)*dim+1:nMix*dim) = D(:,(nMix-1)*dim+1:nMix*dim)*m_m_p((nMix-1)*dim+1:nMix*dim,(nMix-1)*dim+1:nMix*dim)';    
        
        % sigma
        r_sigma(:,(nMix-1)*dim+1:nMix*dim) = D(:,(nMix-1)*dim+1:nMix*dim)*m_S_p((nMix-1)*dim+1:nMix*dim,(nMix-1)*dim+1:nMix*dim)';;
        
    end
    R_mu = R_mu + r_mu;
    R_sigma = R_sigma + r_sigma;
    

end
R_mu = reshape(R_mu(:),dim,mix);
% R_sigma_ = reshape(R_sigma_(:),dim,mix);
R_sigma = reshape(R_sigma(:),dim,mix);


w_new = R_w/P;
mu_new = bsxfun(@rdivide, R_mu, R_w);
 sigma_new = bsxfun(@rdivide, R_sigma, R_w);
% sigma_new_ = bsxfun(@rdivide, R_sigma_, R_w);
% sigma_new = bsxfun(@rdivide, R_sigma, R_w)-mu_new.*mu_new;
% sigma_new = apply_var_floors(w_new, sigma_new, 0.1); 

ubm.w = w_new;
ubm.mu = mu_new;
ubm.sigma = abs(sigma_new); 



function gmm = maximization(N, F, S)
% ML re-estimation of GMM hyperparameters which are updated from accumulators
w  = N / sum(N);
mu = bsxfun(@rdivide, F, N);
sigma = bsxfun(@rdivide, S, N) - (mu .* mu);
% sigma = apply_var_floors(w, sigma, 0.1);
gmm.w = w;
gmm.mu= mu;
gmm.sigma = sigma;

function sigma = apply_var_floors(w, sigma, floor_const)
% set a floor on covariances based on a weighted average of component
% variances
vFloor = sigma * w' * floor_const;
sigma  = bsxfun(@max, sigma, vFloor);
% sigma = bsxfun(@plus, sigma, 1e-6 * ones(size(sigma, 1), 1));

function gmm = gmm_mixup(gmm)
% perform a binary split of the GMM hyperparameters
mu = gmm.mu; sigma = gmm.sigma; w = gmm.w;
[ndim, nmix] = size(sigma);
[sig_max, arg_max] = max(sigma);
eps = sparse(0 * mu);
eps(sub2ind([ndim, nmix], arg_max, 1 : nmix)) = sqrt(sig_max);
% only perturb means associated with the max std along each dim 
mu = [mu - eps, mu + eps];
% mu = [mu - 0.2 * eps, mu + 0.2 * eps]; % HTK style
sigma = [sigma, sigma];
w = [w, w] * 0.5;
gmm.w  = w;
gmm.mu = mu;
gmm.sigma = sigma;

%% Functions for PLDA
function [Ey, Eyy] = expectation_plda(data, Phi, Sigma, spk_counts)
% computes the posterior mean and covariance of the factors
nphi     = size(Phi, 2);
nsamples = size(data, 2);
nspks    = size(spk_counts, 1);

Ey  = zeros(nphi, nsamples);
Eyy = zeros(nphi);

% initialize common terms to save computations
uniqFreqs  	  = unique(spk_counts);
nuniq 		  = size(uniqFreqs, 1);
invTerms      = cell(nuniq, 1);
invTerms(:)   = {zeros(nphi)};
% PhiT_invS_Phi = ( Phi'/Sigma ) * Phi;
PhiT_invS_Phi = ( Phi'*pinv(Sigma) ) * Phi;
I = eye(nphi);
for ix = 1 : nuniq
    nPhiT_invS_Phi = uniqFreqs(ix) * PhiT_invS_Phi;
    Cyy =  pinv(I + nPhiT_invS_Phi);
    invTerms{ix} = Cyy;
end

% data = Sigma\data;
data = pinv(Sigma)*data;
cnt  = 1;
for spk = 1 : nspks
    nsessions = spk_counts(spk);
    % Speaker indices
    idx = cnt : ( cnt - 1 ) + spk_counts(spk);
    cnt  = cnt + spk_counts(spk);
    Data = data(:, idx);
    PhiT_invS_y = sum(Phi' * Data, 2);
    Cyy = invTerms{ uniqFreqs == nsessions };
    Ey_spk  = Cyy * PhiT_invS_y;
    Eyy_spk = Cyy + Ey_spk * Ey_spk';
    Eyy     = Eyy + nsessions * Eyy_spk;
    Ey(:, idx) = repmat(Ey_spk, 1, nsessions);
end

function [Phi, Sigma] = maximization_plda(data, Ey, Eyy)
% ML re-estimation of the Eignevoice subspace and the covariance of the
% residual noise (full).
nsamples = size(data, 2);
Data_sqr = data * data';
Phi      = data * Ey' * pinv(Eyy);
Sigma    = 1/nsamples * (Data_sqr - (Phi * Ey) * data');

function cost = ComputeCost(data,Ey,Eyy,Phi,Sigma,nSamples)
    a1 = data'/Sigma;
    a2 = reshape(a1',1,size(a1,1)*size(a1,2));
    a3 = a2*reshape(data',size(data,1)*size(data,2),1);
    b1 = a1*Phi;
    b2 = reshape(b1',1,size(b1,1)*size(b1,2));
    b3 = b2*reshape(Ey',size(Ey,1)*size(Ey,2),1);
    c3 = trace(0.5*Phi'/Sigma*Phi*Eyy);
    cost = -0.5*log(det(Sigma)) - (a3 - b3)/nSamples - c3/nSamples;



