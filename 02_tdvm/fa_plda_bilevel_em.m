function [patameters_FA, plda] = fa_plda_bilevel_em(M_train, ubm, num, patameters_FA0, plda0)
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
% References:
%   [1] S.J.D. Prince and J.H. Elder, "Probabilistic linear discriminant analysis
%       for inferences about identity," in Proc. IEEE ICCV, Rio de Janeiro, Brazil,
%       Oct. 2007.
%   [2] D. Garcia-Romero and C.Y. Espy-Wilson, "Analysis of i-vector length 
%       normalization in speaker recognition systems," in Proc. INTERSPEECH,
%       Florence, Italy, Aug. 2011, pp. 249-252.
%   [3] P. Kenny, "Bayesian speaker verification with heavy-tailed priors," 
%       in Proc. Odyssey, The Speaker and Language Recognition Workshop, Brno, 
%       Czech Republic, Jun. 2010.
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

ndim = num.IVdim;
spk_labs = num.label_dev;
nphi = num.Zdim;
niter = num.nIters;
nSamples = num.samples;
cost = zeros(niter,1);

% % make sure the labels are sorted
% [spk_labs, Inx] = sort(spk_labs);
% M_train = M_train(:, Inx);
[~, ia, ic] = unique(spk_labs, 'stable');
spk_counts = histc(ic, 1 : numel(ia)); % # sessions per speaker

% % % M = mean(data, 2);
% % % data = bsxfun(@minus, data, M); % centering the data
% % % % data = length_norm(data); % normalizing the length
% % % W1   = 1;% calc_white_mat(cov(data'));
% % % data = W1' * data; % whitening the data

if nargin > 3 
    patameters_FA = patameters_FA0;
else
    fprintf('Randomly initializing the FA hyperparameters ...\n');
    % initializing FA parameters
%     patameters_FA.m = reshape(ubm.mu, num.SVdim, 1);
    patameters_FA.m = mean(M_train,2);
    centeredM = bsxfun(@minus, M_train, patameters_FA.m);
%     variancesM = sum(patameters_FA.centeredM.^2, 2) / num.samples;

    patameters_FA.I = eye(num.IVdim);
    patameters_FA.T = M_train(:,1:num.IVdim)-patameters_FA.m;   %rand(1024*39, Fac);
%     patameters_FA.Sigma = reshape(ubm.sigma, num.SVdim, 1); %ones(1024*39,1);
    patameters_FA.Sigma = sum(centeredM.^2, 2) / num.samples;
end

if nargin > 4
    Phi   = plda0.Phi;
    Sigma = plda0.Sigma;
    mu    = plda0.M;
else
    fprintf('Randomly initializing the PLDA hyperparameters ...\n');
    % Initialize the parameters randomly
    [s1, s2] = RandStream.create('mrg32k3a', 'NumStreams', 2);
    Sigma    = 100 * randn(s1, ndim); % covariance matrix of the residual term
    Phi = randn(s2, ndim, nphi); % factor loading matrix (Eignevoice matrix)
    Phi = bsxfun(@minus, Phi, mean(Phi, 2));
    W2   = calc_white_mat(Phi' * Phi);
    Phi = Phi * W2; % orthogonalize Eigenvoices (columns)
end

% fprintf('Re-estimating the Eigenvoice subspace with %d factors ...\n', nphi);
for iter = 1 : niter
    fprintf('EM iter#: %d \t', iter);
    tim = tic;
    
    %% Lower level for FA    
    % E step
    centeredM = bsxfun(@minus, M_train, patameters_FA.m);
    patameters_FA.B = bsxfun(@rdivide, patameters_FA.T', patameters_FA.Sigma');
    patameters_FA.L = patameters_FA.I + patameters_FA.B*patameters_FA.T;
    Ex = patameters_FA.L\patameters_FA.B*centeredM;
%     Ex = pinv(patameters_FA.L)*patameters_FA.B*patameters_FA.centeredM;
%     Exx = Ex*Ex' + num.samples*pinv(patameters_FA.L);
    
%     % M step
%     patameters_FA.T = (patameters_FA.centeredM*Ex')*inv(Exx);
%     patameters_FA.Sigma = patameters_FA.variancesM - sum(patameters_FA.T .* (patameters_FA.T * Exx), 2) / num.samples;   
    
    mu = mean(Ex, 2);
    data = bsxfun(@minus, Ex, mu); % centering the data
    data0 = data;
    [data, sigma] = length_norm(data); % normalizing the length
    W1   = 1;%calc_white_mat(cov(data'));
    data = W1' * data; % whitening the data
    
    %% Upper level for PLDA 
    i_PLDA = 20;
    for iter_PLDA = 1 : i_PLDA
        % E step
        [Ey, Eyy] = expectation_plda(data, Phi, Sigma, spk_counts);
        % M step for PLDA
        [Phi, Sigma] = maximization_plda(data, Ey, Eyy);    
    end
    clear Eyy
    
    % M step for FA
%     delta_w_Q_PLDA = -pinv(Sigma)*(data-Phi*Ey);
%     inv_delta_w_Q_PLDA_wT = pinv(delta_w_Q_PLDA*data');
%     T = M_train*delta_w_Q_PLDA'*inv_delta_w_Q_PLDA_wT*pinv(patameters_FA.I+data*delta_w_Q_PLDA'*inv_delta_w_Q_PLDA_wT);
%     patameters_FA.T = T;

    % M step for FA
% %     delta_w_Q_PLDA = -Sigma\(data-Phi*Ey);
% %     inv_delta_w_Q_PLDA_wT = inv(delta_w_Q_PLDA*data');
%     delta_w_Q_PLDA = -pinv(Sigma)*(data-Phi*Ey);
%     inv_delta_w_Q_PLDA_wT = pinv(delta_w_Q_PLDA*data');
%     T = centeredM*delta_w_Q_PLDA'*inv_delta_w_Q_PLDA_wT*pinv(patameters_FA.I+data*delta_w_Q_PLDA'*inv_delta_w_Q_PLDA_wT);
%     patameters_FA.T = T;
%     patameters_FA.B = bsxfun(@rdivide, patameters_FA.T', patameters_FA.Sigma');
%     patameters_FA.L = patameters_FA.I + patameters_FA.B*patameters_FA.T;
%     delta_w_Q_PLDA_ = delta_w_Q_PLDA;
%     clear Ex data Ey Eyy


    delta_w_Q_PLDA = -Sigma\(data-Phi*Ey);
    A1 = bsxfun(@rdivide, delta_w_Q_PLDA, sigma);
    A2_1 = sum(bsxfun(@times, delta_w_Q_PLDA, data0));
    A2_2 = bsxfun(@times, A2_1, data0);
    A2 = bsxfun(@rdivide, A2_2, sigma.*sigma);
    delta_w_Q_PLDA_ =  A1 - A2;
    Exx_new = Ex*delta_w_Q_PLDA_' + delta_w_Q_PLDA_*Ex';
    T = centeredM*delta_w_Q_PLDA_'/(Exx_new);  % the best
    patameters_FA.T = T;
    patameters_FA.B = bsxfun(@rdivide, patameters_FA.T', patameters_FA.Sigma');
    patameters_FA.L = patameters_FA.I + patameters_FA.B*patameters_FA.T;


%     cov_w_delta_w_Q_PLDA = (bsxfun(@minus, Ex, mu))*(delta_w_Q_PLDA-mean(delta_w_Q_PLDA,2))'; 
%     T0 = patameters_FA.centeredM*delta_w_Q_PLDA'/(data*delta_w_Q_PLDA' + delta_w_Q_PLDA*data');  % the second best

%     T = patameters_FA.centeredM*delta_w_Q_PLDA'/(data*delta_w_Q_PLDA' + delta_w_Q_PLDA*data' + cov_w_delta_w_Q_PLDA + cov_w_delta_w_Q_PLDA');
%     T = patameters_FA.centeredM*delta_w_Q_PLDA'/(Ex*delta_w_Q_PLDA'+delta_w_Q_PLDA*Ex' + cov_w_delta_w_Q_PLDA + cov_w_delta_w_Q_PLDA');
    
    clear Ey data data0 sigma
    clear delta_w_Q_PLDA A1 A2_1 A2_2 A2 delta_w_Q_PLDA_ Exx_new T

%     patameters_FA.Sigma = patameters_FA.variancesM - sum(patameters_FA.T .* (patameters_FA.T * Exx_new/2), 2) / num.samples;
    
    
%     inv_delta_w_Q_PLDA_wT = inv(delta_w_Q_PLDA*data');
%     T0 = M_train*delta_w_Q_PLDA'*inv_delta_w_Q_PLDA_wT/(patameters_FA.I+data*delta_w_Q_PLDA'*inv_delta_w_Q_PLDA_wT);
%     patameters_FA.T = T;
    
%     A = bsxfun(@rdivide, patameters_FA.T', patameters_FA.Sigma');
%     A = A*patameters_FA.T - patameters_FA.L;
%     delta_w_Q_PLDA = -Sigma\(data-Phi*Ey);
% %     delta_w_Q_PLDA_wT = delta_w_Q_PLDA*data';
%     T = patameters_FA.centeredM*delta_w_Q_PLDA'/(A\delta_w_Q_PLDA*data'*A + data*delta_w_Q_PLDA');
%     patameters_FA.T = T;

    
%     patameters_FA.T = (patameters_FA.centeredM*Ex')*inv(Exx);
%     patameters_FA.Sigma = patameters_FA.variancesM - sum(patameters_FA.T .* (patameters_FA.T * Exx), 2) / num.samples;



%     cost(iter) = ComputeCost(data,Ey,Eyy,Phi,Sigma,nSamples);
    
    tim = toc(tim);
    fprintf('[elaps = %.2f s]\n', tim);
end

% Re-estimation for the PLDA
patameters_FA.B = bsxfun(@rdivide, patameters_FA.T', patameters_FA.Sigma');
patameters_FA.L = patameters_FA.I + patameters_FA.B*patameters_FA.T;
Ex = patameters_FA.L\patameters_FA.B*centeredM;

mu = mean(Ex, 2);
data = bsxfun(@minus, Ex, mu); % centering the data
[data, ~] = length_norm(data); % normalizing the length
W1   = 1;%calc_white_mat(cov(data'));
data = W1' * data; % whitening the data

%% Upper level for PLDA    
for iter_PLDA = 1 : 30
    % E step
    [Ey, Eyy] = expectation_plda(data, Phi, Sigma, spk_counts);
    % M step for PLDA
    [Phi, Sigma] = maximization_plda(data, Ey, Eyy);    
end

plda.Phi   = Phi;
plda.Sigma = Sigma;
plda.W     = W1;
plda.M     = mu;

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
PhiT_invS_Phi = ( Phi'/Sigma ) * Phi;
% PhiT_invS_Phi = ( Phi'*pinv(Sigma) ) * Phi;
I = eye(nphi);
for ix = 1 : nuniq
    nPhiT_invS_Phi = uniqFreqs(ix) * PhiT_invS_Phi;
    Cyy =  pinv(I + nPhiT_invS_Phi);
    invTerms{ix} = Cyy;
end

data = Sigma\data;
% data = pinv(Sigma)*data;
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
Phi      = data * Ey' / (Eyy);
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



