function [W,H,cost] = snmf(V, d, varargin)
% SNMF Sparse non-negative matrix factorization with adaptive mult. updates
% 
% Usage:
%   [W,H] = snmf(V,d,[options])
%
% Input:
%   V                 M x N data matrix
%   d                 Number of factors
%   options
%     .costfcn        Cost function to optimize
%                       'ls': Least squares (default)
%                       'kl': Kullback Leibler
%     .W              Initial W, array of size M x d
%     .H              Initial H, array of size d x N 
%     .lambda         Sparsity weight on H
%     .updateW        Update W [<on> | off]
%     .maxiter        Maximum number of iterations (default 100)
%     .conv_criteria  Function exits when cost/delta_cost exceeds this
%     .plotfcn        Function handle for plot function
%     .plotiter       Plot only every i'th iteration
%     .accel          Wild driver accelleration parameter (default 1)
%     .displaylevel   Level of display: [off | final | <iter>]
% 
% Output:
%   W                 M x d
%   H                 d x N
%
% Example I, Standard NMF:
%   d = 4;                                % Four components
%   [W,H] = snmf(V,d);
%
% Example I, Sparse NMF:
%   d = 2;                                % Two components
%   opts.costfcn = 'kl';                  % Kullback Leibler cost function
%   opts.lambda = 0.1;                   % Sparsity
%   [W,H] = snmf(V,d,opts);
% 
% Authors:
%   Mikkel N. Schmidt and Morten Mørup, 
%   Technical University of Denmark, 
%   Institute for Matematical Modelling
%
% References:
%   [1] M. Mørup and M. N. Schmidt. Sparse non-negative matrix factor 2-D 
%       deconvolution. Technical University of Denmark, 2006.
%   [2] M. N. Schmidt and M. Mørup. Nonnegative matrix factor 2-D 
%       deconvolution for blind single channel source separation. 
%       ICA, 2006.
%   [3] M. N. Schmidt and M. Mørup. Sparse non-negative matrix factor 2-d 
%       deconvolution for automatic transcription of polyphonic music. 
%       Submitted to EURASIP Journal on Applied Signal Processing, 2006.
%   [4] P. Smaragdis. Non-negative matrix factor deconvolution; 
%       extraction of multiple sound sourses from monophonic inputs. 
%       ICA 2004.
%   [5] J. Eggert and E. Korner. Sparse coding and nmf. In Neural Networks,
%       volume 4, 2004.
%   [6] J. Eggert, H. Wersing, and E. Korner. Transformation-invariant 
%       representation and nmf. In Neural Networks, volume 4, 2004.




% -------------------------------------------------------------------------
% Parse input arguments
if nargin>=3, opts = mgetopt(varargin); else opts = struct; end
costfcn = mgetopt(opts, 'costfcn', 'ls', 'instrset', {'ls','kl'});
W = mgetopt(opts, 'W', rand(size(V,1),d));
H = mgetopt(opts, 'H', rand(d,size(V,2)));
W = normalizeW(W);
updateW = mgetopt(opts, 'updateW', 'on', 'instrset', {'on','off'});
nuH = mgetopt(opts, 'nuH', 1);
nuW = mgetopt(opts, 'nuW', 1);
lambda = mgetopt(opts, 'lambda', 0);
maxiter = mgetopt(opts, 'maxiter', 100);
conv_criteria = mgetopt(opts, 'conv_criteria', 1e-4);
accel = mgetopt(opts, 'accel', 1);
plotfcn = mgetopt(opts, 'plotfcn', []);
plotiter = mgetopt(opts, 'plotiter', 1);
displaylevel = mgetopt(opts, 'displaylevel', 'iter', 'instrset', ...
    {'off','iter','final'});
updateWRows = mgetopt(opts, 'updateWRows', []);



% -------------------------------------------------------------------------
% Initialization
sst = sum(sum((V-mean(mean(V(:)))).^2));
Rec = W*H; 
switch costfcn
    case 'ls'
        sse = norm(V-Rec,'fro')^2;
        cost_old = .5*sse + lambda*(sum(abs(H(:))));
    case 'kl'
        ckl = sum(sum(V.*log((V+eps)./(Rec+eps))-V+Rec));
        cost_old = ckl + lambda*(sum(abs(H(:))));
end
delta_cost = 1;
iter = 0;
keepgoing = 1;



% -------------------------------------------------------------------------
% Display information
dheader = sprintf('%12s | %12s | %12s | %12s | %12s | %12s','Iteration','Expl. var.','NuW','NuH','Cost func.','Delta costf.');
dline = sprintf('-------------+--------------+--------------+--------------+--------------+--------------');
if any(strcmp(displaylevel, {'final','iter'}))
    disp('Sparse Non-negative Matrix Factorization');
    disp(['To stop algorithm press ctrl-C'])
    disp('');
end



% -------------------------------------------------------------------------
% Optimization loop
while keepgoing
    pause(0);
    
    if mod(iter,10)==0
        if any(strcmp(displaylevel, {'iter'}))
            disp(dline); disp(dheader); disp(dline);
        end
    end
    
    % Call plotfunction if specified
    if ~isempty(plotfcn) && mod(iter,plotiter)==0
        plotfcn(V,W,H,d); 
    end   
    
    % Update H and W
    switch costfcn
        case 'ls'
            [W,H,nuH,nuW,cost,sse,Rec] = ...
                ls_update(V,W,H,d,nuH,nuW,cost_old,accel,lambda,Rec,updateW);
        case 'kl'
            [W,H,nuH,nuW,cost,sse,Rec] = ...
                kl_update(V,W,H,d,nuH,nuW,cost_old,accel,lambda,Rec,updateW);
    end

    delta_cost = cost_old - cost;
    cost_old=cost;
    iter=iter+1;

    % Display information
    if any(strcmp(displaylevel, {'iter'}))
        disp(sprintf('%12.0f | %12.4f | %12.6f | %12.6f |  %11.2f | %11.2f', ...
            iter,(sst-sse)/sst,nuW/accel,nuH/accel,cost,delta_cost));
    end
    
    % Check if we should stop
    if delta_cost<cost*conv_criteria 
        % Small improvement with small step-size
        if nuH<=accel & nuW<=accel
            if any(strcmp(displaylevel, {'iter','final'}))
                disp('C2NMF has converged');
            end
            keepgoing = 0;
        % Small improvement - maybe because of too large step-size?
        else
            nuH = 1; nuW = 1;
        end
    end
    % Reached maximum number of iterations
    if iter>=maxiter
        if any(strcmp(displaylevel, {'iter','final'}))
            disp('Maximum number of iterations reached');
        end
        keepgoing=0; 
    end
end



% -------------------------------------------------------------------------
% Least squares update function
function [W,H,nuH,nuW,cost,sse,Rec] = ...
    ls_update(V,W,H,d,nuH,nuW,cost_old,accel,lambda,Rec,updateW)

% Update H

if accel>1
    H_old=H;
    grad = (W'*V)./(W'*Rec+eps+lambda);
    while 1
        H = H_old.*(grad.^nuH);
        Rec = W*H; 
        sse = norm(V-Rec,'fro')^2;
        cost = .5*sse+lambda*sum(H(:));
        if cost>cost_old, nuH = max(nuH/2,1);
        else nuH = nuH*accel; break; end
        pause(0);
    end
    cost_old = cost;
else
    H = H.*(W'*V)./(W'*Rec+eps+lambda);
    Rec = W*H; 
    sse = norm(V-Rec,'fro')^2;
    cost = .5*sse+lambda*sum(H(:));
    cost_old = cost;
end

% Update W
if ~strcmp(updateW, 'off')
    if accel>1
        W_old=W;
        Wxa = V*H';
        Wya = Rec*H';
        Wx = Wxa + repmat(sum(Wya.*W,1),[size(W,1),1]).*W;
        Wy = Wya + repmat(sum(Wxa.*W,1),[size(W,1),1]).*W;
        grad = Wx./(Wy+eps);
        while 1
            W = normalizeW(W_old.*(grad.^nuW));
            Rec = W*H; 
            sse = norm(V-Rec,'fro')^2;
            cost = .5*sse+lambda*sum(H(:));
            if cost>cost_old, nuW = max(nuW/2,1);
            else nuW = nuW*accel; break; end
            pause(0);
        end
        cost_old = cost;
    else
        Wxa = V*H';
        Wya = Rec*H';
        Wx = Wxa + repmat(sum(Wya.*W,1),[size(W,1),1]).*W;
        Wy = Wya + repmat(sum(Wxa.*W,1),[size(W,1),1]).*W;
        W = normalizeW(W.*Wx./(Wy+eps));
        Rec = W*H; 
        sse = norm(V-Rec,'fro')^2;
        cost = .5*sse+lambda*sum(H(:));
        cost_old = cost;
    end
end

% -------------------------------------------------------------------------
% Kullback Leibler update function
function [W,H,nuH,nuW,cost,sse,Rec] = ...
    kl_update(V,W,H,d,nuH,nuW,cost_old,accel,lambda,Rec,updateW)

% Update H
if accel>1
    H_old=H;
    VR = V./(Rec+eps);
    O = ones(size(V));
    grad = (W'*VR)./(W'*O+eps+lambda);
    while 1
        H = H_old.*(grad.^nuH);
        Rec = W*H; 
        ckl = sum(sum(V.*log((V+eps)./(Rec+eps))-V+Rec));
        cost = ckl + lambda*(sum(abs(H(:))));
        if cost>cost_old, nuH = max(nuH/2,1);
        else nuH = nuH*accel; break; end
        pause(0);
    end
    cost_old = cost;
else
    H = H.*(W'*(V./(Rec+eps)))./(W'*ones(size(V))+eps+lambda);
    Rec = W*H; 
    ckl = sum(sum(V.*log((V+eps)./(Rec+eps))-V+Rec));
    cost = ckl + lambda*(sum(abs(H(:))));
    cost_old = cost;
end

% Update W
if ~strcmp(updateW, 'off')
    if accel>1
        W_old=W;
        Wxa = (V./(Rec+eps))*H';
        Wya = ones(size(V))*H';
        Wx = Wxa + repmat(sum(Wya.*W,1),[size(W,1),1]).*W;
        Wy = Wya + repmat(sum(Wxa.*W,1),[size(W,1),1]).*W;
        grad = Wx./(Wy+eps);
        while 1
            W = normalizeW(W_old.*(grad.^nuW));
            Rec = W*H; 
            ckl = sum(sum(V.*log((V+eps)./(Rec+eps))-V+Rec));
            cost = ckl + lambda*(sum(abs(H(:))));
            if cost>cost_old, nuW = max(nuW/2,1);
            else nuW = nuW*accel; break; end
            pause(0);
        end
        cost_old = cost;
    else
        Wxa = (V./(Rec+eps))*H';
        Wya = ones(size(V))*H';
        Wx = Wxa + repmat(sum(Wya.*W,1),[size(W,1),1]).*W;
        Wy = Wya + repmat(sum(Wxa.*W,1),[size(W,1),1]).*W;
        W = normalizeW(W.*Wx./(Wy+eps));
        Rec = W*H; 
        ckl = sum(sum(V.*log((V+eps)./(Rec+eps))-V+Rec));
        cost = ckl + lambda*(sum(abs(H(:))));
        cost_old = cost;
    end
end
sse = norm(V-Rec,'fro')^2;

% -------------------------------------------------------------------------
% Normalize W
function W = normalizeW(W)
Q(1,:,1) = sqrt(sum(W.^2,1));
W = W./repmat(Q+eps,[size(W,1),1]);


% -------------------------------------------------------------------------
% Parser for optional arguments
function out = mgetopt(varargin)
% MGETOPT Parser for optional arguments
% 
% Usage
%   Get a parameter structure from 'varargin'
%     opts = mgetopt(varargin);
%
%   Get and parse a parameter:
%     var = mgetopt(opts, varname, default);
%        opts:    parameter structure
%        varname: name of variable
%        default: default value if variable is not set
%
%     var = mgetopt(opts, varname, default, command, argument);
%        command, argument:
%          String in set:
%          'instrset', {'str1', 'str2', ... }
%
% Example
%    function y = myfun(x, varargin)
%    ...
%    opts = mgetopt(varargin);
%    parm1 = mgetopt(opts, 'parm1', 0)
%    ...

% Copyright 2007 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk

if nargin==1
    if isempty(varargin{1})
        out = struct;
    elseif isstruct(varargin{1})
        out = varargin{1}{:};
    elseif isstruct(varargin{1}{1})
        out = varargin{1}{1};
    else
        out = cell2struct(varargin{1}(2:2:end),varargin{1}(1:2:end),2);
    end
elseif nargin>=3
    opts = varargin{1};
    varname = varargin{2};
    default = varargin{3};
    validation = varargin(4:end);
    if isfield(opts, varname)
        out = opts.(varname);
    else
        out = default;
    end
    
    for narg = 1:2:length(validation)
        cmd = validation{narg};
        arg = validation{narg+1};
        switch cmd
            case 'instrset',
                if ~any(strcmp(arg, out))
                    fprintf(['Wrong argument %s = ''%s'' - ', ...
                        'Using default : %s = ''%s''\n'], ...
                        varname, out, varname, default);
                    out = default;
                end
            case 'dim'
                if ~all(size(out)==arg)
                    fprintf(['Wrong argument dimension: %s - ', ...
                        'Using default.\n'], ...
                        varname);
                    out = default;
                end
            otherwise,
                error('Wrong option: %s.', cmd);
        end
    end
end
