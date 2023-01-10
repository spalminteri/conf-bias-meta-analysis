% This code finds the optimal parameters
% This code works for the partial feedback experiments
% This code requires the Optimization toolbox

clear all
close all

%% Defining the dataset
% list of the datasets relevant for this code
load C2020d % Chambon 2020 (GoNogo experiment)                              C4 in the paper
% load L2017a % Lefebvre 2017 (fMRI experiment)                             L1 in the paper 
% load L2017b % Lefebvre 2017 (Lab experiment)                              L2 in the paper 
% load P2017a % Palminteri 2017 (Partial feedback experiment                P1 in the paper 


%% what is in the data files
% con: condition number (it depends on the study, conditions are caracterized typically by different contingencies. contingencies are
% repeater by session (when more than one)
% sta: state or pair of symbols.
% cho: choices 1 or 2 (in the option space, not the motor space)
% out: obtained outcome (-1 or 1)

subjecttot=numel(con); % this get ther number of subjects, by analysing how many elements there are in one of the variables

%%


n_model=2; % because we are fitting two models

nfpm=[3 5]; % Confirmation model has 3 free parameters, the full model 5 free parameters

% fmincon settings (modified from default to include more iterations and function evaluations

options = optimset('Algorithm', 'interior-point', 'Display', 'iter-detailed', 'MaxIter', 10000,'MaxFunEval',10000);

nsub=0;

% subject loop
for k_sub = 1:subjecttot
    
    % model loop
    for k_model = 1:n_model
        
        % prepare starting points and parameter bounds (Courtesy of Maël Lebreton) 
        
        if     k_model == 1  % confirmation
            lb = [0 0 0];        LB = [0 0 0];   % lower bounds xb=to generated starting points / XB=true
            ub = [15 1 1];       UB = [Inf 1 1]; % upper bounds xb=to generated starting points / XB=true
        elseif k_model == 2  % full model
            lb = [0 0 0 0 -1];       LB = [0 0 0 0 -Inf]; % lower bounds xb=to generated starting points / XB=true
            ub = [15 1 1 1 1];       UB = [Inf 1 1 1 Inf]; % upper bounds xb=to generated starting points / XB=true
        end
        
        ddb = ub - lb; % where to look for the random point initialization
        
        % prepare multiple starting points for estimation and the temporary
        % matrices to be filled 
        n_rep           = 5;
        parameters_rep  = NaN(n_rep,nfpm(k_model));     parametersLPP_rep  = NaN(n_rep,nfpm(k_model));
        ll_rep          = NaN(n_rep,1);                 LPP_rep            = NaN(n_rep,1);
        FminHess        = NaN(n_rep,nfpm(k_model),nfpm(k_model));
        
        for k_rep = 1:n_rep
            % prepare starting points and parameter bounds
            x0 = lb + rand(1,length(lb)).*ddb; %  generate the random initiamization (lower bound + something randomly drawn within its range) 
            x0 = x0(1:nfpm(k_model));
            
            
            % run  MAP (maximum a posteriori) estimations
            [parametersLPP_rep(k_rep,1:nfpm(k_model)),LPP_rep(k_rep),~,~,~,~,FminHess(k_rep,:,:)]=fmincon(@(x) Priors_Partial_Final(x,sta{k_sub},cho{k_sub},out{k_sub},0,k_model),x0,[],[],[],[],LB,UB,[],options);
        end
        
        % find best params over repetitions & store optimization outputs
        
        [~,posLPP]                                      = min(LPP_rep);
        parametersLPP(k_sub,k_model,1:nfpm(k_model))    = parametersLPP_rep(posLPP(1),1:nfpm(k_model));
        LPP(k_sub,k_model)                              = LPP_rep(posLPP(1),:) - nfpm(k_model)*log(2*pi)/2 + real(log(det(squeeze(FminHess(posLPP(1),:,:)))))/2;
        
        check_conv(k_sub)                               =  ~any(eig(squeeze(FminHess(posLPP(1),:,:)))<0);
        
        
    end
end

%%
for k_sub = 1:subjecttot
    [~ , rankedLPP(k_sub,:)]=sort(LPP(k_sub,:));
    
    diffLPP(k_sub,:)=(LPP(k_sub,1)-LPP(k_sub,:))./LPP(k_sub,1);
    
end

rankedLPP=11-rankedLPP;



%% Priors_Partial_Final

% this function calculate parameter log(probabilities) used to calculate  the MP
% prior distributions are based on Daw (Neuon 2011)

function [post]=Priors_Partial_Final(params,s,a,r,c,model)


    % confirmation model
if model ==1
    beta  = params(1); pbeta = log(gampdf(beta,1.2,5.0)); 
    lr1   = params(2); plr1  = log(betapdf(lr1,1.1,1.1));
    lr2   = params(3); plr2  = log(betapdf(lr2,1.1,1.1));
    p = [pbeta plr1 plr2];
   
    % full model
elseif model == 2
    beta  = params(1); pbeta = log(gampdf(beta,1.2,5.0)); 
    lr1   = params(2); plr1  = log(betapdf(lr1,1.1,1.1));
    lr2   = params(3); plr2  = log(betapdf(lr2,1.1,1.1));
    tau   = params(4); ptau  = log(betapdf(tau,1.1,1.1));
    phi   = params(5); pphi  = log(normpdf(phi,0,1));
    p = [pbeta plr1 plr2 ptau pphi];
    
    
end

p = -sum(p);


l=Models_Partial_Final(params,s,a,r,c,model); % calling the computational models 


post = p + l; % the MAP

end

%% Models_Partial_Final
% this function calculate the lilekihood of the models given a set of parameters 

function lik = Models_Partial_Final(params,s,a,r,c,model)

% no bias


% confirmation bias
if model == 1
    beta  = params(1); % choice inverse temperature
    lr1   = params(2); % confirmatory learning rate
    lr2   = params(3); % disconfirmatory learning rate
    
    % full model
elseif model == 2
    beta  = params(1); % choice inverse temperature
    lr1   = params(2); % confirmatory learning rate
    lr2   = params(3); % disconfirmatory learning rate
    tau   = params(4); % choice accumuation rate
    phi   = params(5); % choice decision bias
    
end




% initializing the hidden values

Q       = zeros(8,2); %  Q-values
C       = zeros(8,2); %  C-traces

lik=0;

for i = 1:length(a)
    
    if (a(i))~=1.5 % to exclude missed reponses
        
        
        %% confirmation bias
        if model==1
            
            lik = lik + log (1/(1+ exp(-beta*(Q(s(i),a(i))-Q(s(i),3-a(i)))))); % likelihood of the observed choice
            
            PEc =  r(i) - Q(s(i),a(i)); % prediction error for the chosen outcome
            
            Q(s(i),a(i)) = Q(s(i),a(i)) + lr1 * PEc * (PEc>0) +  lr2 * PEc * (PEc<0); % updated
            
            
            %% full model
        elseif model ==2 %
            
            lik = lik + log (1/(1+ exp(-beta*(Q(s(i),a(i))-Q(s(i),3-a(i))) - phi*(C(s(i),a(i))-C(s(i),3-a(i))))));
            
            PEc =  r(i) - Q(s(i),a(i));
            
            Q(s(i),a(i)) = Q(s(i),a(i)) + lr1 * PEc * (PEc>0) +  lr2 * PEc * (PEc<0);
            
            C(s(i),a(i)) = C(s(i),a(i)) + tau * (1 - C(s(i),a(i))); % increasing the chosen option choice trace
            
            C(s(i),3-a(i)) = C(s(i),3-a(i)) + tau * (0 - C(s(i),3-a(i))); % decreasing the unchosen option choice trace
            

        end
        
    end
end

lik = -lik;     % log likelihood vector 

end


