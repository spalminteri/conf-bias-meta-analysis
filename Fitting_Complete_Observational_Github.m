% This code finds the optimal parameters
% This code works for the complete feedback experiments with observational
% trial 
% This code requires the Optimization toolbox
% see Fitting_Partial_Final_GitHub.m for more edited version

clear all
close all


%% Population definition

load C2020b % Chambon 2020 (observational complete)   C2 in the paper

%% what is in the data files
% con: condition number (it depends on the study, conditions are caracterized typically by different contingencies. contingencies are
% repeater by session (when more than one)
% sta: state or pair of symbols.
% cho: choices 1 or 2 (in the option space, not the motor space)
% out: obtained outcome (-1 or 1)
% cou: the forgone outcome (-1 or 1)
% obs: whether (0) or not (1) the trial was observational (i.e., forced choice)

subjecttot=numel(con);



n_model=2;

nfpm=[4 6];  % attention, now both models have an additional free parameter (the observational learning rate)

options = optimset('Algorithm', 'interior-point', 'Display', 'iter-detailed', 'MaxIter', 10000,'MaxFunEval',10000);

nsub=0;

    % subject loop
for k_sub = 1:subjecttot
   
        % estimate models
    for   k_model = 1:n_model
        
        % prepare starting points and parameter bounds
        if     k_model == 1  % confirmation
            lb = [0 0 0 0];        LB = [0 0 0 0];   % lower bounds
            ub = [15 1 1 1];       UB = [Inf 1 1 1]; % upper bounds
            
            
        elseif k_model == 2  % full model 
            lb = [0 0 0 0 -1 0];       LB = [0 0 0 0 -Inf 0]; % lower bounds
            ub = [15 1 1 1 1 1];       UB = [Inf 1 1 1 Inf 1]; % upper bounds 

        end
        
        ddb = ub - lb; % where to look for the random point initialization
        
        % This part requires the Matlab Optimization toolbox
        % prepare multiple starting points for estimation
        n_rep           = 5;
        parameters_rep  = NaN(n_rep,nfpm(k_model));     parametersLPP_rep  = NaN(n_rep,nfpm(k_model));
        ll_rep          = NaN(n_rep,1);                 LPP_rep            = NaN(n_rep,1);
        FminHess        = NaN(n_rep,nfpm(k_model),nfpm(k_model));
        
        for k_rep = 1:n_rep
            % prepare starting points and parameter bounds
            x0 = lb + rand(1,length(lb)).*ddb; 
            x0 = x0(1:nfpm(k_model));

                                                                                                                                                                                    % params,s,a,r,obs,c,con,model
            % run ML and MAP estimations
            [parametersLPP_rep(k_rep,1:nfpm(k_model)),LPP_rep(k_rep),~,~,~,~,FminHess(k_rep,:,:)]=fmincon(@(x) Parameters_Priors_Complete_Observational_Final(x,sta{k_sub},cho{k_sub},out{k_sub},obs{k_sub},cou{k_sub},con{k_sub},k_model),x0,[],[],[],[],LB,UB,[],options);
        end
        
        % find best params over repetitions & store optimization outputs
   
        [~,posLPP]                                      = min(LPP_rep);
        parametersLPP(k_sub,k_model,1:nfpm(k_model))    = parametersLPP_rep(posLPP(1),1:nfpm(k_model));
        LPP(k_sub,k_model)                              = LPP_rep(posLPP(1),:) - nfpm(k_model)*log(2*pi)/2 + real(log(det(squeeze(FminHess(posLPP(1),:,:)))))/2;
        
        check_conv(k_sub)                               =  ~any(eig(squeeze(FminHess(posLPP(1),:,:)))<0);
        

    end
end

%% Models_partial observational 
% this function calculate the lilekihood of the models given a set of parameters 

function [post]=Parameters_Priors_Complete_Observational_Final(params,s,a,r,obs,c,con,model)


    
    %% confirmation bias
if model ==1
    
    beta  = params(1); pbeta = log(gampdf(beta,1.2,5.0)); 
    lr1   = params(2); plr1  = log(betapdf(lr1,1.1,1.1));
    lr2   = params(3); plr2  = log(betapdf(lr2,1.1,1.1));
    lr3   = params(4); plr3  = log(betapdf(lr3,1.1,1.1));
    
    p = [pbeta plr1 plr2 plr3];
    
    %% full model
elseif model == 2
    
    beta  = params(1); pbeta = log(gampdf(beta,1.2,5.0)); 
    lr1   = params(2); plr1  = log(betapdf(lr1,1.1,1.1));
    lr2   = params(3); plr2  = log(betapdf(lr2,1.1,1.1));
    tau   = params(4); ptau  = log(betapdf(tau,1.1,1.1));
    phi   = params(5); pphi  = log(normpdf(phi,0,1));
    lr3   = params(6); plr3  = log(betapdf(lr3,1.1,1.1));
    
    p = [pbeta plr1 plr2 ptau pphi plr3];
    
end

p = -sum(p);

l=Computational_Models_Complete_Observational_Final(params,s,a,r,obs,c,con,model);

post = p + l;

end
%%


%% Models_complete observational 
% this function calculate the lilekihood of the models given a set of parameters 

% this version estimates all model parameters in one go

function lik = Computational_Models_Complete_Observational_Final(params,s,a,r,obs,c,con,model)

% no bias


% confirmation bias
if model == 1
    beta  = params(1);
    lr1   = params(2); % conf learning rate
    lr2   = params(3); % dis learning rate
    lr3   = params(4); % dis learning rate
    % full model
elseif model == 2
    beta  = params(1);
    lr1   = params(2);
    lr2   = params(3);
    tau   = params(4);
    phi   = params(5);
    lr3   = params(6); % dis learning rate
end




% initializing the hidden values

Q       = zeros(16,2); %  Q-values
C       = zeros(16,2); %  C-traces

lik=0;

for i = 1:length(a)
    
    if (a(i))~=1.5 % to exclude missed reponses
        
        
        %% confirmation bias
        if model==1
            PEc =  r(i) - Q(s(i),a(i));
            PEu =  (c(i) - Q(s(i),3-a(i))) * (mod(con(i),2)==0); % why this? 
            if obs(i)==1
                lik = lik + log (1/(1+ exp(-beta*(Q(s(i),a(i))-Q(s(i),3-a(i))))));
                
                
                
                Q(s(i),a(i)) = Q(s(i),a(i)) + lr1 * PEc * (PEc>0) +  lr2 * PEc * (PEc<0);
                
                Q(s(i),3-a(i)) = Q(s(i),3-a(i)) + lr2 * PEu * (PEu>0) +  lr1 * PEu * (PEu<0);
            else
                Q(s(i),a(i)) = Q(s(i),a(i)) + lr3 * PEc;
                Q(s(i),3-a(i)) = Q(s(i),3-a(i)) + lr3 * PEu; % check this change here
            end
            
            %% gradual perseverance model
            
        elseif model ==2 %
            PEc =  r(i) - Q(s(i),a(i));
            PEu =  (c(i) - Q(s(i),3-a(i))) * (mod(con(i),2)==0);
            if obs(i)==1
                lik = lik + log (1/(1+ exp(-beta*(Q(s(i),a(i))-Q(s(i),3-a(i))) - phi*(C(s(i),a(i))-C(s(i),3-a(i))))));
                
                
                
                Q(s(i),a(i)) = Q(s(i),a(i)) + lr1 * PEc * (PEc>0) +  lr2 * PEc * (PEc<0);
                
                Q(s(i),3-a(i)) = Q(s(i),3-a(i)) + lr2 * PEu * (PEu>0) +  lr1 * PEu * (PEu<0);
                
                C(s(i),a(i)) = C(s(i),a(i)) + tau * (1 - C(s(i),a(i)));
                
                C(s(i),3-a(i)) = C(s(i),3-a(i)) + tau * (0 - C(s(i),3-a(i)));
            else
                Q(s(i),a(i)) = Q(s(i),a(i)) + lr3 * PEc;
                
                Q(s(i),3-a(i)) = Q(s(i),3-a(i)) + lr3 * PEu;
                
                
                C(s(i),a(i)) = C(s(i),a(i)) + tau * (0 - C(s(i),a(i)));
                
                C(s(i),3-a(i)) = C(s(i),3-a(i)) + tau * (0 - C(s(i),3-a(i)));
            end
        end
        
    end
    % save the choice for the next round (perseverations)
end

lik = -lik;                                                                % LL vector taking into account both the likelihood

end