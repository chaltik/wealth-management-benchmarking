#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 00:37:47 2018

@author: karen
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as norm_stats
from scipy.optimize import minimize,bisect,root_scalar,shgo,basinhopping
from scipy.integrate import quad
import sympy 

NTX_DIST_START=59.5
RMD_AGE_START=70.5
RMD_AGE_END=115.0
MAX_NTX_CONT=6500.
MAX_NTX_CONT_OLDER=7000.
INFL_RATE=0.03
CAP_GAIN_TAX=0.2


def max_drawdown_wealth(wealth_paths):
    """Calculate the maximum drawdown from a collection of wealth paths."""
    wealth_max = np.maximum.accumulate(wealth_paths, axis=0)
    drawdowns = wealth_max - wealth_paths
    max_drawdowns = np.max(drawdowns, axis=0)
    peaks = np.argmax(max_drawdowns, axis=0)
    return max_drawdowns, peaks


class Cashflow():
    def __init__(self,cf_ix,cf_values):
        self.cf_ix=cf_ix
        self.cf_values=cf_values
    
    def subset(self,t_after=0,t_before=None):
        t_end=self.cf_ix[-1] if t_before is None else t_before
        cf_mask=(self.cf_ix>=t_after)&(self.cf_ix<=t_end)
        if sum(cf_mask)>0:
            return Cashflow(self.cf_ix[cf_mask],self.cf_values[cf_mask]) 
        else:
            return None
       
    def add(self, cashflow):
        T=max(self.cf_ix[-1],cashflow.cf_ix[-1])+1
        sum_values=np.zeros(T)
        sum_ix=range(T)
        sum_values[self.cf_ix] += self.cf_values
        sum_values[cashflow.cf_ix] += cashflow.cf_values
        return Cashflow(sum_ix,sum_values)
    
class Profile():
    def __init__(self,
                 profile_obj_list, # pass in income and expenses with different signs, or their difference as a single object
                 goals_list,
                 T,
                 starting_wealth=0.,
                 min_spending_obj=None):
        self.T=T
        self.starting_wealth=starting_wealth
        self.profile_obj_list=profile_obj_list
        self.inflows=np.zeros(T)
        for o in self.profile_obj_list:
            self.inflows[o.cf_ix] += o.cf_values

        self.goals_list=goals_list
        self.goals=np.zeros(T)
        for g in self.goals_list:
            self.goals[g.cf_ix] += g.cf_values
        self.min_spending_obj=min_spending_obj
        self.min_spending=np.zeros(T)
        if min_spending_obj is not None:
            self.min_spending[min_spending_obj.cf_ix]=min_spending_obj.cf_values

def simulate_wealth_(paths,profile,w_policy,vis=False,show_paths=False,scale=1.):
    T = profile.T
    use_paths=paths[:T]
    return simulate_wealth(use_paths,
                           profile.goals,
                           profile.inflows,
                           w_policy,
                           profile.starting_wealth,
                           vis=vis,
                           show_paths=show_paths,
                           scale=scale)
    
# simulate wealth along all paths for a specific allocation policy
def simulate_wealth(paths,goals,inflows,w_policy,starting_wealth,vis=False,show_paths=False,scale=1.):
    (T,N_assets,N_paths)=paths.shape
    return_paths=(w_policy[:,:,np.newaxis]*paths).sum(axis=1)
    wealth_paths=np.zeros((T+1,N_paths))
    wealth_paths_g=np.zeros((T+1,N_paths))
    wealth_paths_g_cond=np.zeros((T+1,N_paths))
    utility_cond=np.zeros((T,N_paths))
    investment_paths_g=np.zeros((T+1,N_paths))
    max_utility=goals.sum()
    wealth_paths[0,:]=starting_wealth
    wealth_paths_g[0,:]=starting_wealth
    wealth_paths_g_cond[0,:]=starting_wealth
    cum_inflows=np.zeros(T+1)
    cum_inflows[0]=starting_wealth
    cum_inflows[1:]=starting_wealth+np.cumsum(inflows)
    cap_gains_tax=np.zeros((T+1,N_paths))
    for t in np.arange(T):
        wealth_paths_g_cond[t+1,:]=wealth_paths_g_cond[t,:]*(1+return_paths[t,:])+inflows[t]
        ok_to_spend = wealth_paths_g_cond[t+1,:]>goals[t]
        wealth_paths_g_cond[t+1,ok_to_spend] -= goals[t]
        utility_cond[t,ok_to_spend] += goals[t]                       
        wealth_paths_g[t+1,:]=(wealth_paths_g[t,:]*(1+return_paths[t,:])+inflows[t]-goals[t])
        wealth_paths[t+1,:]  =(wealth_paths[t,:]*(1+return_paths[t,:])+inflows[t])        
        investment_paths_g[t+1,:] = wealth_paths_g[t+1,:]-cum_inflows[t+1]+goals[t]
        cap_gains_tax[t+1,:] = 0.2*np.minimum(np.maximum(investment_paths_g[t+1,:],0.0),np.max(goals[t]-inflows[t],0))
        wealth_paths_g[t+1,:] = wealth_paths_g[t+1,:] - cap_gains_tax[t+1,:]

    utility = np.repeat(goals.reshape((T,1)),N_paths,axis=1)
    zero_idx=np.int8(wealth_paths_g[1:,]>0).cumprod(axis=0)
    utility = utility*zero_idx
    final_utility = (utility.sum(axis=0)/max_utility).mean()
    final_utility_cond = (utility_cond.sum(axis=0)/max_utility).mean()
    wealth_paths_g[1:]=wealth_paths_g[1:]*zero_idx
    investment_paths_g[1:]=investment_paths_g[1:]*zero_idx
    terminal_wealth=np.sign(wealth_paths_g[-1,:])
    percent_success=100*terminal_wealth.sum()/N_paths  
    investment_values=wealth_paths[1:] - (starting_wealth+np.cumsum(inflows))[:,np.newaxis]
    mdd=max_drawdown_wealth(investment_values)[0].mean()
    if vis:
        ####
        plt.subplots(1,2,figsize=(16,6))
        ##
        ax1=plt.subplot(1,2,1)
        if show_paths:
            w10top_=np.percentile(wealth_paths[-1,:],90,axis=0)
            success_paths_=(wealth_paths[-1,:]>0)&(wealth_paths[-1,:]<w10top_)
            exit_paths_=wealth_paths[-1,:]==0
            ax1.plot(scale*wealth_paths[:,success_paths_],linewidth=0.2,c='g')
        ax1.plot(scale*np.median(wealth_paths,axis=1),linewidth=2,c='b',linestyle='--',label='median')
        ax1.plot(scale*np.percentile(wealth_paths,10,axis=1),linewidth=2,c='k',label='10th percentile',linestyle='--')
        ax1.plot(scale*(starting_wealth+np.cumsum(inflows)),linewidth=2,c='cyan',linestyle='--',label='Uninvested wealth')
        ax1.plot(scale*(goals.cumsum()),linewidth=2,c='r',label='cumulative goals')
        ax1.grid()
        ax1.set_title('Pre-goal wealth paths vs goals',fontsize=16)
        ax1.legend(prop={'size':16})
        ##
        ax2=plt.subplot(1,2,2)
        w10top=np.percentile(wealth_paths_g[-1,:],90,axis=0)
        success_paths=(wealth_paths_g[-1,:]>0)&(wealth_paths_g[-1,:]<w10top)
        exit_paths=wealth_paths_g[-1,:]==0
        if show_paths:
            ax2.plot(scale*wealth_paths_g[:,success_paths],linewidth=0.1,c='g')
            ax2.plot(scale*wealth_paths_g[:,exit_paths],linewidth=0.1,c='pink')
        ax2.plot(scale*np.median(wealth_paths_g,axis=1),linewidth=2,c='b',linestyle='--',label='median')
        ax2.plot(scale*np.percentile(wealth_paths_g,10,axis=1),linewidth=2,c='k',label='10th percentile',linestyle='--')
        ax2.grid()
        ax2.set_title('$P$(success) ={:.1f}\%, Utility={:.1f}\%'.format(percent_success,final_utility*100),fontsize=18)
        ax2.legend(prop={'size':16})
        ax2.tick_params(labelsize=14)
        ##
        plt.show()
    return wealth_paths_g,percent_success,wealth_paths,final_utility,mdd,wealth_paths_g_cond,final_utility_cond


def simulate_wealth_with_threshold_(paths,profile,w_policy,wealth_boundary,vis=False,show_paths=False,scale=1.):        
    T = profile.T
    use_paths=paths[:T]    
    return simulate_wealth_with_threshold(use_paths,
                                          profile.goals,
                                          profile.inflows,
                                          w_policy,
                                          profile.starting_wealth,
                                          wealth_boundary,
                                          profile.min_spending,
                                          vis=vis,
                                          show_paths=show_paths,
                                          scale=scale)

# simulate wealth along all paths with conditional goal spending
def simulate_wealth_with_threshold(paths,goals,inflows,w_policy,starting_wealth,wealth_boundary,min_spending,vis=False,show_paths=False,scale=1.):
    (T,N_assets,N_paths)=paths.shape
    return_paths=(w_policy[:,:,np.newaxis]*paths).sum(axis=1)
    wealth_paths=np.zeros((T+1,N_paths))
    wealth_paths_g=np.zeros((T+1,N_paths))
    wealth_paths_g_cond=np.zeros((T+1,N_paths))
    utility_cond=np.zeros((T,N_paths))
    investment_paths_g=np.zeros((T+1,N_paths))
    max_utility=goals.sum()
    wealth_paths[0,:]=starting_wealth
    wealth_paths_g[0,:]=starting_wealth
    wealth_paths_g_cond[0,:]=starting_wealth
    cum_inflows=np.zeros(T+1)
    cum_inflows[0]=starting_wealth
    cum_inflows[1:]=starting_wealth+np.cumsum(inflows)
    cap_gains_tax=np.zeros((T+1,N_paths))
    cap_gains_tax_cond=np.zeros((T+1,N_paths))
    for t in np.arange(T):
        # conditional spending - only spend minimum if below provided boundary
        wealth_paths_g_cond[t+1,:]=wealth_paths_g_cond[t,:]*(1+return_paths[t,:])+inflows[t]
        ok_to_spend = wealth_paths_g_cond[t+1,:]>=wealth_boundary[t+1]
        wealth_paths_g_cond[t+1,ok_to_spend] -= goals[t]
        wealth_paths_g_cond[t+1,~ok_to_spend] -= min_spending[t]
        utility_cond[t,ok_to_spend] += goals[t]
        utility_cond[t,~ok_to_spend] += min_spending[t]
        
        # unconditional spending
        wealth_paths_g[t+1,:]=(wealth_paths_g[t,:]*(1+return_paths[t,:])+inflows[t]-goals[t])
        wealth_paths[t+1,:]  =(wealth_paths[t,:]*(1+return_paths[t,:])+inflows[t])        
        investment_paths_g[t+1,:] = wealth_paths_g[t+1,:]-cum_inflows[t+1]+goals[t]
        cap_gains_tax[t+1,:] = 0.2*np.minimum(np.maximum(investment_paths_g[t+1,:],0.0),np.max(goals[t]-inflows[t],0))
        wealth_paths_g[t+1,:] = wealth_paths_g[t+1,:] - cap_gains_tax[t+1,:]

    utility       = np.repeat(goals.reshape((T,1)),N_paths,axis=1)
    zero_idx      = np.int8(wealth_paths_g[1:,]>0).cumprod(axis=0)
    utility       = utility*zero_idx
    zero_idx_cond = np.int8(wealth_paths_g_cond[1:,]>0).cumprod(axis=0)
    utility_cond  = utility_cond*zero_idx_cond
    
    final_utility = 100*(utility.sum(axis=0)/max_utility).mean()
    final_utility_cond = 100*(utility_cond.sum(axis=0)/max_utility).mean()
    
    wealth_paths_g[1:]      = wealth_paths_g[1:]*zero_idx
    wealth_paths_g_cond[1:] = wealth_paths_g_cond[1:]*zero_idx_cond
    investment_paths_g[1:]=investment_paths_g[1:]*zero_idx
    terminal_wealth=np.sign(wealth_paths_g[-1,:])
    terminal_wealth_cond=np.sign(wealth_paths_g_cond[-1,:])
    percent_success=100*terminal_wealth.sum()/N_paths
    percent_success_cond=100*terminal_wealth_cond.sum()/N_paths
    investment_values=wealth_paths[1:] - (starting_wealth+np.cumsum(inflows))[:,np.newaxis]
    mdd=max_drawdown_wealth(investment_values)[0].mean()
    avg_run_out = np.int8(wealth_paths_g[1:]>0.).sum(axis=0)
    avg_run_out[avg_run_out==0]=T
    avg_run_out = avg_run_out.mean()/T
    avg_run_out_cond = np.int8(wealth_paths_g_cond[1:]>0.).sum(axis=0)
    avg_run_out_cond[avg_run_out_cond==0]=T
    avg_run_out_cond = avg_run_out_cond.mean()/T
    
    if vis:
        _,(ax1,ax2,ax3,ax4)=plt.subplots(1,4,figsize=(28,6))
        if show_paths:
            w10top_=np.percentile(wealth_paths_g[-1,:],90,axis=0)
            success_paths_=(wealth_paths_g[-1,:]>0)&(wealth_paths_g[-1,:]<w10top_)
            exit_paths_=wealth_paths_g[-1,:]==0
            ax1.plot(scale*wealth_paths_g[:,success_paths_],linewidth=0.2,c='g')
            ax1.plot(scale*wealth_paths_g[:,exit_paths_],linewidth=0.2,c='pink')
        ax1.plot(scale*np.median(wealth_paths_g,axis=1),linewidth=2,c='b',linestyle='--',label='median')
        ax1.plot(scale*np.percentile(wealth_paths_g,10,axis=1),linewidth=2,c='k',label='10th percentile',linestyle='--')
        ax1.plot(scale*(starting_wealth+np.cumsum(inflows)),linewidth=2,c='cyan',linestyle='--',label='Uninvested wealth')
        ax1.plot(scale*(goals.cumsum()),linewidth=2,c='r',label='cumulative goals')
        ax1.grid()
        ax1.set_title('Unconditional Spending Wealth Paths\n$P$ ={:.1f}\%, $U$={:.1f}\%, $T_0$={:.2f}'.format(percent_success,final_utility,avg_run_out),fontsize=16)
        ax1.legend(prop={'size':16})
        ##
        w10top=np.percentile(wealth_paths_g_cond[-1,:],90,axis=0)
        success_paths=(wealth_paths_g_cond[-1,:]>0)&(wealth_paths_g_cond[-1,:]<w10top)
        exit_paths=wealth_paths_g_cond[-1,:]==0
        if show_paths:
            ax2.plot(scale*wealth_paths_g_cond[:,success_paths],linewidth=0.2,c='g')
            ax2.plot(scale*wealth_paths_g_cond[:,exit_paths],linewidth=0.2,c='pink')
        ax2.plot(scale*np.median(wealth_paths_g_cond,axis=1),linewidth=2,c='b',linestyle='--',label='median')
        ax2.plot(scale*np.percentile(wealth_paths_g_cond,10,axis=1),linewidth=2,c='k',label='10th percentile',linestyle='--')
        ax2.plot(scale*wealth_boundary,linewidth=2,c='r',label='Spending Boundary',linestyle='--')
        ax2.grid()
        ax2.set_title('Conditional Spending Wealth Paths\n$P$ ={:.1f}\%, $U$={:.1f}\%, $T_0$={:.2f}'.format(percent_success_cond,final_utility_cond,avg_run_out_cond),fontsize=18)
        ax2.legend(prop={'size':16})
        ax2.tick_params(labelsize=14)
        ##
        ax3.plot(scale*utility.cumsum(axis=0),linewidth=0.2,c='b')
        ax3.grid()
        ax3.set_title('Unconditional Utility accumulation',fontsize=16)
        ##
        ax4.plot(scale*utility_cond.cumsum(axis=0),linewidth=0.2,c='b')
        ax4.grid()
        ax4.set_title('Conditional Utility accumulation',fontsize=16)
        
        plt.show()
    return wealth_paths_g,percent_success,wealth_paths,final_utility,mdd,wealth_paths_g_cond,final_utility_cond,avg_run_out,avg_run_out_cond,utility_cond


#### probability of achieving goals using allocation trajectory weights, starting with starting_wealth
def prob_achieved(paths,goals,inflows,weights,starting_wealth,risk_adjust=0.):
    _,prob,_,_,mdd,_,_ = simulate_wealth(paths,goals,inflows,weights,starting_wealth)
    return 0.01*prob - risk_adjust*mdd

def prob_achieved_(paths,profile,w_policy,risk_adjust=0.):
    _,prob,_,_,mdd,_,_ = simulate_wealth_(paths,profile,w_policy)
    return 0.01*prob - risk_adjust*mdd

#### average utility using allocation trajectory weights, starting with starting_wealth
def utility_achieved(paths,goals,inflows,weights,starting_wealth,risk_adjust=0.):
    _,_,_,ut,mdd,_,_ = simulate_wealth(paths,goals,inflows,weights,starting_wealth)
    return ut - risk_adjust*mdd

def utility_achieved_(paths,profile,w_policy,risk_adjust=0.):
    _,_,_,ut,mdd,_,_ = simulate_wealth_(paths,profile,w_policy)
    return ut - risk_adjust*mdd

####
def bisection(f,a_in,b_in,tol,disp=False):
    a=a_in
    b=b_in
    x = (a+b)/2.0
    attempt=0
    while attempt < 3:
        if (f(a)*f(b)>0):
            a=a/np.sqrt(2)
            b=b*np.sqrt(2)
        attempt +=1
    it=0    
    while (b-a)/(abs(a)+abs(b)) > tol:
        if disp:
            print('iteration {}: x={}, a={}, b={}, f(x)={},f(a)={},f(b)={}'.format(it,x,a,b,f(x),f(a),f(b)))
        if f(x) == 0:
            return x
        elif f(a)*f(x) < 0:
            b = x
        else :
            a = x
        x = (a+b)/2.0
        it +=1
        if it==20:
            break;
    return x   
####
def min_wealth_for_policy(paths,goals,inflows,w_policy,prob_thresh,use_prob=False,risk_adjust=0.):
    if use_prob:
        target_fun=prob_achieved
    else:
        target_fun=utility_achieved
    fzero=target_fun(paths,
                     goals,
                     inflows,
                     w_policy,
                     0.0,
                     risk_adjust=risk_adjust)
    if fzero > prob_thresh-0.01:
        return 0.0
    else:
        return bisection(lambda w0: target_fun(paths,
                                               goals,
                                               inflows,
                                               w_policy,
                                               w0,
                                               risk_adjust=risk_adjust)-prob_thresh,0,2.0*goals.sum(),tol=0.01)
    
def min_wealth_for_policy_(paths,profile,w_policy,prob_thresh,use_prob=False,risk_adjust=0.):
    if use_prob:
        target_fun=prob_achieved
    else:
        target_fun=utility_achieved
        
    fzero=target_fun(paths[:profile.T],
                     profile.goals,
                     profile.inflows,
                     w_policy,
                     0,
                     risk_adjust=risk_adjust)
    if fzero > prob_thresh-0.01:
        return 0.0
    else:
        return bisection(lambda w0: target_fun(paths[:profile.T],
                                               profile.goals,
                                               profile.inflows,
                                               w_policy,
                                               w0,
                                               risk_adjust=risk_adjust)-prob_thresh,0,2.0*profile.goals.sum(),tol=0.01)

def max_spending_for_policy(paths,goals,income,w_policy,w0,prob_thresh,use_prob=False,risk_adjust=0.):
    if use_prob:
        target_fun=prob_achieved
    else:
        target_fun=utility_achieved    
    p0=target_fun(paths,goals,income,w_policy,w0,risk_adjust=0.)
    if p0<prob_thresh:
        print('Either goals or income or target probability need to be adjusted, as even p(zero spending)={:.2f} < {:.2f}'.format(p0,prob_thresh))
        return 0.0
    else:
        return bisection(lambda x: target_fun(paths,
                                              goals,
                                              income*(1-x),
                                              w_policy,
                                              w0,risk_adjust=risk_adjust)-prob_thresh,0,1,tol=0.01)
    
def max_spending_for_policy_(paths,profile,w_policy,prob_thresh,use_prob=False,risk_adjust=0.):
    if use_prob:
        target_fun=prob_achieved_
    else:
        target_fun=utility_achieved_    
    p0=target_fun(paths,profile,w_policy,w0,risk_adjust=0.)
    if p0<prob_thresh:
        print('Either goals or income or target probability need to be adjusted, as even p(zero spending)={:.2f} < {:.2f}'.format(p0,prob_thresh))
        return 0.0
    else:
        return bisection(lambda x: target_fun(paths,
                                              Profile(profile.inflows*(1-x),
                                                      profile.goals_list,
                                                      profile.T,
                                                      profile.starting_wealth),
                                              w_policy,
                                              risk_adjust=risk_adjust)-prob_thresh,0,1,tol=0.01)
        
####
def goal_single(t:int,T:int): 
    goals=np.zeros(T)
    goals[t]=1
    return goals    
####

def gp_kernel(t,x):
    e = 0.5*(1 + np.tanh((t-x[1])/x[0]))
    bc = 1 - e
    b = bc*np.tanh(t/x[2])
    c = bc - b
    return np.array([e,b,c]).reshape((1,3))

def build_policy_for_goals(x,goals,kernel_fun=gp_kernel):
    T=len(goals)    
    times=np.arange(T,dtype=np.int16)
    kernel = np.vstack([kernel_fun(t,x) for t in range(T)])
    n=np.zeros(T)

    # placeholders for actual policy 
    stocks=np.empty(T)
    cash=np.empty(T)
    bonds=np.empty(T)
    n[0]=goals.sum()
    stocks[0]=(kernel[:,0]*goals).sum()/n[0]
    cash[0]  =(kernel[:,2]*goals).sum()/n[0]
    
    # weigh all the contributions by the goal weight
    for t in range(1,T):
        stocks[t]=(kernel[:-t,0]*goals[t:]).sum()
        cash[t]  =(kernel[:-t,2]*goals[t:]).sum()
        n[t]     = goals[t:].sum()
        if n[t]>0:
            stocks[t] /= n[t]
            cash[t] /= n[t]
        else:
            if t>0:
                stocks[t]=stocks[t-1]
                cash[t]=cash[t-1]
    
    bonds=1.0-stocks-cash
    return np.column_stack((stocks,bonds,cash))



