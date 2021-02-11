

import numpy as np
import pandas as pd

from wfpt import wfpt_like, wfpt_gen
from scipy.stats import norm
from RunDEMC import dists

import StudentTMulti as st
import OCPD as oncpd
from functools import partial
# from RL_MAP import RL_MAP
from CPD_CAL import cpd_cal
from RL_POST_MultVar import RL_POST_MultVar

class MMIModel():
    """Generative model of the MMI task"""
    default_params = {'gamma': 0,
                      'gamma_neg': None,
                      'c':0.5,
                      #'beta': 0.75,
                      #'beta_certain': None,
                      #'a': None,
                      #'w': None,
                      #'t0': None,
                      #'p0': 0.5,
                     }
    
    def __init__(self, params=None, max_time=3.0, 
                 trange_nsamp=1000, gen_nsamp=1000,
                 wfpt_nsamp=5000, ignore_non_resp=False):
        # set params based on defaults and those passed in
        self.params = dict(**self.default_params)
        if params is not None:
            self.params.update(params)
        
        # set the max_time
        self.max_time = max_time
        self.trange_nsamp = trange_nsamp
        self.gen_nsamp = gen_nsamp
        self.wfpt_nsamp = wfpt_nsamp
        self.ignore_non_resp = ignore_non_resp
        
        # initialize stuff
        self.p = 0.5
        #self.p = self.params['p0']
        if self.params['gamma_neg'] is None:
            # just set it to gamma
            self.params['gamma_neg'] = self.params['gamma']
#         if self.params['beta_certain'] is None:
#             # just set it to beta
#             self.params['beta_certain'] = self.params['beta']
        
    def update_p(self, trial):
        if trial.Choice == 'Gamble':
            RPE = self._apply_gamma(trial['OutcomeAmount']) - trial['LTA']
#             I = float(trial['Won'])
#             beta = self.params['beta']
        elif trial.Choice == 'None':
            # they gambled with a non-response,
            # but maybe didn't make a prediction
            # for now, just treat it like an active gamble
            RPE = self._apply_gamma(trial['OutcomeAmount']) - trial['LTA']
#             I = float(trial['Won'])
#             beta = self.params['beta']
        else:
            # they took certain value
            RPE = 0.0
            
            # move back towards initial probability
#             I = self.params['p0']

            # use the certain beta if there is one
#             beta = self.params['beta_certain']
            
        # calc r 
        # (we may need to add a temperature param here)
        # (we also could weigh positive and negative RPE differently)
        #r = np.exp(-np.abs(RPE))
        r = 1.0
        
        # update p
#         self.p = r*beta*self.p + (1-r*beta)*I
        #self.p = np.clip(self.p + self.params['beta']*RPE, 0, 1)
        self.p=trial['win_prob']
        
        return self.p, RPE, r
    
    def _apply_gamma(self, value):
        # apply positive or neg gamma (may simply be symmetric)
        if value < 0:
            gamma = self.params['gamma_neg']
        else:
            gamma = self.params['gamma']
        return np.sign(value)*np.abs(value)**gamma
    
    def calc_E(self, trial):
        # determine the certain and gambling expected values
        EC = self._apply_gamma(trial['CertainAmount'])
        EG = self.p*self._apply_gamma(trial['GreaterAmount']) + \
             (1-self.p)*self._apply_gamma(trial['LesserAmount'])
        return EC,EG
    
    def calc_Reward(self, trial):
        # determine the subjective value
        rwrd = self._apply_gamma(trial['OutcomeAmount'])
        
        return rwrd
    
#     def calc_latents(self, trial, latents):
#         latents2 = {}
#         #latents2['CA_sum'] = self.params['lambda'] * latents['CA_sum']
#         #latents2['EG_sum'] = self.params['lambda'] * latents['EG_sum']

#         latents2['RPE_sum'] = self.params['lambda'] * latents['RPE_sum']
#         latents2['LTA_sum'] = self.params['lambda'] * latents['LTA_sum']

#         latents2['LTA_sum'] = latents2['LTA_sum'] + trial['LTA']

#         if trial.Choice == 'Gamble':
#             #latents2['EG_sum'] = latents2['EG_sum'] + trial['EG']
#             #latents2['LTA_sum'] = latents2['LTA_sum'] + trial['EG']
#             latents2['RPE_sum'] = latents2['RPE_sum'] + trial['RPE']
#         # else:
#         #     #latents2['CA_sum'] = latents2['CA_sum'] + trial['EC']
#         #     latents2['LTA_sum'] = latents2['LTA_sum'] + trial['EC']
#         return latents2
    
    def calc_trial_like(self, trial, save_post=False):
        # Compute likelihood from reaction time
        # see what response was made and map it to the choice
        if trial.Choice == 'Gamble':
            choice = np.array([1])
        elif trial.Choice == 'Certain':
            choice = np.array([0])
        else:
            # they made no choice
            # we could consider skipping these
            choice = np.array([2])
        
        
        # calc the like
        if self.ignore_non_resp and choice==np.array([2]):
            log_like = 0.0
            
        else:
            #### Exclusion of RT (likelihood depends only on choices)
            p_c=dists.invlogit(trial['v_mean']) # prob. of making a choice
            likelihood=(p_c**choice) * ((1-p_c)**(1-choice))
            log_like=np.log(likelihood)
            
            ## Debug
            #import pdb; pdb.set_trace()
            
            # calc the log like
#             log_like = np.log(wfpt_like(choice, np.array([trial.RT]), 
#                                         v_mean=trial['Ediff'], a=self.params['a'], 
#                                         w_mode=self.params['w'], t0=self.params['t0'],
#                                         nsamp=self.wfpt_nsamp,
#                                         max_time=self.max_time,
#                                         trange_nsamp=self.trange_nsamp))[0]

            
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # if the trial is also a mood trial we could also add in a like calc
        # for a model, too
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        if np.isnan(trial['Mood']):
            mood_log_like = 0.0                 
        
        
        if not np.isnan(trial['Mood']):
            curr_mood = dists.logit(trial['Mood']/1000)
            
            ### Modification for P_win
            pred_mood = self.params['b'] + self.params['w_p'] * trial['P_win']
                        
            log_like = log_like + np.log(norm.pdf(curr_mood, pred_mood, np.sqrt(self.params['s_v'])))
            
            ##Mood_log_likelihood 
            mood_log_like = np.log(norm.pdf(curr_mood, pred_mood, np.sqrt(self.params['s_v'])))
                    
        # see if running conditional sim
        if save_post:
            # run wfpt_gen
            choices, rts = wfpt_gen(v_mean=trial['Ediff'], 
                                    a=self.params['a'], 
                                    w_mode=self.params['w'], 
                                    wfpt_nsamp=self.wfpt_nsamp,
                                    nsamp=self.gen_nsamp, 
                                    trange=np.linspace(0, self.max_time-self.params['t0'], self.trange_nsamp))
            
            # calc prob of making the observed choice
            ind = choices==choice
            p_choice = ind.mean()
            
            # calc mean log rt
            choice_mean_log_rt = np.log(rts[ind]+self.params['t0']).mean()
            
            
            return log_like, p_choice, choice_mean_log_rt
        
        return log_like, mood_log_like 
    
    def proc_trials(self, trials, save_posts=False):
        # loop over trials
        #latents = {'CA_sum':0, 'EG_sum':0, 'RPE_sum':0}
        #latents = {'LTA_sum':0, 'RPE_sum':0}
        reward_sum=0
        #avg_reward_trial=0
        for i in range(len(trials)):
            # calc_E and save it to the current trial
            EC,EG = self.calc_E(trials.iloc[i])
            trials.at[trials.index[i], 'EC'] = EC
            trials.at[trials.index[i], 'EG'] = EG
            trials.at[trials.index[i], 'Ediff'] = EC-EG
            
            ### v_mean
            trials.at[trials.index[i], 'v_mean'] = self.params['c']*(EC-EG)            
            
            # calc_R and save it to the current trial
            rwrd = self.calc_Reward(trials.iloc[i])
            
            ### Long Term subjective reward calculation (Correction March 12, 2020)
            if i==0:
                trials.at[trials.index[i], 'LTA']=0
            else:
                trials.at[trials.index[i], 'LTA']=reward_sum/trial_num        
            reward_sum=reward_sum+rwrd
            trial_num=i+1
            
           #### Modification for P_win
            if i==0:
                trials.at[trials.index[i], 'P_win']=0.5
            else:
                trials.at[trials.index[i], 'P_win']=Pwin

            # update_p (QUESTION: Does it matter whether update_p happens before or after calc likelihood)
            new_p, RPE, r = self.update_p(trials.iloc[i])
            trials.at[trials.index[i], 'new_p'] = new_p
            trials.at[trials.index[i], 'RPE'] = RPE
            trials.at[trials.index[i], 'r'] = r
            
            #### Modification for P_win
            Pwin = new_p
            
#             # update
#             latents = self.calc_latents(trials.iloc[i], latents)
#             #trials.at[trials.index[i], 'CA_sum'] = latents['CA_sum']
#             trials.at[trials.index[i], 'LTA_sum'] = latents['LTA_sum']
#             trials.at[trials.index[i], 'RPE_sum'] = latents['RPE_sum']

            
            # calc_trial_like
            if save_posts:
                log_like, p_choice, choice_mean_log_rt = self.calc_trial_like(trials.iloc[i], 
                                                                              save_post=save_posts)
                trials.at[trials.index[i], 'log_like'] = log_like
                trials.at[trials.index[i], 'p_choice'] = p_choice
                trials.at[trials.index[i], 'choice_mean_log_rt'] = choice_mean_log_rt
                
            else:
                log_like, mood_log_like = self.calc_trial_like(trials.iloc[i])
                trials.at[trials.index[i], 'log_like'] = log_like
                ### Mood_log_like
                trials.at[trials.index[i], 'mood_log_like'] = mood_log_like
                #trials.at[trials.index[i], 'log_like'] = self.calc_trial_like(trials.iloc[i])
                #trials.at[trials.index[i], 'mood_log_like'] = mood_log_like
            ### Debug  
            #assert(not np.isnan(trials.at[trials.index[i], 'log_like']))
            
#             Reac_time=np.array([trials.at[trials.index[i], 'RT']])
#             v_mean=trials.at[trials.index[i], 'v_mean']
#             GAMMA=self.params['gamma']
#             BETA = self.params['beta']
#             C = self.params['c']
#             #a=self.params['a']                           
#             #w_mode=self.params['w']
#             #t0=self.params['t0']
#             P0=self.params['p0']
#             Sv=self.params['s_v']
            
            if np.isnan(trials.at[trials.index[i], 'log_like']):
                #print((Reac_time,v_mean,GAMMA,BETA,C,P0,Sv), flush=True)
                raise ValueError("Log_Like NaN value")       
            if np.isnan(trials.at[trials.index[i], 'mood_log_like']):
                #print((Reac_time,v_mean,GAMMA,BETA,C,P0,Sv), flush=True)
                raise ValueError("Mood_Log_Like NaN value")
            if trials.at[trials.index[i], 'mood_log_like']==None:
                print([trials.index[i], 'mood_log_like'])
                print(trials.at[trials.index[i], 'log_like'])
                raise ValueError("Zero value")

            #import pdb; pdb.set_trace()
        
        # return trials with useful columns added
        return trials

    

    
    
########### WinProb_Context Function Dominant Time line (DS) #################
def winProb_context(dat,cps):
    
    if len(cps)==2:
        
        cp1=int(cps[0])
        print('1st Change point:',cp1)
        cp2=int(cps[1])
        print('2nd Change point:',cp2)

        ## Initializing context_01 as one vector
        context_01 = np.ones(len(dat))
        ## Initializing context_02 as zero vector
        context_02 = np.zeros(len(dat))
        

        ## Setting the non-context_01 as zeros
        context_01[cp1:cp2+1] = 0
        ## Setting the context_02 as ones
        context_02[cp1:cp2+1] = 1

        ## multiplying the win by context vector
        win_context01=dat.win*context_01
        win_context02=dat.win*context_02


        # Choose only the values related to context_01 and context_02 
        win_context01 = win_context01[context_01 == 1.0]
        no_trial_context_01=len(win_context01)

        win_context02 = win_context02[context_02 == 1.0]
        no_trial_context_02=len(win_context02)
        # outCome_context01_arr=np.array(outCome_context01)
        # outCome_context01_arr[80]

        trial_context_01_no = np.array([i for i in range(no_trial_context_01)]) + 1
        trial_context_02_no = np.array([i for i in range(no_trial_context_02)]) + 1 


        ### Compute winProb for "context_01"
        temp_01 = np.cumsum(win_context01)/trial_context_01_no
        winProb_context_01 = np.zeros((no_trial_context_01))  # 0.5*np.ones(no_trial_context_01)
        winProb_context_01[1:] = temp_01[:-1]
        ## setting first element as 0.5
        winProb_context_01[0] = 0.5

        ### Compute winProb for "context_02"
        temp_02 = np.cumsum(win_context02)/trial_context_02_no
        winProb_context_02 = np.zeros((no_trial_context_02)) # 0.5*np.ones( no_trial_context_02)
        winProb_context_02[1:] = temp_02[:-1]
        ## setting first element as 0.5
        #winProb_context_02[0] = 0.5

        ## Merging winProb for context_01 and context_02
        winProb_context_total = np.zeros(len(dat)) #0.5*np.ones(len(dat))
        winProb_context_total[0:cp1] = winProb_context_01[0:cp1]
        winProb_context_total[cp1:cp2+1] = winProb_context_02
        winProb_context_total[cp2+1:] = winProb_context_01[cp1:]
        
    if len(cps)==1:
        
        cp1=int(cps[0])
        print('1st Change point:',cp1)
        cp2=len(dat)-1
        #print('2nd Change point:',cp2)

        ## Initializing context_01 as one vector
        context_01 = np.ones(len(dat))
        ## Initializing context_02 as zero vector
        context_02 = np.zeros(len(dat))
        

        ## Setting the non-context_01 as zeros
        context_01[cp1:cp2+1] = 0
        ## Setting the context_02 as ones
        context_02[cp1:cp2+1] = 1

        ## multiplying the win by context vector
        win_context01=dat.win*context_01
        win_context02=dat.win*context_02


        # Choose only the values related to context_01 and context_02 
        win_context01 = win_context01[context_01 == 1.0]
        no_trial_context_01=len(win_context01)

        win_context02 = win_context02[context_02 == 1.0]
        no_trial_context_02=len(win_context02)
        # outCome_context01_arr=np.array(outCome_context01)
        # outCome_context01_arr[80]

        trial_context_01_no = np.array([i for i in range(no_trial_context_01)]) + 1
        trial_context_02_no = np.array([i for i in range(no_trial_context_02)]) + 1 


        ### Compute winProb for "context_01"
        temp_01 = np.cumsum(win_context01)/trial_context_01_no
        winProb_context_01 = np.zeros((no_trial_context_01))  # 0.5*np.ones(no_trial_context_01)
        winProb_context_01[1:] = temp_01[:-1]
        ## setting first element as 0.5
        winProb_context_01[0] = 0.5

        ### Compute winProb for "context_02"
        temp_02 = np.cumsum(win_context02)/trial_context_02_no
        winProb_context_02 = np.zeros((no_trial_context_02)) # 0.5*np.ones( no_trial_context_02)
        winProb_context_02[1:] = temp_02[:-1]
        ## setting first element as 0.5
        #winProb_context_02[0] = 0.5

        ## Merging winProb for context_01 and context_02
        winProb_context_total = np.zeros(len(dat)) #0.5*np.ones(len(dat))
        winProb_context_total[0:cp1] = winProb_context_01#[0:cp1]
        winProb_context_total[cp1:cp2+1] = winProb_context_02
        #winProb_context_total[cp2+1:] = winProb_context_01[cp1:]
        
    if len(cps)==0:
        
#         winProb_context_total = np.zeros(len(dat))
#         winProb_context_total[1:] = dat.win_prob[:-1]
#         winProb_context_total[0]=0.5
        
        #dat['win'] = dat.Won.astype(int)
        Ntrial_idx = dat.index.values + 1
        winProb_context_total = np.cumsum(dat.win)/Ntrial_idx 

    return winProb_context_total




#########  WinProb_Context Function for dynamic time-line using POST (DS) ###############
def winProb_context_dyn_POST(t,dat,P):
    ## initialization
#     winProb_context = 0.0 
    if t==0:
        winProb_context = 0.5  
    if t>0:
        sum_win = 0
        for r in range(1,int(t+1)):
            low = int(t-r)
            #print(low)
            ## python for loop range high will go upto (t-1)
            high = int(t) 
            #print(high-1)
            #print(r)
            for u in range(low,high):
                sum_win = sum_win + P[r,t]*(1/r)*dat.win[u]
            winProb_context = sum_win
            
    return winProb_context


########  WinProb_Context Function for dynamic time-line using MAP (DS) #####################
def winProb_context_dyn_MAP(t,dat,R):
    ## initialization
#     winProb_context = 0.0 
    if t==0:
        winProb_context = 0.5
    if t>0:
        low = int(t-R[t])
        #print(low)
        ## python for loop range high will go upto (t-1)
        high = int(t) 
        #print(high-1)
        sum_win = 0
        #print(R[t])
        for u in range(low,high):
            sum_win = sum_win + dat.win[u]
        winProb_context = (1/R[t])*sum_win
    
    return winProb_context
        
# read in the data
def load_mmi_data(filename):
    
    dat = pd.read_csv(filename)
    # ignore rows with no trial info
    dat= dat.dropna(how ='any', subset = ['certainAmount'])
    #dat = dat.loc[~np.isnan(dat.winAmount)]
    dat=dat.reset_index(drop=True)
    # grab the columns of interest
    # CertainAmount, Outcome1Amount (one of the gambling outcomes), 
    # Outcome2Amount (one of the gambling outcomes), 
    # Outcome (which outcome occurred, certain or Outcome1 or Outcome2), 
    # GetAnswer_RT
#     cols = ['CertainAmount', 'Outcome1Amount', 'Outcome2Amount', 'Outcome', 'GetAnswer__RT',
#             'MoodTarget', 'TrialHappiness__SliderResp', 'DoRating', 'CurrentAmount'
#            ]
    cols=['certainAmount','happySlider.response','winAmount','loseAmount','choiceKey.rt','outcome','doRating']
    dat = dat[cols]

    # add useful columns
    # first rename some
    #dat['RT'] = dat['GetAnswer__RT']/1000.
    dat['RT'] = dat['choiceKey.rt'] # original RT data is in seconds
    dat['RT'] = dat['RT'].fillna(0.0) # removing NaN values with 0
    
    #dat.at[12, 'RT'] = 2
    #dat.loc[1,'RT']=2
    dat['Mood'] = dat['happySlider.response']*1000.0 # original Mood data range 0-1
    
    ### replacing Mood Rating 0-->1 and 1000-->999
    dat.Mood = dat.Mood.replace({0: 1.0, 1000: 999.0})
    
    # Certain amount
    dat['CertainAmount'] = dat['certainAmount']
    
    # add a choice
    dat['Choice'] = 'None'
    ind = (dat['RT']>0)
    dat.loc[ind&(dat['outcome']=='certain'), 'Choice'] = 'Certain'
    dat.loc[ind&(dat['outcome']=='win'), 'Choice'] = 'Gamble'
    dat.loc[ind&(dat['outcome']=='lose'), 'Choice'] = 'Gamble'

    # add outcome amount
    dat['OutcomeAmount'] = dat['certainAmount']
    ind = (dat['outcome']=='win')
    dat.loc[ind, 'OutcomeAmount'] = dat.loc[ind, 'winAmount']
    ind = (dat['outcome']=='lose')
    dat.loc[ind, 'OutcomeAmount'] = dat.loc[ind, 'loseAmount']

    # add greater and lesser amounts
    dat['GreaterAmount'] = dat['winAmount']
    dat['LesserAmount'] = dat['loseAmount']
    dat['Won'] = dat['OutcomeAmount'] == dat['GreaterAmount']

    
###### POST(r,t) and RL_MAP calculations (DS) ##################
    # dim refers to the no. of VAR are used to calculate POST or MAP
    dim = 1
    # 'maxes' and 'POST' are the MAP and full posterior 
    POST, maxes, RL_maxes = RL_POST_MultVar(dat,dim)
    

########### winProb_context Dominant_time_line (DS) ###################
   ##### Change Point calculatons
    trial_nos = np.array([i for i in range(len(dat))])
    frequency = 11
    cut_off=20
    cps,count = cpd_cal(frequency, maxes, trial_nos, cut_off)
    dat['win'] = dat.Won.astype(int)
    winProb_context_total = winProb_context(dat,cps)
    dat['win_prob'] = winProb_context_total
    
    
########## winProb_context_Dynamic_Timeline_MAP (DS) #####################
#     dat['win'] = dat.Won.astype(int)
#     win_dyn_prob_context = np.zeros(len(dat))
#     for t in range(len(dat)):
#         # maxes are the RL_MAP
#         win_dyn_prob_context[t] = winProb_context_dyn_MAP(t,dat,maxes)
#     dat['win_prob'] = win_dyn_prob_context


############## Win_Prob_Context_Dynamic_TimeLine_POST (DS) #####################
#     dat['win'] = dat.Won.astype(int)
#     win_dyn_prob_context_POST = np.zeros(len(dat))
#     for t in range(len(dat)):
#         win_dyn_prob_context_POST[t] = winProb_context_dyn_POST(t,dat,POST)
#     dat['win_prob'] = win_dyn_prob_context_POST
    
    
############ Non-Contextual win-prob Model (DS) ##########################      
#     dat['win'] = dat.Won.astype(int)
#     Ntrial_idx = dat.index.values + 1
#     dat['win_prob'] = np.cumsum(dat.win)/Ntrial_idx 

    return dat

