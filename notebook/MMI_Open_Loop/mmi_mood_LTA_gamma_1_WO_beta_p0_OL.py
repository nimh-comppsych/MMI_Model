

import numpy as np
import pandas as pd

from wfpt import wfpt_like, wfpt_gen
from scipy.stats import norm
from RunDEMC import dists

class MMIModel():
    """Generative model of the MMI task"""
    default_params = {'gamma': 1.0,
                      'gamma_neg': None,
                      #'beta': 1,
                      #'beta_certain': None,
                      'a': 3.0,
                      'w': 0.5,
                      't0': 0.25
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
        if self.params['gamma_neg'] is None:
            # just set it to gamma
            self.params['gamma_neg'] = self.params['gamma']
#         if self.params['beta_certain'] is None:
#             # just set it to beta
#             self.params['beta_certain'] = self.params['beta']
        
    def update_p(self, trial):
        if trial.Choice == 'Gamble':
            RPE = self._apply_gamma(trial['OutcomeAmount']) - trial['LTA']
            #I = float(trial['Won'])
            #beta = self.params['beta']
        elif trial.Choice == 'None':
            # they gambled with a non-response,
            # but maybe didn't make a prediction
            # for now, just treat it like an active gamble
            RPE = self._apply_gamma(trial['OutcomeAmount']) - trial['LTA']
            #I = float(trial['Won'])
            #beta = self.params['beta']
        else:
            # they took certain value
            RPE = 0.0
            
            # move back towards initial probability
            #I = self.params['p0']

            # use the certain beta if there is one
            #beta = self.params['beta_certain']
            
        # calc r 
        # (we may need to add a temperature param here)
        # (we also could weigh positive and negative RPE differently)
        #r = np.exp(-np.abs(RPE))
        r = 1.0
        
        # update p
        #self.p = r*beta*self.p + (1-r*beta)*I
        self.p=trial['win_prob']
        #self.p = np.clip(self.p + self.params['beta']*RPE, 0, 1)
        
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
    
    def calc_latents(self, trial, latents):
        latents2 = {}
        #latents2['CA_sum'] = self.params['lambda'] * latents['CA_sum']
        #latents2['EG_sum'] = self.params['lambda'] * latents['EG_sum']

        latents2['RPE_sum'] = self.params['lambda'] * latents['RPE_sum']
        latents2['LTA_sum'] = self.params['lambda'] * latents['LTA_sum']

        latents2['LTA_sum'] = latents2['LTA_sum'] + trial['LTA']

        if trial.Choice == 'Gamble':
            #latents2['EG_sum'] = latents2['EG_sum'] + trial['EG']
            #latents2['LTA_sum'] = latents2['LTA_sum'] + trial['EG']
            latents2['RPE_sum'] = latents2['RPE_sum'] + trial['RPE']
        # else:
        #     #latents2['CA_sum'] = latents2['CA_sum'] + trial['EC']
        #     latents2['LTA_sum'] = latents2['LTA_sum'] + trial['EC']
        return latents2
    
    def calc_trial_like(self, trial, save_post=False):
        # Compute likelihood from reaction time
        # see what response was made and map it to the choice
        if trial.Choice == 'Gamble':
            choice = np.array([1])
        elif trial.Choice == 'Certain':
            choice = np.array([2])
        else:
            # they made no choice
            # we could consider skipping these
            choice = np.array([0])

        # calc the like
        if self.ignore_non_resp and choice==np.array([0]):
            log_like = 0.0
        else:
            # calc the log like
            log_like = np.log(wfpt_like(choice, np.array([trial.RT]), 
                                        v_mean=trial['Ediff'], a=self.params['a'], 
                                        w_mode=self.params['w'], t0=self.params['t0'],
                                        nsamp=self.wfpt_nsamp,
                                        max_time=self.max_time,
                                        trange_nsamp=self.trange_nsamp))[0]
            
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # if the trial is also a mood trial we could also add in a like calc
        # for a model, too
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        if np.isnan(trial['Mood']):
            mood_log_like = 0.0                 
            
        if not np.isnan(trial['Mood']):
            curr_mood = dists.logit(trial['Mood']/1000)
            pred_mood = self.params['b'] + \
                        self.params['w_LTA'] * trial['LTA_sum'] + \
                        self.params['w_RPE'] * trial['RPE_sum']
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
    
    def proc_trials(self, trials, save_posts=False, return_mood_like=False):
        # loop over trials
        #latents = {'CA_sum':0, 'EG_sum':0, 'RPE_sum':0}
        latents = {'LTA_sum':0, 'RPE_sum':0}
        reward_sum=0
        #avg_reward_trial=0
        for i in range(len(trials)):
            # calc_E and save it to the current trial
            EC,EG = self.calc_E(trials.iloc[i])
            trials.at[trials.index[i], 'EC'] = EC
            trials.at[trials.index[i], 'EG'] = EG
            trials.at[trials.index[i], 'Ediff'] = EC-EG
            
            # calc_R and save it to the current trial
            rwrd = self.calc_Reward(trials.iloc[i])
            
            ### Long Term subjective reward calculation (Correction March 12, 2020)
            if i==0:
                trials.at[trials.index[i], 'LTA']=0
            else:
                trials.at[trials.index[i], 'LTA']=reward_sum/trial_num        
            reward_sum=reward_sum+rwrd
            trial_num=i+1
            

            # update_p (QUESTION: Does it matter whether update_p happens before or after calc likelihood)
            new_p, RPE, r = self.update_p(trials.iloc[i])
            trials.at[trials.index[i], 'new_p'] = new_p
            trials.at[trials.index[i], 'RPE'] = RPE
            trials.at[trials.index[i], 'r'] = r
            # update
            latents = self.calc_latents(trials.iloc[i], latents)
            #trials.at[trials.index[i], 'CA_sum'] = latents['CA_sum']
            trials.at[trials.index[i], 'LTA_sum'] = latents['LTA_sum']
            trials.at[trials.index[i], 'RPE_sum'] = latents['RPE_sum']

            
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
                
                
                
#             Reac_time=np.array([trials.at[trials.index[i], 'RT']])
#             v_mean=trials.at[trials.index[i], 'Ediff']
#             GAMMA=self.params['gamma']
#             BETA = self.params['beta']
#             A=self.params['a']
#             W=self.params['w']
#             T0=self.params['t0']
#             P0=self.params['p0']
#             Sv=self.params['s_v']
#             LAMBDA=self.params['lambda']
#             WLTA=self.params['w_LTA']
#             WRPE=self.params['w_RPE']
#             B=self.params['b']
            #import pdb; pdb.set_trace()
            if np.isnan(trials.at[trials.index[i], 'log_like']):
                #print((Reac_time,v_mean,GAMMA,BETA,P0,Sv,A,W,T0,LAMBDA,WLTA,WRPE,B), flush=True)
                raise ValueError("Log_Like NaN value") 
            if np.isnan(trials.at[trials.index[i], 'mood_log_like']):
                #print((Reac_time,v_mean,GAMMA,BETA,C,P0,Sv), flush=True)
                raise ValueError("Mood_Log_Like NaN value")
            if trials.at[trials.index[i], 'mood_log_like']==None:
                print([trials.index[i], 'mood_log_like'])
                print(trials.at[trials.index[i], 'log_like'])
                raise ValueError("Zero value")
            
        # return trials with useful columns added
        return trials

        
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
#     ind = (dat.Outcome2Amount > dat.Outcome1Amount)
#     dat.loc[ind, 'GreaterAmount'] = dat.loc[ind, 'Outcome2Amount']
#     dat.loc[ind, 'LesserAmount'] = dat.loc[ind, 'Outcome1Amount']
    dat['Won'] = dat['OutcomeAmount'] == dat['GreaterAmount']
    
    dat['win'] = dat.Won.astype(int)  
    Ntrial_idx = dat.index.values + 1
    dat['win_prob'] = np.cumsum(dat.win)/Ntrial_idx 
    
#     dat['actual_p'] = 0.7
#     dat.loc[dat['MoodTarget']==70, 'actual_p'] = 0.3

    return dat
