#!/usr/bin/env python
# coding: utf-8

# # MMI Model with Mood
# ## Authors: Per B. Sederberg & Adam W. Fenton & Charles Zheng

# In[1]:


# load matplotlib inline mode
#get_ipython().run_line_magic('matplotlib', 'inline')
import click

@click.command()
@click.argument('subj_input_dir')
@click.argument('subj_out_dir')
@click.argument('sid')


# load matplotlib  inline mode
#matplotlib inline
def mmi_model_Mood_LTA_gamma_wo_RT_p0_beta(subj_input_dir,subj_out_dir,sid):
    # import some useful libraries
    import numpy as np                # numerical analysis linear algebra
    import pandas as pd               # efficient tables
    #import matplotlib.pyplot as plt   # plotting
    from scipy import stats

    from RunDEMC.density import kdensity
    from RunDEMC import Model, Param, dists, calc_bpic, joint_plot

    from mmi_mood_LTA_gamma_Pwin_wo_RT_p0_beta_Random import MMIModel, load_mmi_data

    from joblib import Parallel, delayed

    import pickle
    from pathlib import Path

    try:
        import scoop
        from scoop import futures
    except ImportError:
        print("Error loading scoop, reverting to joblib.")
        scoop = None


    # ## Load and examine data


    subj_input_dir=Path(subj_input_dir)
    subj_out_dir=Path(subj_out_dir)
    
    
    pattern=f'*{sid}*.xlsx'
    input_file=list(subj_input_dir.glob(pattern))
    print(input_file[0])
    dat = load_mmi_data(input_file[0])


    # In[4]:


    #dat.at[2, 'Choice']=='Gamble'


    # In[5]:


    # find the minimum RT for the param range
    min_RT = dat.loc[dat.RT>0, 'RT'].min()
    print('Min RT:', min_RT)


    # ## Use RunDEMC to fit the model to a participant's data
    # define model evaluation functions for RunDEMC
    def eval_mod(params, param_names, bdat=None):
        # use global dat if none based in
        if bdat is None:
            bdat = dat

        # turn param list into dict
        mod_params = {x: params[n]
                      for n, x in enumerate(param_names)}
    #     try:
    #         print(mod_params['lambda'])
    #     except:
    #         print("Problems")
#         if mod_params['lambda']<0 or mod_params['lambda']>=1 :
#             return np.log(0), np.log(0)

        if mod_params['gamma']<0 or mod_params['gamma']>=np.inf :
            return np.log(0), np.log(0)

#         if mod_params['beta']<0 or mod_params['beta']>1 :
#             return np.log(0), np.log(0)

#         if mod_params['p0']<0 or mod_params['p0']>1 :
#             return np.log(0), np.log(0)

#         if mod_params['w_LTA']<0 or mod_params['w_LTA']>=np.inf :
#             return np.log(0), np.log(0)

#         if mod_params['w_RPE']<0 or mod_params['w_RPE']>=np.inf :
#             return np.log(0), np.log(0)

      ### Modification P_win
        if mod_params['w_p']<0 or mod_params['w_p']>=np.inf :
            return np.log(0), np.log(0)

        if mod_params['b']<=-np.inf or mod_params['b']>=np.inf :
            return np.log(0), np.log(0)

        if mod_params['s_v']<=0 or mod_params['s_v']>=np.inf :
            return np.log(0), np.log(0)

        if mod_params['c']<0 or mod_params['c']>=np.inf :
            return np.log(0), np.log(0) 


        ## calculate the log_likes and mood_log_likes 
        mod_res = MMIModel(params=mod_params, 
                      ignore_non_resp=True).proc_trials(bdat.copy())
        ll = mod_res.log_like.sum()
        mood_ll = mod_res.mood_log_like.sum()

        return ll,mood_ll

    # this is the required def for RunDEMC
    def eval_fun(pop, *args):
        bdat = args[0]
        pnames = args[1]
        if scoop and scoop.IS_RUNNING:
            res_tups = list(futures.map(eval_mod, [indiv for indiv in pop],
                                     [pnames]*len(pop), [bdat]*len(pop)))
        else:
            res_tups= Parallel(n_jobs=-1)(delayed(eval_mod)(indiv,pnames, bdat)
                                      for indiv in pop)

            likes = np.array([rt[0] for rt in res_tups])
            mood_likes = np.array([rt[1] for rt in res_tups])


        return likes, mood_likes


    
    def get_best_fit(m, burnin=250, verbose=True):
        best_ind = m.weights[burnin:].argmax()
        if verbose:
            print("Weight:", m.weights[burnin:].ravel()[best_ind])
        indiv = [m.particles[burnin:,:,i].ravel()[best_ind] 
                 for i in range(m.particles.shape[-1])]
        pp = {}
        for p,v in zip(m.param_names,indiv):
            pp[p] = v
            if verbose:
                print('"%s": %f,'%(p,v))
        return pp


    # ### Base  model

    # In[20]:


        # set up model params
    params = [Param(name='gamma',
                    display_name=r'$\gamma$',
                    prior=dists.gamma(1.5, 0.5),
                    ),
              Param(name='c',
                    display_name=r'$c$',
                    prior=dists.gamma(1.5, 0.5),
                    ),
#               Param(name='beta',
#                     display_name=r'$\beta$',
#                     prior=dists.normal(0, 1.4),
#                     transform=dists.invlogit
#                     ),
    #           Param(name='w',
    #                 display_name=r'$w$',
    #                 prior=dists.normal(0, 1.4),
    #                 transform=dists.invlogit
    #                 ),
    #           Param(name='a',
    #                 display_name=r'$a$',
    #                 prior=dists.gamma(2.0, 0.5),
    #                 ),
    #           Param(name='t0',
    #                 display_name=r'$t_0$',
    #                 prior=dists.normal(0, 1.4),
    #                 transform=lambda x: dists.invlogit(x)*min_RT,
    #                 ),
#               Param(name='p0',
#                     display_name=r'$p_0$',
#                     prior=dists.normal(0, 1.4),
#                     transform=dists.invlogit
#                     ),
#               Param(name='lambda',
#                     display_name=r'$lambda',
#                     prior=dists.normal(0, 1.4),
#                     transform=dists.invlogit
#                     ),
    #           Param(name='lambda',
    #                 display_name=r'$lambda',
    #                 prior=dists.beta(0.5, 0.5)
    #                 ),
#               Param(name='w_LTA',
#                     display_name=r'w_{LTA}',
#                     prior=dists.normal(0, 1),
#                     transform=np.exp,
#                     inv_transform=np.log),
    #           Param(name='w_EG',
    #                 display_name=r'w_{EG}',
    #                 prior=dists.normal(0, 1),
    #                 transform=np.exp,
    #                 inv_transform=np.log),
#               Param(name='w_RPE',
#                     display_name=r'w_{RPE}',
#                     prior=dists.normal(0, 1),
#                     transform=np.exp,
#                     inv_transform=np.log),
              Param(name='w_p',
                    display_name=r'w_{p}',
                    prior=dists.normal(0, 1),
                    transform=np.exp,
                    inv_transform=np.log),
              Param(name='b',
                    display_name=r'b',
                    prior=dists.normal(0, 3)),
              Param(name='s_v',
                    display_name=r's_v',
                    prior=dists.exp(3))
            ]
    
    
    # grab the param names
    pnames = [p.name for p in params]
    m = Model('mmi', params=params,
              like_fun=eval_fun,
              like_args=(dat, pnames),
              init_multiplier=3,
              # num_chains=gsize,
              verbose=True)

    # do some burnin
    times = m.sample(150, burnin=True)

    # now map the posterior
    times = m.sample(650, burnin=False)
    
    
     # show the chains are mixing and converging
    #plt.plot(m.weights[30:], alpha=.3);
    pickle_name=subj_out_dir / f'mWgt_{sid}.pickle'
    print(pickle_name)
    pickle_out = open(pickle_name,"wb")
    pickle.dump(m.weights, pickle_out)
    pickle_out.close()

    #print("Best fitting params:")
    #pp = get_best_fit(m, burnin=250, verbose=True)
    print("Best fitting params:")
    pp = get_best_fit(m, burnin=250, verbose=True)
    pickle_name=subj_out_dir / f'mBFprm_{sid}.pickle'
    print(pickle_name)
    pickle_out = open(pickle_name,"wb")
    pickle.dump(pp, pickle_out)
    pickle_out.close()

    # In[42]:
    #### BPIC calculations
    burnin=250
    
    log_like_prior = m.weights - m.log_likes
    #print(log_like_prior)
    weight_mood = m.posts + log_like_prior #m.posts is log_like_Mood
    #print(weight_mood)
    print("Mood_BPIC :",calc_bpic(weight_mood[burnin:])['bpic'])
    Mood_BPIC=calc_bpic(weight_mood[burnin:])['bpic']
    pickle_name=subj_out_dir / f'Mood_BPIC_{sid}.pickle'
    print(pickle_name)
    pickle_out = open(pickle_name,"wb")
    pickle.dump(Mood_BPIC, pickle_out)
    pickle_out.close()

    log_like_RT = m.log_likes - m.posts
    weight_RT = log_like_RT + log_like_prior
    print("RT_BPIC :",calc_bpic(weight_RT[burnin:])['bpic'])
    RT_BPIC=calc_bpic(weight_RT[burnin:])['bpic']
    pickle_name=subj_out_dir / f'RT_BPIC_{sid}.pickle'
    print(pickle_name)
    pickle_out = open(pickle_name,"wb")
    pickle.dump(RT_BPIC, pickle_out)
    pickle_out.close()

   
    print(calc_bpic(m.weights[burnin:])['bpic'])  
    mBPIC=calc_bpic(m.weights[burnin:])['bpic']
    pickle_name=subj_out_dir / f'Total_BPIC_{sid}.pickle'
    print(pickle_name)
    pickle_out = open(pickle_name,"wb")
    pickle.dump(mBPIC, pickle_out)
    pickle_out.close()




    # In[46]:


    # plot the param distributions (note, we did not get full posteriors, yet)
#     plt.figure(figsize=(12,10))
#     burnin=30
#     for i in range(len(m.param_names)):
#         plt.subplot(3,4,i+1)
#         plt.hist(m.particles[burnin:, :, i].flatten(), bins='auto', alpha=.5, density=True);
#         plt.title(m.param_names[i])


    # In[ ]:

    pickle_name=subj_out_dir / f'mParticles_{sid}.pickle'
    print(pickle_name)
    pickle_out = open(pickle_name,"wb")
    pickle.dump(m.particles, pickle_out)
    pickle_out.close()
                        
    pickle_name=subj_out_dir / f'mParams_{sid}.pickle'
    print(pickle_name)
    pickle_out = open(pickle_name,"wb")
    pickle.dump(m.param_names, pickle_out)
    pickle_out.close()                    
                        




    
if __name__ == '__main__':
    mmi_model_Mood_LTA_gamma_wo_RT_p0_beta()                        


