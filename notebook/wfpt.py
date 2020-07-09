

import numpy as np
import scipy.stats.distributions as dists


def wfpt_like(choices, rts, v_mean, a, w_mode, w_std=0.0,
              v_std=0.0, t0=0.0, nsamp=5000, err=.0001, 
              max_time=5.0, trange_nsamp=1000):
    """
    Calculate WFPT likelihoods for choices and rts


    """
    # fill likes
    likes = np.zeros(len(choices))

    # process the v_mean and w_mode
    if w_std > 0.0:
        # calc with beta distribution
        mu = w_mode
        sigma = w_std
        kappa = mu * (1 - mu) / sigma**2 - 1
        alpha = mu * kappa
        beta = (1 - mu) * kappa

        if alpha <= 0.0 or beta <= 0.0:
            # illegal param
            return likes
        
        # sample from the beta distribution
        w = dists.beta(alpha, beta).rvs(nsamp)
    else:
        w = w_mode
    
    # proc the v
    if v_std > 0.0:
        v = dists.norm(v_mean, v_std).rvs(nsamp)[np.newaxis]
    else:
        v = v_mean
    
    # loop over the two choices
    # first choice 1, no change in v or w
    ind = np.where(choices == 1)[0]

    # loop over rts, setting likes for that choice
    for i in ind:
        # calc the like, adjusting rt with t0
        likes[i] = wfpt(rts[i]-t0, v=v, a=a, w=w, err=err)
        
       ### Debug 
#     if np.isnan(likes).any():
#         print(likes, flush=True)
#         print((rts[i]-t0, v, a, w, err), flush=True)
#         raise ValueError("likes is nan")
#     if (likes < 0).any():
#         raise ValueError("likes is negative")

    
    
    
    # then choice 2 with flip of v and w
    v = -v
    w = 1-w
    ind = np.where(choices == 2)[0]
    
    # loop over rts, setting likes for that choice
    for i in ind:
        # calc the like, adjusting rt with t0
        likes[i] = wfpt(rts[i]-t0, v=v, a=a, w=w, err=err)
    
#     ### Debug
#     if np.isnan(likes).any():
#         print(likes, flush=True)
#         print((rts[i]-t0, v, a, w, err), flush=True)
#         raise ValueError("likes is nan")
#     if (likes < 0).any():
#         raise ValueError("likes is negative")

    
    # finally the non-responses (the v and w can be either direction)
    ind = np.where(choices == 0)[0]
    for i in ind:
        likes[i] = wfpt_gen(v_mean=v_mean, a=a, w_mode=w_mode,
                            v_std=v_std, nsamp=nsamp,
                            trange=np.linspace(0, max_time-t0, trange_nsamp),
                            only_prob_no_resp=True)
        
#     ### Debug    
#     if np.isnan(likes).any():
#         print(likes, flush=True)
#         print((rts[i]-t0, v, a, w, err), flush=True)
#         raise ValueError("likes is nan")
#     if (likes < 0).any():
#         raise ValueError("likes is negative")

    
    return likes


def wfpt(t, v, a, w, err=.0001):
    """
    Wiener First Passage of Time

    Params
    ------
    t : reaction time
    v : drift rate
    a : boundary
    w : starting point
    err : algorithm tolearance

    Returns
    -------
    p : likelihood for the specified time and params

    Reference
    ---------
    
    https://compcogscisydney.org/publications/NavarroFuss2009.pdf

    """
    # this function is ported from R, hence the comments
    # if(t>0){
    if t <= 0.0:
        return 0.0

    ### Modification
    if np.isnan(v):
        return 0.0
    
    if v==np.inf or v==-np.inf:
        return 0.0
    
    # make w and v 2d
    w = np.atleast_2d(w)
    v = np.atleast_2d(v)
    ## Debug
    ##import pdb; pdb.set_trace()
    # tt=t/(a^2)
    tt = t/(a**2)
    # if(pi*tt*err<1){
    if (np.pi*tt*err) < 1.0:
        # kl=sqrt(-2*log(pi*tt*err)/(pi^2*tt))
        kl = np.sqrt(-2.*np.log(np.pi*tt*err)/(np.pi**2*tt))
        # kl=max(kl,1/(pi*sqrt(tt)))
        kl = max(kl, 1/(np.pi * np.sqrt(tt)))
    # } else {
    else:
        # kl=1/(pi*sqrt(tt))
        kl = 1 / (np.pi * np.sqrt(tt))
        # }
    # if(2*sqrt(2*pi*tt)*err<1){
    if (2*np.sqrt(2.*np.pi*tt)*err) < 1.0:
        # ks=2+sqrt(-2*tt*log(2*sqrt(2*pi*tt)*err))
        ks = 2 + np.sqrt(-2. * tt * np.log(2. * np.sqrt(2*np.pi*tt)*err))
        # ks=max(ks,sqrt(tt)+1)
        ks = max(ks, np.sqrt(tt) + 1)
    # } else {
    else:
        # ks=2
        ks = 2.0
        # }
    # p=0
    p = 0.0
    # if(ks<kl){
    if ks < kl:
        # K=ceiling(ks)
        K = np.ceil(ks)
        # along=seq(-floor((K-1)/2),ceiling((K-1)/2),1)
        along = np.arange(-np.floor((K-1)/2),
                          np.ceil((K-1)/2)+1)[:, np.newaxis] ##Added 1 # 
        # for(k in 1:length(along)){
        # p=p+(w+2*along[k])*exp(-((w+2*along[k])^2)/2/tt)
        # }
        p = np.sum((w + 2 * along) *
                   np.exp(-((w + 2. * along)**2) / 2. / tt), 0)

        # p=p/sqrt(2*pi*tt^3)
        p /= np.sqrt(2.*np.pi*tt**3)

    # } else {
    else:
        # K=ceiling(kl)
        K = np.ceil(kl)

        # for(k in 1:K){
        # p=p+k*exp(-(k^2)*(pi^2)*tt/2)*sin(k*pi*w)
        # }
        along = np.arange(1, K+1)[:, np.newaxis]
        p = np.sum(along * np.exp(-(along**2) * (np.pi**2) * tt/2.) *
                   np.sin(along * np.pi * w), 0)
        # p=p*pi
        p *= np.pi
        ## Debug
        #import pdb; pdb.set_trace()
        # }
    
    ### Modification
    xxx=np.exp(-v * a * w - (v**2) * t / 2)
    if xxx==np.inf or np.isnan(xxx):
        return 0.0
    
    
    # out=p*exp(-v*a*w -(v^2)*t/2)/(a^2)
    out = p * np.exp(-v * a * w - (v**2) * t / 2) / (a**2)
    
    ## Debug
    #print((ks,kl,xxx,p,out,out.mean()), flush=True)
    
    return out.mean()


def wfpt_gen(v_mean, a, w_mode, w_std=0.0,
             v_std=0.0, wfpt_nsamp=5000,
             err=.0001, nsamp=1000, trange=None, only_prob_no_resp = False):
    # generate default range if not provided
    if trange is None:
        trange = np.linspace(0, 5.0, 1000)

    # calc cdf of each
    dx = trange[1] - trange[0]
    rts = np.concatenate([trange, trange, [-1.]])
    ntimes = len(trange)
    choices = np.array([1]*ntimes + [2]*ntimes + [0])

    likes = wfpt_like(choices[:-1], rts[:-1],
                      v_mean, a, w_mode, w_std=w_std,
                      v_std=v_std, t0=0.0, nsamp=wfpt_nsamp, err=err)

    if only_prob_no_resp:
        p_non_resp = np.clip(1 - (likes*dx).sum(), 0, 1)
        return p_non_resp
    else:
        cdfs = np.concatenate([(likes*dx).cumsum(), [1.0]])

        # draw uniform rand numbers to determine choices and rts
        inds = [(cdfs > np.random.rand()).argmax() for i in range(nsamp)]

        return choices[inds], rts[inds]
    
