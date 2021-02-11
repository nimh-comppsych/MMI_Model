import StudentTMulti as st
import OCPD as oncpd
import numpy as np
from functools import partial

## POST and MAP calculations (DS)
def RL_POST_MultVar(dat,dim):
    
    ## Compute RPE
    x =((dat.Choice == 'Gamble') | (dat.Choice == 'None')).astype(int)
    dat['Gamble'] = x
    #EG1 = dat.GreaterAmount*winProb + dat.LesserAmount*(1-winProb)
    EG2 = dat.GreaterAmount*0.5 + dat.LesserAmount*0.5
    #RPE1 = (dat.OutcomeAmount.values - EG1)*x.values
    RPE2 = (dat.OutcomeAmount.values - EG2)*x.values
    #RPE3 = (dat.OutcomeAmount.values - LTA)*x.values
    dat['Ruth_RPE'] = (dat.OutcomeAmount - EG2)*x

    if dim == 1:
        prior = oncpd.StudentT(alpha=1, beta=1, kappa=1, mu=0)
        data = RPE2 ## Ruth. RPE
    if dim == 3:
        prior = st.StudentTMulti(dim)
        xx = dat[['GreaterAmount','LesserAmount','CertainAmount']]
        data = xx.values  ### xx.values contains 3D data
    if dim == 4:
        prior = st.StudentTMulti(dim)
        xx = dat[['GreaterAmount','LesserAmount','CertainAmount','Ruth_RPE']]
        data = xx.values  ### xx.values contains 4D data

    POST, maxes, RL_maxes = oncpd.online_changepoint_detection(data,partial(oncpd.constant_hazard, 27),prior)

    return POST, maxes, RL_maxes