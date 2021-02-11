## Context Models ##

# The "mmi_mood_LTA_gamma_0_Pwin_wo_RT_p0_beta_WinContext_OL.py" file can be used to calculate 3 types of conetext models by uncommenting the related part (please see under the "load_mmi_data(filename)" function) of the codes and commenting unrelated part of the codes. For example, the default set up of the code will calculate  the "Dominant-Time" context model only. But one can easily calculate other types just by uncommenting the "MAP"( inside the codes, please see under "winProb_context_Dynamic_Timeline_MAP") or "POST" (inside the codes, please see under "Win_Prob_Context_Dynamic_TimeLine_POST") part of the codes and commenting the dominant time-line part (inside the codes, pleae see under winProb_context Dominant_time_line) of the codes. 

# Inside the codes, 'dim' refers to the no. of VAR are used to calculate POST or MAP. 'dim' can take values 1(RPE), 3(Outcome Values/Certain), and 4(Values+RPE). By default, dim = 1.

# For convenience, I put 'DS' initial in the places where I made changes in the codes and created new functions. I hope it will make the verificaton of the codes much easier.


# The "mmi_mood_LTA_gamma_0_Pwin_wo_RT_p0_beta_WinContext_OL.py" file can also be used for the non-contextual model calculations. It can simply be done just by uncommenting the non-contextual model part (inside the codes, pleae see under "Non-Contextual win-prob Model") and commenting other context model parts of the codes.


