# Authors: Szymon Drzazga, Maciej Kucab, Ksymena Poradzis
# Affiliation: Jagiellonian University
#Contact: Ksymena Poradzisz: ksymena.poradzisz@gmail.com
#         Maciej Kucab: maciejkucab@duck.com
#         Szymon Drzazga: 
#Last Update: [28.12.2023]
#Description: 

import numpy as np 








def F12_LHS(r):
    #dr_beta_rr  =  partial_r (beta/r)
    #dtheta_beta_phi = partial_theta beta_theta
    return r * dr_beta_rr - dtheta_beta_phi

def F12_RHS(r):
    alpha
