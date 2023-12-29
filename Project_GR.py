# Authors: Szymon Drzazga, Maciej Kucab, Ksymena Poradzis
# Affiliation: Jagiellonian University
#Contact: Ksymena Poradzisz: ksymena.poradzisz@gmail.com
#         Maciej Kucab: maciejkucab@duck.com
#         Szymon Drzazga: 
#Last Update: [28.12.2023]
#Description: 

import numpy as np 


def F8_LHS(r,u):
# D_u (sqrt(1-u^2) phi^6 K^r_r) + (e^eta/(sqrt(1-u^2))) D_u ((1-u^2)e^(-eta)phi^6 K^phi_phi

    return 0

def F8_RHS(r,u):
# sqrt(1-u^2)/r^2 D_r (r^2 phi^6 K^r_theta)

    return 1


def F12_LHS(r):
    #dr_beta_rr  =  partial_r (beta/r)
    #dtheta_beta_phi = partial_theta beta_theta
    return r * dr_beta_rr - dtheta_beta_phi

def F12_RHS(r):
    alpha
