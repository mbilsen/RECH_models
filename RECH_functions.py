### File containing RECH model functions
## mc forecasting changed 01.03.2023

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
import yfinance as yf

def ql_loss(sigma_series, h_series):
    return np.mean( (sigma_series/h_series) - np.log(sigma_series/h_series) -1 )

def mse(sigma_series, h_series):
    return np.mean( (sigma_series - h_series)**2 )

def sigmoid(x):
    try:
        sig = 1 / (1 + math.exp(-x))
    except:
        sig = 0
    return sig

def relu(x, bound = 20):
    return min(max(0,x),bound)

def garch(pars, nun_lin_func, returns):
    (omega, alpha , beta) = pars
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = omega/(1- alpha - beta)
            # w[i] = 0.1/(1- alpha - beta)
        else:
            sigma_2[i] = nun_lin_func(omega) + alpha * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, 1, 1

def garch_loglike(start_v, nun_lin_func, returns):
    sigma_2 = garch(start_v, nun_lin_func, returns)[0]
    neg_LogL = - np.sum(-np.log(sigma_2) -  returns**2/sigma_2)
    return neg_LogL

def gjr(pars, nun_lin_func, returns):
    (omega, alpha, rho, beta) = pars
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = omega/(1 - alpha - 0.5 * rho - beta)
        else:
            sigma_2[i] = omega + alpha * returns[i-1]**2 + 0.5 * (returns[i-1] < 0) * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, 1, 1

def gjr_loglike(start_v, nun_lin_func, returns):
    sigma_2 = gjr(start_v, nun_lin_func, returns)[0]
    neg_LogL = - np.sum(-np.log(sigma_2) - returns**2/sigma_2)
    return neg_LogL


def SRN_garch(pars, nun_lin_func, returns):
    (alpha, beta, gamma0, gamma1, v_1, v_2, v_3, b) = pars
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    w = np.zeros(iT)
    h = np.zeros(iT)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = 0.1/(1- alpha - beta)
        else:
            h[i] = nun_lin_func(v_1 * np.sign(returns[i-1]) * returns[i-1]**2 + v_2 * sigma_2[i-1] + v_3 * h[i-1] + b)
            w[i] = gamma0 + gamma1 * h[i]
            sigma_2[i] = w[i] + alpha * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, w, h

def SRN_garch_loglike(start_v, nun_lin_func, returns):
    sigma_2 = SRN_garch(start_v, nun_lin_func, returns)[0]
    neg_LogL = - np.sum(-np.log(sigma_2) -  returns**2/sigma_2)
    return neg_LogL


def MGU_garch(pars, nun_lin_func, returns):
    (alpha, beta, gamma_0, gamma_1, v_11, v_12, v_21, v_22, mu_1, mu_2, b_h, b_z) = pars
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    w = np.zeros(iT)
    h = np.zeros(iT)
    h_hat = np.zeros(iT)
    z = np.zeros(iT)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = 0.1/(1- alpha - beta)
        else:
            z[i] = sigmoid(v_21 * np.sign(returns[i-1]) * returns[i-1]**2 + v_22 * sigma_2[i-1] + mu_2 * h[i-1] + b_z) # here logistic function instead of ReLU (for convex combination in h)
            h_hat[i] = nun_lin_func(v_11 * np.sign(returns[i-1]) * returns[i-1]**2 + v_12 * sigma_2[i-1] + mu_1 * z[i] * h[i-1] + b_h)
            h[i] = z[i] * h_hat[i] + (1-z[i]) * h[i-1]
            w[i] = gamma_0 + gamma_1 * h[i]
            sigma_2[i] = w[i] + alpha * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, w, h

def MGU_garch_loglike(start_v, nun_lin_func, returns):  
    sigma_2 = MGU_garch(start_v, nun_lin_func, returns)[0]
    LogL = - np.sum(-np.log(sigma_2) -  returns**2/sigma_2)
    return LogL

def GRU_garch(pars, nun_lin_func, returns):
    (alpha, beta, gamma_0, gamma_1, v_11, v_12, v_21, v_22, v_31, v_32, mu_1, mu_2, mu_3, b_h, b_r, b_z) = pars
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    w = np.zeros(iT)
    h = np.zeros(iT)
    h_hat = np.zeros(iT)
    r = np.zeros(iT)
    z = np.zeros(iT)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = 0.1/(1- alpha - beta)
        else:
            z[i] = sigmoid(v_31 * np.sign(returns[i-1]) * returns[i-1]**2 + v_32 * sigma_2[i-1] + mu_3 * h[i-1] + b_z) # here sigmoid instead of ReLU
            r[i] = sigmoid(v_21 * np.sign(returns[i-1]) * returns[i-1]**2 + v_22 * sigma_2[i-1] + mu_2 * h[i-1] + b_r)
            h_hat[i] = nun_lin_func(v_11 * np.sign(returns[i-1]) * returns[i-1]**2 + v_12 * sigma_2[i-1] + mu_1 * r[i] * h[i-1] + b_h)
            h[i] = z[i] * h_hat[i] + (1-z[i]) * h[i-1]
            w[i] = gamma_0 + gamma_1 * h[i] 
            sigma_2[i] = w[i] + alpha * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, w, h

def GRU_garch_loglike(start_v, nun_lin_func, returns):
    sigma_2 = GRU_garch(start_v, nun_lin_func, returns)[0]
    LogL = - np.sum(-np.log(sigma_2) -  returns**2/sigma_2)
    return LogL


def LSTM_garch(pars, nun_lin_func, returns):
    (alpha, beta, gamma_0, gamma_1, v_11, v_12, v_21, v_22, v_31, v_32, v_41, v_42, mu_1, mu_2, mu_3, mu_4, b_c, b_o, b_i, b_f) = pars
    iT = len(returns)
    sigma_2 = np.zeros(iT)
    w = np.zeros(iT)
    h = np.zeros(iT)
    f = np.zeros(iT)
    ij = np.zeros(iT)
    o = np.zeros(iT)
    c = np.zeros(iT)
    c_tilde = np.zeros(iT)
    for i in range(iT):
        if i == 0:
            sigma_2[i] = 0.1/(1- alpha - beta)
        else:
            f[i] = sigmoid(v_41 * np.sign(returns[i-1]) *  returns[i-1]**2 + v_42 * sigma_2[i-1] + mu_4 * h[i-1] + b_f) # here sigmoid instead of ReLU
            ij[i] = sigmoid(v_31 * np.sign(returns[i-1]) *  returns[i-1]**2 + v_32 * sigma_2[i-1] + mu_3 * h[i-1] + b_i)
            o[i] = sigmoid(v_21 * np.sign(returns[i-1]) *  returns[i-1]**2 + v_22 * sigma_2[i-1] + mu_2 * h[i-1] + b_o)
            c_tilde[i] = nun_lin_func(v_21 * np.sign(returns[i-1]) *  returns[i-1]**2 + v_22 * sigma_2[i-1] + mu_2 * h[i-1] + b_o)
            c[i] = f[i] * c[i-1] + ij[i] * c_tilde[i]
            h[i] = o[i] * c[i]
            w[i] = gamma_0 + gamma_1 * h[i] 
            sigma_2[i] = w[i] + alpha * returns[i-1]**2 + beta * sigma_2[i-1]
    return sigma_2, w, h

def LSTM_garch_loglike(start_v, nun_lin_func, returns):
    sigma_2 = LSTM_garch(start_v, nun_lin_func, returns)[0]
    LogL = - np.sum(-np.log(sigma_2) -  returns**2/sigma_2)
    return LogL



def mc_garch2(parameters, returns, sim_len, horizon, non_lin_func = relu):
    omega, alpha, beta = parameters
    #omega = omega/10000
    #parameters = omega, alpha, beta
    sigma_t = garch(parameters, non_lin_func, returns)[0][-1]
    if horizon == 1:
        #return omega + alpha * returns[-1]**2 + beta * sigma_t
        return sigma_t
    M = sim_len
    C = horizon
    y_cand_m = np.zeros(M)
    y_cand_c = 0
    for m in range(M):
        """calculate the path from t to t+H M-times"""
        for c in range(C):
            if c == 0:
                y_cand_c = np.random.normal(0,1) * sigma_t**0.5
                sigma_t = omega + alpha * y_cand_c**2 + beta * sigma_t
            else:
                y_cand_c = np.random.normal(0,1) * sigma_t**0.5
                sigma_t = omega + alpha * y_cand_c**2 + beta * sigma_t
        y_cand_m[m] = y_cand_c**2
    return np.mean(y_cand_m)

def mc_gjr2(parameters, returns, sim_len, horizon, non_lin_func = relu):
    omega, alpha, rho, beta = parameters
    #omega = omega/10000
    #parameters = omega, alpha, beta
    sigma_t = gjr(parameters, non_lin_func, returns)[0][-1]
    if horizon == 1:
        #return omega + alpha * returns[-1]**2 + beta * sigma_t
        return sigma_t
    M = sim_len
    C = horizon
    y_cand_m = np.zeros(M)
    y_cand_c = 0
    for m in range(M):
        """calculate the path from t to t+H M-times"""
        for c in range(C):
            if c == 0:
                y_cand_c = np.random.normal(0,1) * sigma_t**0.5
                sigma_t = omega + alpha * y_cand_c**2 + rho * (y_cand_c < 0) * y_cand_c**2 + beta * sigma_t
            else:
                y_cand_c = np.random.normal(0,1) * sigma_t**0.5
                sigma_t = omega + alpha * y_cand_c**2 + rho * (y_cand_c < 0) * y_cand_c**2 + beta * sigma_t
        y_cand_m[m] = y_cand_c**2
    return np.mean(y_cand_m)



def mc_srn2(parameters, nl_func, returns, sim_len, horizon):
    """monte carlo forecasting of srn garch model"""
    alpha, beta, gamma0, gamma1, v1, v2, w, b = parameters
    M = sim_len
    C = horizon
    nun_lin_func = nl_func
    sigma_t = SRN_garch(parameters, nl_func, returns)[0][-1]
    omega_t = SRN_garch(parameters, nl_func, returns)[1][-1]
    h_t = SRN_garch(parameters, nl_func, returns)[2][-1]
    if C == 1:
        return sigma_t, omega_t
    y_cand_m = np.zeros(M)
    #y_cand_c = np.zeros(C)
    omega_cand_m = np.zeros(M)
    for m in range(M):
        for c in range(C):
            y_cand_c = (np.random.normal(0,1) * sigma_t**0.5)
            h_t = nun_lin_func( v1 * np.sign(y_cand_c) * y_cand_c**2  + v2  * sigma_t + w * h_t + b )
            omega_t = gamma0 + gamma1 * h_t
            sigma_t = omega_t + alpha * y_cand_c**2 + beta * sigma_t
        y_cand_m[m] = y_cand_c**2
        omega_cand_m[m] = omega_t
    return np.mean(y_cand_m), np.mean(omega_cand_m)
            
    


def mc_mgu2(parameters, nl_func, returns, sim_len, horizon):
    alpha, beta, gamma_0, gamma_1, v_11, v_12, v_21, v_22, w_1, w_2, b_h, b_z = parameters
    
    sigma_t = MGU_garch(parameters, nl_func, returns)[0][-1]
    omega_t = MGU_garch(parameters, nl_func, returns)[1][-1]
    h_t = MGU_garch(parameters, nl_func, returns)[2][-1]
    if horizon == 1:
        return sigma_t, omega_t
    M = sim_len
    C = horizon
    nun_lin_func = nl_func
    y_cand_m = np.zeros(M) # to be filled with SQUARED y, SQUARED!! lengths is m 
    #y_cand_c = np.zeros(C) # to be filled with SQUARED y, SQUARED!! length is c, filled with averages of the simulated y_cand_m
    omega_cand_m = np.zeros(M)
    for m in range(M):
        for c in range(C):
            y_cand_c = (np.random.normal(0,1) * sigma_t**0.5)
            z_t = sigmoid( v_21 * np.sign(y_cand_c) * y_cand_c**2 + v_22 * sigma_t + w_1 * h_t + b_z)
            h_hat = nun_lin_func( v_11 * np.sign(y_cand_c) * y_cand_c**2 + v_12 * sigma_t + w_2 * h_t + b_h )
            h_t = z_t * h_hat + (1-z_t) * h_t
            omega_t = gamma_0 + gamma_1 * h_t
            sigma_t = omega_t + alpha * y_cand_c**2 + beta * sigma_t
        y_cand_m[m] = y_cand_c**2
        omega_cand_m[m] = omega_t
    return np.mean(y_cand_m), np.mean(omega_cand_m)


def mc_gru2(parameters, nl_func, returns, sim_len, horizon):
    alpha, beta, gamma_0, gamma_1, v_11, v_12, v_21, v_22, v_31, v_32, w_1, w_2, w_3, b_h, b_r, b_z = parameters
    sigma_t = GRU_garch(parameters, nl_func, returns)[0][-1]
    omega_t = GRU_garch(parameters, nl_func, returns)[1][-1]
    h_t = GRU_garch(parameters, nl_func, returns)[2][-1]
    if horizon == 1:
        return sigma_t, omega_t
    M = sim_len
    C = horizon
    y_cand_m = np.zeros(M)
    #y_cand_c = np.zeros(C)
    omega_cand_m = np.zeros(M)
    
    for m in range(M):
        for c in range(C):
            y_cand_c = np.random.normal(0,1) * sigma_t**0.5
            z_t = sigmoid( v_31 * np.sign(y_cand_c) * y_cand_c**2 + v_32 * sigma_t + w_3 * h_t + b_z)
            r_t = sigmoid( v_21 * np.sign(y_cand_c) * y_cand_c**2 + v_22 * sigma_t + w_2 * h_t + b_r)
            h_hat = nl_func( v_11 * np.sign(y_cand_c) * y_cand_c**2 + v_12 * sigma_t + w_1 * r_t * h_t + b_h )
            h_t = z_t * h_hat + (1-z_t) * h_t
            omega_t = gamma_0 + gamma_1 * h_t
            sigma_t = omega_t + alpha * y_cand_c**2 + beta * sigma_t
        y_cand_m[m] = y_cand_c**2
        omega_cand_m[m] = omega_t
    return np.mean(y_cand_m), np.mean(omega_cand_m)



def mc_lstm2(parameters, nl_func, returns, sim_len, horizon):
    alpha, beta, gamma_0, gamma_1, v_11, v_12, v_21, v_22, v_31, v_32, v_41, v_42, w_1, w_2, w_3, w_4, b_c, b_o, b_i, b_f = parameters
    sigma_t = LSTM_garch(parameters, nl_func, returns)[0][-1]
    omega_t = LSTM_garch(parameters, nl_func, returns)[1][-1]
    h_t = LSTM_garch(parameters, nl_func, returns)[2][-1]
    if horizon == 1:
        return sigma_t, omega_t
    M = sim_len
    C = horizon
    y_cand_m = np.zeros(M)
    #y_cand_c = np.zeros(C)
    omega_cand_m = np.zeros(M)
    c_rnn = 0
    for m in range(M):
        for c in range(C):
            y_cand_c = (np.random.normal(0,1) * sigma_t**0.5)
            f_t = sigmoid( v_41 * np.sign(y_cand_c) * y_cand_c**2 + v_42 * sigma_t + w_4 * h_t + b_f )
            i_t = sigmoid( v_31 * np.sign(y_cand_c) * y_cand_c**2 + v_32 * sigma_t + w_3 * h_t + b_i )
            o_t = sigmoid( v_21 * np.sign(y_cand_c) * y_cand_c**2 + v_22 * sigma_t + w_2 * h_t + b_o )
            c_tilde = nl_func( v_11 * np.sign(y_cand_c) * y_cand_c**2 + v_12 * sigma_t + w_1 * h_t + b_c )
            c_rnn = f_t * c_rnn + i_t * c_tilde
            h_t = o_t * c_rnn
            omega_t = gamma_0 + gamma_1 * h_t
            sigma_t = omega_t + alpha * y_cand_c**2 + beta * sigma_t
        y_cand_m[m] = y_cand_c**2
        omega_cand_m[m] = omega_t
    return np.mean(y_cand_m), np.mean(omega_cand_m)



