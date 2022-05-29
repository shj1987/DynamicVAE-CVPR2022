#!/usr/bin/env python3
# coding: utf-8

from math import exp

class PIDControl():
    """incremental PID controller"""
    def __init__(self, Kp, Ki, initial_value):
        """define them out of loop"""
        self.W_k1 = initial_value
        self.W_min = 0.1 ##important, otherwise wind up.
        self.e_k1 = 0.0
        self.Kp = Kp
        self.Ki = Ki

    def _Kp_fun(self, Err, scale=1):
        return 1.0/(1.0 + float(scale)*exp(Err))
        
    def pid(self, exp_KL, kl_loss):
        """
        Incremental PID algorithm
        Input: KL_loss
        return: weight for KL divergence, beta
        """
        error_k = exp_KL - kl_loss
        ## comput U as the control factor
        dP = self.Kp * (self._Kp_fun(error_k) - self._Kp_fun(self.e_k1))
        dI = self.Ki * error_k
        
        if self.W_k1 < self.W_min:
            dI = 0
        dW = dP + dI
        ## update with previous W_k1
        Wk = dW + self.W_k1
        self.W_k1 = Wk        
        self.e_k1 = error_k
        
        ## min and max value
        if Wk < self.W_min:
            Wk = self.W_min
        
        return Wk, error_k
        
        