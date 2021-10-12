# -*- coding: utf-8 -*-
"""
Created on Fri May 31 02:06:43 2019

@author: Michael Yin
"""

import numpy as np
def forward_prob(A=[], B=[], pi=[], O=[]):
    # Prevent the invalid input parameter
    if len(A) == 0 or len(B) == 0 or len(pi) == 0 or len(O) == 0:
        return None
    
    # Initializing alpha matrix
    numObservations, numStates = len(O), len(A)
    alpha = np.zeros((numStates, numObservations))
    
    # Dynamic programming for the forward probability
    alpha[:, 0] = B[:, O[0]] * pi
    
    for i in range(1, numObservations):
        for j in range(numStates):
            alpha[j, i] = np.sum(alpha[:, i-1] * A[:, j]) * B[j, O[i]]
    return np.sum(alpha[:, -1])

def backward_prob(A=[], B=[], pi=[], O=[]):
    # Prevent the invalid input parameter
    if len(A) == 0 or len(B) == 0 or len(pi) == 0 or len(O) == 0:
        return None
    
    # Initializing alpha matrix
    numObservations, numStates = len(O), len(A)
    beta = np.zeros((numStates, numObservations))
    
    # Dynamic programming for the backward probability
    beta[:, -1] = 1
    for i in range(numObservations - 2, 0 - 1, -1):
        for j in range(numStates):
            beta[j, i] = np.sum(A[j, :] * B[:, O[i + 1]] * beta[:, i + 1])
    return np.sum(beta[:, 0] * pi * B[:, O[0]])

# https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm

def viterbi_my(A=[], B=[], pi=[], O=[]):
    # Prevent the invalid input parameter
    if len(A) == 0 or len(B) == 0 or len(pi) == 0 or len(O) == 0:
        return None
    
    # Initializing some parameters
    numObservations, numStates = len(O), len(A)
    delta, deltaInd = np.zeros((numStates, numObservations)), np.zeros((numStates, numObservations))
    optimalPath = [None] * numObservations
    
    # Dynamic programming for the delta variable
    delta[:, 0] = pi * B[:, O[0]]
    for i in range(1, numObservations):
        for j in range(numStates):
            tmp = delta[:, i-1] * A[:, j]
            deltaInd[j, i] = np.argmax(tmp)
            delta[j, i] = max(tmp * B[j, O[i]])
    
    # Backtracking for the optimal path, and using list to simulate the stack
    optimalPath[-1] = np.argmax(delta[:, -1])
    for i in range(numObservations - 1, 0, -1):
        optimalPath[i-1] = int(deltaInd[optimalPath[-1], i])
    return optimalPath, delta, deltaInd

def viterbi(A, B, Pi, y):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2


    
if __name__ == "__main__":
#    # Long case
#    A = np.array([[0.5, 0.1, 0.4],
#                  [0.3, 0.5, 0.2],
#                  [0.2, 0.2, 0.6]])
#    
#    B = np.array([[0.5, 0.5],
#                  [0.4, 0.6],
#                  [0.7, 0.3]])
#    
#    pi = np.array([0.2, 0.3, 0.5])
#    
#    O = [0, 1, 0, 0, 1, 0, 1, 1]
    
    # Short case
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    
    pi = np.array([0.2, 0.4, 0.4])
    
    O = [0, 1, 0]
    
#    A = np.array([[0.55, 0.15, 0.3],
#                  [0.25, 0.3, 0.45],
#                  [0.3, 0.35, 0.35]])
#    
#    B = np.array([[0.2, 0.8],
#                  [0.8, 0.2],
#                  [0.6, 0.4]])
#    
#    pi = np.array([0.3, 0.3, 0.4])
#    
#    O = [0, 1, 1, 0, 0, 1, 1, 1]
    
    alpha = forward_prob(A, B, pi, O)
    beta = backward_prob(A, B, pi, O)
    x, T1, T2 = viterbi(A, B, pi, O)
    path, delta, deltaInd = viterbi_my(A, B, pi, O)
    
    print(alpha)
    print(beta)
    print(x, T1, T2)
    print(path, delta, deltaInd)