# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 18:11:29 2014

@author: jarenas
"""

import numpy as np
from itertools import product
from scipy import spatial
import sys


def dd_hellinger(theta1,theta2):
    
    """ Calcula la distancia de Hellinger entre distribuciones discretas. 
    
    Parametros de entrada:
        * theta1 :        Matriz de dimensiones (n1 x K)
        * theta2 :        Matriz de dimensiones (n2 x K)
        
    Devuelve: Una matriz de dimensiones (n1 x n2), donde cada componente
    se obtiene como la distancia de Hellinger entre las correspondientes filas
    de theta1 y theta2    
    """    
    _SQRT2 = np.sqrt(2)    
    
    (n1, col1) = theta1.shape
    (n2, col2) = theta2.shape
    if col1 != col2:
        sys.exit("Error en llamada a Hellinger: Las dimensiones no concuerdan")
    return spatial.distance.cdist(np.sqrt(theta1),np.sqrt(theta2),'euclidean') / _SQRT2
    
    
def dd_cosine(theta1,theta2):
    
    """ Calcula la distancia coseno entre distribuciones discretas. 
    
    Parametros de entrada:
        * theta1 :        Matriz de dimensiones (n1 x K)
        * theta2 :        Matriz de dimensiones (n2 x K)
        
    Devuelve: Una matriz de dimensiones (n1 x n2), donde cada componente
    se obtiene como la distancia coseno entre las correspondientes filas
    de theta1 y theta2    
    """
    (n1, col1) = theta1.shape
    (n2, col2) = theta2.shape
    if col1 != col2:
        sys.exit("Error en llamada a D. Coseno: Las dimensiones no concuerdan")
    #Normalize to get output between 0 and 1
    return spatial.distance.cdist(theta1,theta2,'cosine')/2


def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    (should only be used for the Jensen-Shannon divergence)
 
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    #print np.all([p != 0,q!= 0],axis=0)
    #Notice standard practice would be that the p * log(p/q) = 0 for p = 0,
    #but p * log(p/q) = inf for q = 0. We could use smoothing, but since this
    #function will only be called to calculate the JS divergence, we can also
    #use p * log(p/q) = 0 for p = q = 0 (if q is 0, then p is also 0)
    return np.sum(np.where(np.all([p != 0,q!= 0],axis=0), p * np.log(p / q), 0))


def dd_js(theta1,theta2):
    """ Calcula la distancia de Jensen-Shannon entre distribuciones discretas. 
    
    Parametros de entrada:
        * theta1 :        Matriz de dimensiones (n1 x K)
        * theta2 :        Matriz de dimensiones (n2 x K)
        
    Devuelve: Una matriz de dimensiones (n1 x n2), donde cada componente
    se obtiene como la distancia de J-S entre las correspondientes filas
    de theta1 y theta2    
    """
    (n1, col1) = theta1.shape
    (n2, col2) = theta2.shape
    if col1 != col2:
        sys.exit("Error en llamada a D. JS: Ambas matrices no tienen las mismas columnas")

    js_div = np.empty( (n1,n2) )
    for idx,pq in zip(product(range(n1),range(n2)),product(theta1,theta2)):
        av = (pq[0] + pq[1])/2
        js_div[idx[0],idx[1]] = 0.5 * (kl(pq[0],av) + kl(pq[1],av))
        
    return js_div