#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy
import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Pauli, SparsePauliOp
import matplotlib.pyplot as plt
import scipy
import numpy as np
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi

def amat(i,j):
    matrix=np.zeros((4,4))
    if i!=j:
        matrix[i,j]=1
        matrix[j,i]=-1
    return matrix
    

matrix=scipy.linalg.expm(-3*np.array(amat(0,1))+3*np.array(amat(1,2))+4*np.array(amat(2,3))+10*amat(3,0))


# In[2]:


matrix.transpose()@matrix


# In[9]:


matrixh = np.array([[0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])
matrixy = np.array([[0,0,1/2,0],
                    [0,0,0,1/2],
                    [1,0,0,0],
                    [0,1,0,0]])
matrixx = np.array([[0, 0, 0,-1/2],
                    [0, 0, -1/2,0],
                    [0, -1,0,   0],
                    [-1,0, 0,   0]])


# In[11]:


import array_to_latex as a2l
print("m_e")
a2l.to_ltx(matrixh, frmt = '{:6.2f}', arraytype = 'array')
print("m_x")
a2l.to_ltx(matrixx, frmt = '{:6.2f}', arraytype = 'array')
print("m_y")
a2l.to_ltx(matrixy, frmt = '{:6.2f}', arraytype = 'array')


# In[15]:


print("The objective function is given by \
$$\
f(v)=(v| m_e |v) +\| (v| (m_y - E_y)  |v) \|^2\
$$")


# \begin{align}
# x
# \end{align}

# In[16]:


def h(param,args):
#Minimize (vec|m_e|vec) with the constraint that |vec) is an eigenvector of m_y.
    m_e = np.array([[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])
    m_y = np.array([[0,0,1/2,0],
                        [0,0,0,1/2],
                        [1,0,0,0],
                        [0,1,0,0]])
    m_x = np.array([[0, 0, 0,-1/2],
                        [0, 0, -1/2,0],
                        [0, -1,0,   0],
                        [-1,0, 0,   0]])
    x,y,z,w,eigy=param
    vec=args[0]
    listS=args[1]
    matrix=scipy.linalg.expm(x*np.array(amat(0,1))+y*np.array(amat(1,2))+z*np.array(amat(2,3))+w*amat(3,0))
    mv=np.dot(matrix,vec)
    mvn=np.sqrt(np.dot(mv,mv))
    ymv=np.dot(matrixy,mv)-eigy*mv
    e=np.dot(mv,np.dot(m_e,mv))/mvn

    ey=np.dot(mv,np.dot(m_y,mv))/mvn
    ex=np.dot(mv,np.dot(m_x,mv))/mvn
    listS.append([ex,ey,e])
    return e+np.dot(ymv,ymv)/mvn


# In[28]:


import seaborn as sns
sns.set(context="paper" , style ="whitegrid",rc={"figure.facecolor":"white"})
from scipy.optimize import minimize
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(context="paper" , style ="whitegrid",rc={"figure.facecolor":"white"})

from scipy.optimize import minimize

import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
print("OPTIMIZATION START TO GET THE GROUND STATE")
listS=list()
args=(np.array([1,0,0,0]),listS)
result = minimize(fun = h,
                  x0=np.array([1,1,1,1,1]), 
                  args = (args,),
                  method="Nelder-Mead")
print(result)


# In[18]:


print("OPTIMUM : PARAMATER")
import array_to_latex as a2l
a2l.to_ltx(result.x, frmt = '{:6.4f}', arraytype = 'array')


# In[19]:


print("OPTIMUM : x,y,e")
print(pd.DataFrame(np.array([[x for x in listS[-1]]]),columns=["x","y","e"]).to_latex())


# In[26]:


print("AN ILLUSTRATION OF THE QUANTUM CIRCUIT")
circuit = QuantumCircuit(2)
x0=result.x
matrix=scipy.linalg.expm(x0[0]*np.array(amat(0,1))+x0[1]*np.array(amat(1,2))+x0[2]*np.array(amat(2,3))+x0[3]*amat(3,0))
circuit.unitary(matrix, [0, 1])

circuit.decompose().draw()

