#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sympy


# In[ ]:


def MySeries(f,x,xc,N=5):
    ret=f.subs(x,xc)
    c=1
    power=1
    for i in range(1,N+1):
        p=sympy.diff(f,x,i).subs(x,xc)
        c*=i
        power*=(x-xc)
        ret += p/c*power
    return ret
    


# In[ ]:


import numpy as np
import csv
def getmat(filename):
    rowmx=list()
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for t in reader:
            #print(t)
            rowmx.append(t)
    x=rowmx[0]
    y=np.array([eval(t) for t in x])
    return np.array(np.array_split(y,8))

mx=getmat('save_mx.txt')    
my=getmat('save_my.txt')    
me=getmat('save_me.txt')    
mx=mx.transpose()
my=my.transpose()
me=me.transpose()
rowmx=list()
rowb=list()
with open('save_b.txt', 'r') as file:
    reader = csv.reader(file)
    for t in reader:
        rowb.append(t)
print(rowb)
ee,ww=np.linalg.eig(mx)
vv=ww.transpose()
vv.shape,my.shape,me.shape


# In[ ]:


import array_to_latex as a2l
A = np.array([[1.23456, 23.45678],[456.23, 8.239521]])
latexmx="m_x="+a2l.to_ltx(mx, frmt = '{:6.3f}', arraytype = 'pmatrix',print_out=False)
latexmy="m_y="+a2l.to_ltx(my, frmt = '{:6.3f}', arraytype = 'pmatrix',print_out=False)
latexme="m_e="+a2l.to_ltx(me, frmt = '{:6.3f}', arraytype = 'pmatrix',print_out=False)
#latexme=a2l.to_ltx(rowb[0], frmt = '{:6.3f}', arraytype = 'pmatrix')


# In[ ]:


print(latexmx)
print(latexmy)
print(latexme)


# Let $E_{j,k}$ be the matrix with 1 in the (j,k)-th entry and 0 elsewhere.
# The generators of the unitary transformation are given by
# $$
#     T(i,k)= E_{j,k}-E_{k,j}
# $$
# The unitary transformation is defined by
# 
# $$U(p)=\exp \left(p_{0,1}\cdot T(0,1)+p_{1,2}\cdot T(1,2)+\cdots +p_{N-1,N}\cdot T(N-1,N)+p_{N-1,N}\cdot T(N,1\}\right)$$
# with a parameter set $p$.
# The statevector in the VQE is computed by
# $$
# \ket{v_p}=U(p)\ket{v_0}
# $$
# from an intial state vector $\ket{v_0}$.

# $$
# b^T=(ye^3, ye^2, ye, y, e^3, e^2, e, 1)
# $$

# In[ ]:


ee


# In[ ]:


import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Pauli, SparsePauliOp
import matplotlib.pyplot as plt
import scipy
import numpy as np
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
matrix = [[0, 0, 0, 1],
          [0, 0, 1, 0],
          [1, 0, 0, 0],
          [0, 1, 0, 0]]

def amat(i,j):
    matrix=np.zeros((8,8))
    if i!=j:
        matrix[i,j]=1
        matrix[j,i]=-1
    return matrix
    

matrix=scipy.linalg.expm(-3*np.array(amat(0,1))+3*np.array(amat(1,2))+4*np.array(amat(2,3))+10*amat(3,0))


# In[ ]:


matrix.transpose()@matrix


# In[ ]:


me


# In[ ]:


x,y,e, R=sympy.symbols("x y e R")
OF=-2*e*(10000*x**2 + 9016*x*y + 10000*y**2 - 10000) + 13071*x**4 + 17494*x**3*y + 19208*x**2*y**2 - 53057*x**2 + 12474*x*y**3 - 53899*x*y + 7746*y**4 - 34640*y**2 + 13670


# In[ ]:


from sympy import exp

S=[[1.0,
  0.0195664191272055*exp(-2.53503275295838*R**2) + 0.076311941530018*exp(-1.1698731668665*R**2) + 0.027081935036154*exp(-0.586403975243215*R**2) + 0.2354269557701*exp(-0.461759323775025*R**2) + 0.0365514533608989*exp(-0.421651855735931*R**2) + 0.235025305061874*exp(-0.271557480345828*R**2) + 0.00902751015238458*exp(-0.165982731133525*R**2) + 0.100566314731487*exp(-0.15420094815142*R**2) + 0.162414036269465*exp(-0.124970698789036*R**2)],
 [0.0195664191272055*exp(-2.53503275295838*R**2) + 0.076311941530018*exp(-1.1698731668665*R**2) + 0.027081935036154*exp(-0.586403975243215*R**2) + 0.2354269557701*exp(-0.461759323775025*R**2) + 0.0365514533608989*exp(-0.421651855735931*R**2) + 0.235025305061874*exp(-0.271557480345828*R**2) + 0.00902751015238458*exp(-0.165982731133525*R**2) + 0.100566314731487*exp(-0.15420094815142*R**2) + 0.162414036269465*exp(-0.124970698789036*R**2),
  1.0]]


# In[ ]:


SV=np.array([[1,S[0][1].subs(R,1.46)],[S[1][0].subs(R,1.46),1]])
print("OVERLAP MATRIX S IN HARTREE-FOCK: USED LATER")
print(SV,S)


# In[ ]:


mx=getmat('save_mx.txt')    
my=getmat('save_my.txt')    
me=getmat('save_me.txt')    
mx=mx.transpose()
my=my.transpose()
me=me.transpose()

ee,ww=np.linalg.eig(mx)
vv=ww.transpose()
vv.shape,my.shape,me.shape


# In[ ]:


def totale(OF,xyer,C=10000):
    ret=OF.subs(x,xyer[0]).subs(y,xyer[1]).subs(e,xyer[2])
    return sympy.N(ret)/C


# In[ ]:


LE=list()
for v in vv:
    LE2=[np.dot(v.conj().transpose(),np.dot(A,v)) for A in [mx,my,me]]
    LE2=LE2+[complex(totale(OF, LE2))]
    LE.append(LE2)
import pandas as pdX
pdX.options.display.precision = 4
DFL=pdX.DataFrame(LE,columns=["$(i|m_x^T|i)$","$(i|m_y^T|i)$","$(i|m_e^T|i)$","TOTAL ENERGY"])
print(" Solutions of HF equations ")
print(DFL.applymap(lambda x: '{:,.4f}'.format(x)).to_latex())
#DFL.to_latex(float_format="%.2c")


# In[ ]:


v=vv[6]
vg=[np.dot(v.conj().transpose(),np.dot(A,v)) for A in [mx,my]]
print("Ground State (x,y)")
print(vg)


# In[ ]:


OF=-2*e*(10000*x**2 + 9016*x*y + 10000*y**2 - 10000) + 13071*x**4 + 17494*x**3*y + 19208*x**2*y**2 - 53057*x**2 + 12474*x*y**3 - 53899*x*y + 7746*y**4 - 34640*y**2 + 13670


# In[ ]:


print("Polynomial Objective Function")
print(OF)


# The objective function for the ground state is given by
# $$
# E_{total}(x_0,y_0,z_0)+\sum_{p=x,y,e} \left|\frac{\partial E_{total}}{\partial p}\right|^2_{x=x_0,y=y_0,e=e_0}
# $$

# In[ ]:


def amat(i,j):
    #print(i,j)
    matrix=np.zeros((8,8))
    if i!=j:
        matrix[i,j]=1
        matrix[j,i]=-1
    return matrix

def totale(OF,xyer,C=10000):
    ret=OF.subs(x,xyer[0]).subs(y,xyer[1]).subs(e,xyer[2])
    return sympy.N(ret)/C

def mv(param,args):
    #print(param)
    vec=args[0]
    matrix=np.zeros((8,8))
    for i,p in enumerate(param):
            matrix += p*amat(i, (i+1) % 8)
    matrix=scipy.linalg.expm(matrix)
    mv=np.dot(matrix,vec)
    mvn=np.sqrt(np.dot(mv,mv))
    v=mv/np.sqrt(mvn)
    LE=[np.real(np.dot(v.conj().transpose(),np.dot(A,v))) for A in [mx,my,me]]
    resnA=totale(OF,LE)
    resn=0
    for p in [x,y,e]:
        resn+=totale(sympy.diff(OF,p),LE)**2
 
    args[1].append([LE,resnA,resn])
    return resn

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
MONIT=list()
print("MIMIMIZATION OF THE OBJECTIVE FUNCTION (TOWARD GROUND STATE)")
args=(np.array([1,1,1,1,1,1,1,1]),MONIT)
x0=np.array([random.random()-0.5,random.random()-0.5,random.random()-0.5,random.random()-0.5,\
            random.random()-0.5,random.random()-0.5,random.random()-0.5,random.random()-0.5]), 
x0=np.array([-7.786e-01,  3.235e-01,  1.782e+00 , 6.400e-01 , 3.262e-02,
                 -1.399e-01, -2.724e-01 , 6.050e-02])
result = minimize(fun = mv,x0=x0,
                  args = (args,),
                  method="Nelder-Mead")
print(result)




# In[ ]:


import pandas as pd
pd.options.display.precision = 4
print("INFORMATION ON THE OPTIMUM")
print(pd.DataFrame([MONIT[-1][0]+[float(MONIT[-1][1])]],columns=['x','y','e','Total Energy']).to_latex(float_format="%.4f"))


# In[ ]:


print(\
"We would get the Ground State (x,y,e)\n[-0.8019009264604655, -0.33673901126749184, -1.59761591489665]")


# In[ ]:


print("The objective function for the excited states is given by\n\
$$\n\
\\sum_{p=x,y,e} \\left|\frac{\\partial E_{total}}{\\partial p}\\right|^2_{x=x_1,y=y_1,e=e_1}\
+|(x_0,y_0|S|x_1,y_1)|^2\n\
$$\n\
where $(x_0,y_0)$ and $(x_1,y_1)$ stand for the normalized wavefunctions in the ground and excited states,respectively. The second term curbs the path of the optimization so that it shall not arrive at the ground state. It might remain non-zero at the optima, because the excited state in this case is not the virtual orbital orthogonal to the ground state.")


# In[ ]:


def amat(i,j):
    #print(i,j)
    matrix=np.zeros((8,8))
    if i!=j:
        matrix[i,j]=1
        matrix[j,i]=-1
    return matrix

def totale(OF,xyer,C=10000):
    ret=OF.subs(x,xyer[0]).subs(y,xyer[1]).subs(e,xyer[2])
    return sympy.N(ret)/C

def mve(param,args):

    vec=args[0]
    matrix=np.zeros((8,8))
    for i,p in enumerate(param):
            matrix += p*amat(i, (i+1) % 8)
    matrix=scipy.linalg.expm(matrix)
    mv=np.dot(matrix,vec)
    mvn=np.sqrt(np.dot(mv,mv))
    v=mv/np.sqrt(mvn)

    LE=[np.real(np.dot(v.conj().transpose(),np.dot(A,v))) for A in [mx,my,me]]

    resnA=totale(OF,LE)
    resn=0
    for p in [x,y,e]:
        resn+=totale(sympy.diff(OF,p),LE)**2
    excited=args[2]
    v1=np.array(np.array([LE[0],LE[1]]))

    lv1=np.dot(v1,np.dot(SV,v1))
    lvg=np.dot(vg,np.dot(SV,vg))
    
    if excited==True:
        vdotv2vg=np.dot(v1,np.dot(SV,vg))/lv1/lvg
        resn+=vdotv2vg**2
        #print(LE,np.array([LE[0],LE[1]]),vg,vdotv2vg**2)
    args[1].append([LE,resnA,resn])
    return resn


# In[ ]:


MONIT=list()
print("MIMIMIZATION OF THE OBJECTIVE FUNCTION (EXCITED STATE)")
print("FIRST STEP TO FIND THE WHEREABOUT OF THE EXCITED STATE")
args=(np.array([1,0,0,0,0,0,0,0]),\
     MONIT,True)
x0=np.array([random.random()-0.5,random.random()-0.5,random.random()-0.5,random.random()-0.5,\
                      random.random()-0.5,random.random()-0.5,random.random()-0.5,random.random()-0.5])
x0=np.array([0.1,0,0,0,0,0,0,0.1])

result = minimize(fun = mve,
                  x0=x0, 
                  args = (args,),
                  method="Nelder-Mead")
x0=result.x
print("Optimum [ [x,y,e], Etotal, fobj ]")
print(MONIT[-1])
print("SECOND STEP TO DETERMINE THE LOCATION OF THE EXCITED STATE")
args=(np.array([1,0,0,0,0,0,0,0]),\
     MONIT,False)
result = minimize(fun = mve,
                  x0=x0, 
                  args = (args,),
                  method="Nelder-Mead")
x0=result.x
print("Optimum [ [x,y,e], Etotal, fobj ]")
print(MONIT[-1])


# In[ ]:


#print(matrix,"\n",matrix.transpose()@matrix)
print("THE ILLUSTRATION OF THE QUANTUM CIRCUIT THAT OBTAINES THE OPTIMUM")
circuit = QuantumCircuit(3)
param=x0
matrix=np.zeros((8,8))
for i,p in enumerate(param):
        matrix += p*amat(i, (i+1) % 8)
matrix=scipy.linalg.expm(matrix)
circuit.unitary(matrix, [0, 1,2])

stv1 = qi.Statevector.from_instruction(circuit)

circuit.decompose().decompose().draw()


# In[ ]:


circuit.decompose().decompose().draw('latex', filename='./file.jpg')


# In[ ]:


circuit.decompose().decompose().draw("mpl")

