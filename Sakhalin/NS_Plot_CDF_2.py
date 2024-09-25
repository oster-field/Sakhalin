import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

pmin = 0.10
pmid = 0.4469770245124512
pmax = 0.7291797264228802
Fy = np.load('Find_CDF_F.npy')
Fx = np.load('Find_CDF_x.npy')
Fy0 = np.load('Find_CDF_F_2.npy')
Fx0 = np.load('Find_CDF_x_2.npy')
Fy1 = np.load('Find_CDF_F_3.npy')
Fx1 = np.load('Find_CDF_x_3.npy')
F = np.load('PARAMETER_ERF_F.npy')
x = np.load('PARAMETER_ERF_X.npy')

fig = plt.figure()
ax = fig.add_subplot(111)
'''ax.plot(Fx1, Fy1, color='g', alpha=.7, linewidth=2.5, label='$F_{A}, ϵ=0.1$')
ax.plot(Fx0, Fy0, color='b', alpha=.7, linewidth=2.5, label='$F_{A}, ϵ=0.44$')
ax.plot(Fx, Fy, color='r', alpha=.7, linewidth=2.5, label='$F_{A}, ϵ=0.73$')
ax.plot(Fx1, (np.exp(-2 * Fx1**2) / 2) * (1 + erf((Fx1 * np.sqrt(2 - 2 * pmin**2))/pmin)), linestyle='dotted', linewidth=2, color='g',  label='$F_{1}, ϵ=0.1$')
ax.plot(Fx0, (np.exp(-2 * Fx0**2) / 2) * (1 + erf((Fx0 * np.sqrt(2 - 2 * pmid**2))/pmid)), linestyle='dotted', linewidth=2, color='b', label='$F_{1}, ϵ=0.44$')
ax.plot(Fx, (np.exp(-2 * Fx**2) / 2) * (1 + erf((Fx * np.sqrt(2 - 2 * pmax**2))/pmax)), linestyle='dotted', linewidth=2, color='r', label='$F_{1}, ϵ=0.73$')'''
"""ax.plot(Fx1, Fy1 - (np.exp(-2 * Fx1**2) / 2) * (1 + erf((Fx1 * np.sqrt(2 - 2 * pmin**2))/pmin)), linestyle='--', linewidth=2, color='g', label='$F_{A} - F_{1}, ϵ=0.1$')
ax.plot(Fx0, Fy0 - (np.exp(-2 * Fx0**2) / 2) * (1 + erf((Fx0 * np.sqrt(2 - 2 * pmid**2))/pmid)), linestyle='--', linewidth=2, color='b', label='$F_{A} - F_{1}, ϵ=0.44$')
ax.plot(Fx, Fy - (np.exp(-2 * Fx**2) / 2) * (1 + erf((Fx * np.sqrt(2 - 2 * pmax**2))/pmax)), linestyle='--', linewidth=2, color='r', label='$F_{A} - F_{1}, ϵ=0.73$')
A1 = 14.4
A2 = 3.325
A3 = 2.6
ax.plot(Fx1, 1/2 * (1 - erf(A1 * Fx1)), linewidth=8, color='g', alpha=0.4, label='$γ; ψ=14.4 , ϵ=0.1$')
ax.plot(Fx0, 1/2 * (1 - erf(A2 * Fx0)), linewidth=8, color='b', alpha=0.4, label='$γ; ψ=3.3 , ϵ=0.44$')
ax.plot(Fx, 1/2 * (1 - erf(A3 * Fx)), linewidth=8, color='r', alpha=0.4, label='$γ; ψ=2.6 , ϵ=0.73$')"""

ax.scatter(0.10308760658103854, 13.7)
ax.scatter(0.20874363196567766, 6.7)
ax.scatter(0.3077790214346142, 4.6)
ax.scatter(0.40636088778860774, 3.5)
ax.scatter(0.5026597003239123, 2.75)
ax.scatter(0.598085356378237, 2.3)
ax.scatter(0.6994836364161559, 1.87)
ax.scatter(0.8083538893455501, 1.6)
ax.scatter(0.896852388991725, 1.35)
ax.scatter(0.9086482808563965, 1.3)

#  ax.scatter(0.9907, 1.25, s=6)
#  ax.plot(x, F, color='g', alpha=.5, linewidth=6, label='Numerical curve', marker='.')
x1 = np.arange(0.1, 1, 0.01)
psi = 1
ax.plot(x1, (np.sqrt(2) / x1) * psi, color='black', label='(√2(1 + ϵ⁴)) / ϵ', linewidth=2)
ax.set_xlabel('ϵ', fontsize=20)
ax.set_ylabel('ψ', fontsize=20)
ax.set_xlim([0, 1])
ax.tick_params(labelsize=20)
ax.legend(fontsize=15)
ax.grid()
plt.show()
