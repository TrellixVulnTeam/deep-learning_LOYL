import numpy as np

epsilon = .00001

P = np.array([1./3, 1./3, 1./3, epsilon])
Q = np.array([1./4, 1./4, 1./4, 1./4])
print(P)
print(Q)

dkl_Q_to_P = sum(Q * np.log(Q/P))
dkl_P_to_Q = sum(P * np.log(P/Q))

print(dkl_Q_to_P)
print(dkl_P_to_Q)
