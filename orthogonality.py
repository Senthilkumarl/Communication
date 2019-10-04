from math import sqrt, pi
import numpy as np
import scipy
import matplotlib.pyplot as plt



dt = 0.001
f = 10
T = 1/f
N = 1
IsThreeScalar = False
Phi = [lambda t:sqrt(2/T)*np.cos( 2.0*pi*f*t + pi/4), lambda t:sqrt(2/T)*np.cos( 2.0*pi*f*t - pi/4)]
Phi = [lambda t:sqrt(2/T)*np.cos( 2.0*pi*f*t), lambda t:sqrt(2/T)*np.sin( 2.0*pi*f*t)]
if IsThreeScalar:
    Phi = [lambda t:sqrt(2/T)*np.cos( 2.0*pi*f*t), lambda t:sqrt(2/T)*np.cos( 2.0*pi*2*f*t), lambda t:sqrt(2/T)*np.cos( 2.0*pi*3*f*t)]

for i in range(N):
     t=np.arange(i*T,(i+1)*T+dt,dt)

     #One scalar
     x = 2
     Y = x*Phi[0](t)
     x_det = np.trapz(Y*Phi[0](t),dx=dt)
     print("One scalar : ",x_det)

     #Two scalar
     print("1*Phi1 = ",np.trapz(Phi[1](t)*Phi[1](t),dx=dt))
     print("Phi0*Phi0 = ",np.trapz(Phi[0](t)*Phi[0](t),dx=dt))
     print("Phi0*Phi1 = ",np.trapz(Phi[0](t)*Phi[1](t),dx=dt))
     print("Phi1*Phi0 = ",np.trapz(Phi[1](t)*Phi[0](t),dx=dt))
     x1, x2 = 2,12
     Y = x1*Phi[0](t) + x2*Phi[1](t)
     x1_det = np.trapz(Y*Phi[0](t),dx=dt)
     x2_det = np.trapz(Y*Phi[1](t),dx=dt)
     print("Two scalar : ",x1_det, x2_det)

     #Three scalar
     if IsThreeScalar:
         print("Phi0*Phi0 = ",np.trapz(Phi[0](t)*Phi[0](t),dx=dt))
         print("Phi1*Phi1 = ",np.trapz(Phi[1](t)*Phi[1](t),dx=dt))
         print("Phi2*Phi2 = ",np.trapz(Phi[2](t)*Phi[2](t),dx=dt))
         print("Phi0*Phi1 = ",np.trapz(Phi[0](t)*Phi[1](t),dx=dt))
         print("Phi0*Phi2 = ",np.trapz(Phi[0](t)*Phi[2](t),dx=dt))
         print("Phi1*Phi0 = ",np.trapz(Phi[1](t)*Phi[0](t),dx=dt))
         print("Phi1*Phi2 = ",np.trapz(Phi[1](t)*Phi[2](t),dx=dt))
         print("Phi2*Phi0 = ",np.trapz(Phi[2](t)*Phi[0](t),dx=dt))
         print("Phi2*Phi1 = ",np.trapz(Phi[2](t)*Phi[1](t),dx=dt))

         x1, x2, x3 = 2,12,26
         Y = x1*Phi[0](t) + x2*Phi[1](t) + + x3*Phi[2](t)
         x1_det = np.trapz(Y*Phi[0](t),dx=dt)
         x2_det = np.trapz(Y*Phi[1](t),dx=dt)
         x3_det = np.trapz(Y*Phi[2](t),dx=dt)
         print("Three scalar : ",x1_det, x2_det, x3_det)
