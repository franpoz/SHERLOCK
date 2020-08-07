import numpy as np
import time
import matplotlib.pyplot as plt
import os

#
# Script para calcular las HZ basado en las ecuaciones de 
# kopparapu et al (2013)
#
# Estas ecuaciones son validas para estrellas entre 2600<Teff<7200k
# Hay que dar la Teff y la L de la estrella en cuestion
#


Teff = 2983. #temperatura effectiva de la estrella que estemos considerando
L = 0.001 #luminosidad en L_sun
L_sun = 1.
Ts = Teff-5780.

#ecuaciones (2) y (3) en kopparapu et al 2013, para orbitas no eccentricas
def Seff(Seff_sun,a,b,c,d):
    Seff = Seff_sun + a*Ts + b*Ts**2 + c*Ts**3 + d*Ts**4
    return Seff

def Dis(Seff):
    Dis = (L/L_sun/Seff)**0.5
    return Dis

#considerando ecuaciones (3) y (4) para orbitas eccentricas en kopparapu et al 2013
def Seff_prime(A,e):
    Seff_prime=A/np.sqrt(1.-e**2)
    return Seff_prime

##
## coeficientes para calcular los flujos estelares (tabla 3 en kopparapu et al 2013)
##

e_end = 1.00
Nout = 10000
e = np.linspace(0.0001,e_end,Nout,endpoint=False)

#Recent Venus

s_eff_rv = 1.7763
a_rv = 1.4335e-4
b_rv = 3.3954e-9
c_rv = -7.6364e-12
d_rv = -1.1950e-15
Seff_rv=Seff(s_eff_rv,a_rv,b_rv,c_rv,d_rv)
Dis_rv=Dis(Seff_rv)
Seff_prime_rv=Seff_prime(Seff_rv,e)
Dis_prime_rv=Dis(Seff_prime_rv)
print("Recent Venus distance: ", Dis_rv, "AU")


#Runaway Greenhouse

s_eff_rg = 1.0385
a_rg = 1.2456e-4
b_rg = 1.4612e-8
c_rg = -7.6345e-12
d_rg = -1.7511e-15
Seff_rg=Seff(s_eff_rg,a_rg,b_rg,c_rg,d_rg)
Dis_rg=Dis(Seff_rg)
Seff_prime_rg=Seff_prime(Seff_rg,e)
Dis_prime_rg=Dis(Seff_prime_rg)
print("Runaway Greenhouse distance: ", Dis_rg, "AU")

#Moist Greenhouse

s_eff_mog = 1.0146
a_mog = 8.1884e-5
b_mog = 1.9394e-9
c_mog = -4.3618e-12
d_mog = -6.8260e-16
Seff_mog=Seff(s_eff_mog,a_mog,b_mog,c_mog,d_mog)
Dis_mog=Dis(Seff_mog)
Seff_prime_mog=Seff_prime(Seff_mog,e)
Dis_prime_mog=Dis(Seff_prime_mog)
print("Moist Greenhouse distance: ", Dis_mog, "AU")

#Maximun Greenhouse

s_eff_mag = 0.3507
a_mag = 5.9578e-5
b_mag = 1.6707e-9
c_mag = -3.0058e-12
d_mag = -5.1925e-16
Seff_mag=Seff(s_eff_mag,a_mag,b_mag,c_mag,d_mag)
Dis_mag=Dis(Seff_mag)
Seff_prime_mag=Seff_prime(Seff_mag,e)
Dis_prime_mag=Dis(Seff_prime_mag)
print("Maximum Greenhouse distance: ", Dis_mag, "AU")

#Early Mars

s_eff_em = 0.3207
a_em = 5.4471e-5
b_em = 1.5275e-9
c_em = -2.1709e-12
d_em = -3.8282e-16
Seff_em=Seff(s_eff_em,a_em,b_em,c_em,d_em)
Dis_em=Dis(Seff_em)
Seff_prime_em=Seff_prime(Seff_em,e)
Dis_prime_em=Dis(Seff_prime_em)
print("Early Mars distance: ", Dis_em, "AU")


#
# La zona de habitabilidad esta entre Moist Greenhouse y Maximum Greenhouse
#

print("Zona de Habitabilidad - Conservative: ", Dis_mog, "-", Dis_mag, "AU")
print("Zona de Habitabilidad - Optimistic: ", Dis_rv, "-", Dis_em, "AU")


##
## Vamos a plotear eccentricidad vs semimajor axis, para considerar orbitas eccentricas
##


a_end = 10.
Nout = 10000
a = np.linspace(0.0001,a_end,Nout,endpoint=False)

fig1 = plt.figure(figsize=(5,3))
plt.subplot(111)
#plt.plot(a,1.-Dis_rv/a,'b-',a,Dis_em/a-1,'b-')
#plt.plot(a,1.-Dis_mog/a,'b-',a,Dis_mag/a-1,'b-')
#plt.plot(a,1.-Dis_prime_rv/a,'r--',a,Dis_prime_em/a-1,'r--' )
#plt.plot(a,1.-Dis_prime_mog/a,'r--',a,Dis_prime_mag/a-1,'r--' )
colorx='green'
HZ_opt=[]
a_opt=[]
for i in range(0, len(a)):
    if (Dis_em/a[i]-1. > 1.-Dis_rv/a[i] and 1.-Dis_rv/a[i] > 0.):
        HZ_opt.append(1.-Dis_rv/a[i])
        a_opt.append(a[i])
    if (Dis_em/a[i]-1. < 1.-Dis_rv/a[i] and Dis_em/a[i]-1 > 0.):
        HZ_opt.append(Dis_em/a[i]-1.)
        a_opt.append(a[i])
colorx='green'
HZ_con=[]
a_con=[]
for i in range(0, len(a)):
    if (Dis_mag/a[i]-1. > 1.-Dis_mog/a[i] and 1.-Dis_mog/a[i] > 0.):
        HZ_con.append(1.-Dis_mog/a[i])
        a_con.append(a[i])
    if (Dis_mag/a[i]-1. < 1.-Dis_mog/a[i] and Dis_mag/a[i]-1 > 0.):
        HZ_con.append(Dis_mag/a[i]-1.)
        a_con.append(a[i])


plt.fill_between(a_opt,HZ_opt, 0,facecolor=colorx, alpha=0.5,edgecolor="none")
plt.fill_between(a_con,HZ_con, 0,facecolor=colorx, alpha=0.8,edgecolor="none")
plt.text(0.015,0.35,"Optimistic HZ",size=10)
plt.plot(0.01,0.36,"s",color=colorx,alpha=0.5)
plt.text(0.015,0.32,"Conservative HZ",size=10)
plt.plot(0.01,0.33,"s",color=colorx,alpha=0.8)




##calculo del planeta a traves de su periodo orbital, especial para TESS

p1 = 15# periodo orbital en dias dado por TESS
T1 = p1*24.*3600. #periodo orbital en segundos

p2 = 18  # periodo orbital en dias dado por TESS
T2 = p2*24.*3600. #periodo orbital en segundos

#p3 = 217.2  # periodo orbital en dias dado por TESS
#T3 = p3*24.*3600. #periodo orbital en segundos

G = 6.674e-11 #m3 kg-1 s-2
Mass = 0.12 #Msun
Mass_kg = Mass*2.e30 #masa del cuerpo central en kg

semieje1 = (G*Mass_kg*T1**2/4./(np.pi**2))**(1./3.)
semieje1_au = semieje1/1.496e11
print("a1 = ", semieje1_au)


semieje2 = (G*Mass_kg*T2**2/4./(np.pi**2))**(1./3.)
semieje2_au = semieje2/1.496e11
print("a2 = ", semieje2_au)

#semieje3 = (G*Mass_kg*T3**2/4./(np.pi**2))**(1./3.)
#semieje3_au = semieje3/1.496e11
#print "a3 = ", semieje3_au

r_earth=5

plt.errorbar(semieje1_au,0.042,yerr=[[0.028],[0.040]],marker='o',markersize=1.44*r_earth,color='cyan',markeredgecolor='black',ecolor='black',elinewidth=0.75,markeredgewidth=0.75)

plt.errorbar(semieje2_au,0.034,yerr=[[0.024],[0.044]],marker='o',markersize=1*r_earth,color='cyan',markeredgecolor='black',ecolor='black',elinewidth=0.75,markeredgewidth=0.75)

#plt.errorbar(semieje3_au,0.55,0.10,marker='o',markersize=6.5*r_earth,color='cyan',markeredgecolor='black',ecolor='black',elinewidth=0.75,markeredgewidth=0.75)


#plt.text(0.018,0.017,"03",size=6)
#plt.plot(0.0316,0.02,'o',markersize=1.43*r_earth,color='Orange')
#plt.text(0.0316,0.02,"02",size=6)
#plt.plot(0.0505,0.02,'o',markersize=1.34*r_earth,color='Blue')
#plt.text(0.0505,0.02,"01",size=6)



#plt.fill_between(a,Dis_em/a-1., 0, where=Dis_em/a-1. < 1.-Dis_rv/a, facecolor=colorx, alpha=0.5,edgecolor="none")

plt.ylabel("Eccentricity",size=12)
plt.xlabel("Semimajor axis (au)",size=12)
plt.ylim(0,0.5)
plt.xlim(0,0.075)
#plt.title("GJ 628")
#####
#### Formato de salida: png, jpg, png o eps
####

formato = str('.png')
#formato = str('.png')
#formato = str('.jpg')
#formato = str('.eps')
plt.savefig('sp0004-1709'+formato,
                  bbox_inches='tight', # Elimina margenes en blanco
                  dpi=200)

