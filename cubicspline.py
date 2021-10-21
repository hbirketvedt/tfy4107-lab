# TFY41xx Fysikk vaaren 2021.
#
# Programmet tar utgangspunkt i hoeyden til de 8 festepunktene.
# Deretter beregnes baneformen y(x) ved hjelp av 7 tredjegradspolynomer, 
# et for hvert intervall mellom to festepunkter, slik at baade banen y, 
# dens stigningstall y' = dy/dx og dens andrederiverte
# y'' = d2y/dx2 er kontinuerlige i de 6 indre festepunktene.
# I tillegg velges null krumning (andrederivert) 
# i banens to ytterste festepunkter (med bc_type='natural' nedenfor).
# Dette gir i alt 28 ligninger som fastlegger de 28 koeffisientene
# i de i alt 7 tredjegradspolynomene.

# De ulike banene er satt opp med tanke paa at kula skal 
# (1) fullfoere hele banen selv om den taper noe mekanisk energi underveis;
# (2) rulle rent, uten aa gli ("slure").

# Vi importerer noedvendige biblioteker:
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import scipy as sc
import pandas as pd

# Horisontal avstand mellom festepunktene er 0.200 m
h = 0.200
xfast=np.asarray([0,h,2*h,3*h,4*h,5*h,6*h,7*h])

#konstanter
g = 9.81
y0 = 0.254
c=2/5
# Vi begrenser starthøyden (og samtidig den maksimale høyden) til
# å ligge mellom 250 og 300 mm
ymax = 300
# yfast: tabell med 8 heltall mellom 50 og 300 (mm); representerer
# høyden i de 8 festepunktene
yfast=np.asarray(np.random.randint(50, ymax, size=8))
#konverter fra m til mm
yfast =yfast/1000
# inttan: tabell med 7 verdier for (yfast[n+1]-yfast[n])/h (n=0..7); dvs
# banens stigningstall beregnet med utgangspunkt i de 8 festepunktene.
inttan = np.diff(yfast)/h
attempts=1
# while-løkken sjekker om en eller flere av de 3 betingelsene ovenfor
# ikke er tilfredsstilt; i så fall velges nye festepunkter inntil
# de 3 betingelsene er oppfylt
# while (yfast[0] < yfast[1]*1.04 or
#        yfast[0] < yfast[2]*1.08 or
#        yfast[0] < yfast[3]*1.12 or
#        yfast[0] < yfast[4]*1.16 or
#        yfast[0] < yfast[5]*1.20 or
#        yfast[0] < yfast[6]*1.24 or
#        yfast[0] < yfast[7]*1.28 or
#        yfast[0] < 0.250 or
#        np.max(np.abs(inttan)) > 0.4 or
#        inttan[0] > -0.2):
#           yfast=np.asarray(np.random.randint(0, ymax, size=8))
#
#           #konverter fra m til mm
#           yfast =yfast/1000
#
#           inttan = np.diff(yfast)/h
#           attempts=attempts+1

# Omregning fra mm til m:
# xfast = xfast/1000
# yfast = yfast/1000

trial_1 = np.asarray(pd.read_csv("data\\trial_1.csv"))
trial_2 = np.asarray(pd.read_csv("data\\trial_2.csv"))
trial_3 = np.asarray(pd.read_csv("data\\trial_3.csv"))
trial_4 = np.asarray(pd.read_csv("data\\trial_4.csv"))
trial_5 = np.asarray(pd.read_csv("data\\trial_5.csv"))
trial_6 = np.asarray(pd.read_csv("data\\trial_6.csv"))
trial_7 = np.asarray(pd.read_csv("data\\trial_7.csv"))
trial_8 = np.asarray(pd.read_csv("data\\trial_8.csv"))
trial_9 = np.asarray(pd.read_csv("data\\trial_9.csv"))
trial_10 = np.asarray(pd.read_csv("data\\trial_10.csv"))
trial_11 = np.asarray(pd.read_csv("data\\trial_11.csv"))


yfast = [0.254, 0.202, 0.226, 0.154, 0.093, 0.057, 0.08,  0.002]
# Når programmet her har avsluttet while-løkka, betyr det at
# tallverdiene i tabellen yfast vil resultere i en tilfredsstillende bane. 

#Programmet beregner deretter de 7 tredjegradspolynomene, et
#for hvert intervall mellom to nabofestepunkter.


#Med scipy.interpolate-funksjonen CubicSpline:
cs = CubicSpline(xfast, yfast, bc_type='natural')

xmin = 0.000
xmax = 1.401
dx = 0.001

x = np.arange(xmin, xmax, dx)   

#funksjonen arange returnerer verdier paa det "halvaapne" intervallet
#[xmin,xmax), dvs slik at xmin er med mens xmax ikke er med. Her blir
#dermed x[0]=xmin=0.000, x[1]=xmin+1*dx=0.001, ..., x[1400]=xmax-dx=1.400, 
#dvs x blir en tabell med 1401 elementer
Nx = len(x)
y = cs(x)       #y=tabell med 1401 verdier for y(x)
dy = cs(x,1)    #dy=tabell med 1401 verdier for y'(x)
d2y = cs(x,2)   #d2y=tabell med 1401 verdier for y''(x)



def v(y):
    return np.sqrt((10*g*(y0-y))/7)

# def v(trial):
#     print(trial)
#     y = []
#     for list in trial:
#         y = np.asarray(y)
#     return np.sqrt((10 * g * (y0 - y)) / 7)



def helningsvinkel(dy):
    return np.arctan(dy)

# def v(t):
#     list = [0]
#     for index in range(1, len(x)):
#         list.append((1/2)*v(index-1) + v(index))
#     return np.array(list)


def vx():
    return v(y) * np.cos(helningsvinkel(dy))

plt.plot(x, vx())
plt.show()


def delta_t_hjelpefunksjon(vx0, vx1):
    return 2*dx/(vx0 + vx1)

def delta_t():
    list = []
    vsvs = vx()
    for index in range(1, len(x)):
        list.append(delta_t_hjelpefunksjon(vsvs[index-1], vsvs[index]))
    return np.array(list)

def summed_t():
    list = []
    deltas = delta_t()
    for index in range(len(deltas)):
        sum = 0
        for j in range(index):
            sum += deltas[j]
        list.append(sum)
    return np.array(list)

def remove_last_x():
    sub_arr = x[:-1].copy()
    return sub_arr


# def x_pos():
#     delta_t_list = delta_t()
#     x = 0
#     for delta in delta_t_list:
#         t -= delta
#         x += dx
#         if t<=0:
#             break
#     return x




#Eksempel: Plotter banens form y(x)
# baneform = plt.figure('y(x)',figsize=(12,6))
# plt.plot(x,y,xfast,yfast,'*')
# plt.title('Banens form')
# plt.xlabel('$x$ (m)',fontsize=20)
# plt.ylabel('$y(x)$ (m)',fontsize=20)
# plt.ylim(0.0,0.40)
# plt.grid()
# plt.show()
#Figurer kan lagres i det formatet du foretrekker:
#baneform.savefig("baneform.pdf", bbox_inches='tight')
#baneform.savefig("baneform.png", bbox_inches='tight')
#baneform.savefig("baneform.eps", bbox_inches='tight')


# print('Antall forsøk',attempts)
# print('Festepunkthøyder (m)',yfast)
# print('Banens høyeste punkt (m)',np.max(y))
#
# print('NB: SKRIV NED festepunkthøydene når du/dere er fornøyd med banen.')
# print('Eller kjør programmet på nytt inntil en attraktiv baneform vises.')

# def v(y):
#     return np.sqrt((10*g*(y0-y))/7)

# plt.plot(x, v(y))
# # plt.xticks(list(range(len(x))), x)
# # plt.xlabel("x")
# plt.xlabel('$x$ (m)',fontsize=20)
# plt.ylabel('$v(x)$ (m)',fontsize=20)
# plt.grid()
#
# plt.show()


#
# plt.plot(x, helningsvinkel(dy))
# plt.xlabel('$x$ (radianer)',fontsize=12)
# plt.ylabel('$helningsvinkel$ (m)',fontsize=20)
# plt.grid()
#
# plt.show()
#
#
# plt.plot(x, vx())
# plt.xlabel('$x$ (radianer)',fontsize=12)
# plt.ylabel('$vx(x)$ (m)',fontsize=20)
# plt.grid()
#
# plt.show()

# plt.plot(x, t())
# plt.xlabel('$x$ (radianer)',fontsize=12)
# plt.ylabel('$t(x)$ (m)',fontsize=20)
# plt.grid()
# 
# plt.show()

# plt.plot(summed_t(), remove_last_x())
# plt.xlabel('$t$ (s)',fontsize=12)
# plt.ylabel('$x$ (m)',fontsize=20)
# plt.grid()
#
# plt.show()

