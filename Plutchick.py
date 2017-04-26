#Define la arquitectura del sofware
#Construcción de función cuadrica en álgebra lineal
#La fórmula canónica A*x^2+B*y^2+C*z^2+D*x*y+E*x*z+F*y*z+G*x+H*y+I*z+J=0
#La resolución del sistema dibuja una cuadrica

import os
import numpy
import sys
#Arrays inputs
dict['X^2,Y^2,Z^2'] = {'A':{}, 'B':{}, 'C':{}}
dict['XY,XZ,YZ'] = {'D':{}, 'E':{}, 'F':{}}
dict['X,Y,Z'] = {'G':{}, 'H':{}, 'I':{}}
str('JA') = ('J')


#Forma de la cuádrica (1,x,y,z)*A*np.transpose(1,x,y,z)
vector_canónico = np.matrix([1, X, Y, Z])
vector_canónico2 = np.transpose(vector_canónico)
A = dict{'A': {}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'H':{}, 'I':{}, 'J':{}}
matriz0 = np.matrix([[J, G/2, H/2, I/2][G/2, A, D/2, E/2][H/2, D/2, B, G/2][I/2, E/2, G/2, C])
                    
def Cuadrica():
    return vector_canónico * matriz0 * vector_canónico2
                    
#Invariantes de la cuádrica
#Det(lambdaI3 - A00) = lambda***3 - I*lambda**2 + lambda*J - K

#I=A+B+C                    
def I():
    return sum 'A'+'B'+'C'
def J():
    return sum ('A*B'- 'D^2/4') + ( 'A*C' - 'E^2/4' ) + ('B*C' - 'G^2/4')                
                    
def K():
    return np.lianalg.det(A)
                    if ro = 3:
                        if det(A) > 0
                        return 'draw_elipsoide()'
                        if det(A) < 0
                        return 'draw_elipsoide()'
                        if det(A) = 0
                        return 'cono_imaginario()'
                    if ro = 1:
                        if det(A) > 0
                        return 'draw_hiperboloide_hiperbolico_1_hoja()'
                        if det(A) < 0
                        return 'draw_hiperboloide_eliptico_2_hojas()'
                        if det(A) = 0
                        return 'draw_cono real()'
                    else :
                        det(A)

def Centro():
    if (1,x,y,z)*A*np.transpose(1,x,y,z)               
