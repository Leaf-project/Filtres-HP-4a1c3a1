#pragma once

#include "etc.cuh"

/*
Deux options d'optimisation s'offrent a moi
	1) depart -> depart + T                             Soit on fait une suite continue de t
	2) [randint(depart, fin) for _ in range(T)]         Soit des t aleatoirement choisis parmis DEPART:FIN
*/

//	Si MODE==t_CONTINUE alors    t = depart+i
//	Si MODE==t_PSEUDO_ALEA alors t = pseudo(grain, i)

extern uint MODE_t_MODE;
extern uint grain_t_MODE;

#define t_CONTINUE    0
#define t_PSEUDO_ALEA 1

#define t_MODE(MODE, _depart, _fin, i, graine) ( \
	MODE == t_CONTINUE ? (_depart+i) : \
		( _depart+PSEUDO_ALEA((grain+i)) % (_fin-_depart) ))

#define t_MODE_GENERALE(_depart, _DEPART, _FIN, i) ( \
	MODE_t_MODE == t_CONTINUE ? t_MODE(t_CONTINUE, _depart, NULL, i, NULL) : \
	t_MODE(t_PSEUDO_ALEA, _DEPART, _FIN, i, grain_t_MODE))

#define cuda_t_MODE_GENERALE(MODE, GRAINE, _depart, _DEPART, _FIN, i) ( \
	MODE == t_CONTINUE ? t_MODE(t_CONTINUE, _depart, NULL, i, NULL) : \
	t_MODE(t_PSEUDO_ALEA, _DEPART, _FIN, i, GRAINE))

/*	Ne seront munis de pseudo-alea seulement le Dar et l'Exi. Car le reste se refere au GRAND_T et n'est pas liée au temps.

Seule les normalisee_diff[] auront le _t_MODE

et donc le SCORE(x) et dSCORE(c)

*/