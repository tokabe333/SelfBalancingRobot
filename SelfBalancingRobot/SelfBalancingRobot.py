# -*- Coding: utf-8 -*-
from control.matlab import place as pl
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import * 
import random

# |------------------------------------------------------|
# |関数で書くよりベタ書きのほうが楽，今後使い回すこともないので|
# |------------------------------------------------------|

# -------------------- 行列作る --------------------
g = 9.81				#// 重力加速度
m = 0.03				#// 車輪質量
R = 0.04				#// 車輪半径
Jw = m * R ** (2 / 2)		#// 車輪慣性モーメント
M = 0.6					#// 車体質量
H = 0.144				#// 車体高さ
L = H / 2				#// 車輪中心から車体重心までの距離
Jp = M * L ** (2 / 3)   #// 車体慣性モーメント
Jm = 1 * 10 ** (-5)		#// DCモータ慣性モーメント
Rm = 6.69				#// DCモータ抵抗
Kb = 0.468				#// DCモータ逆起電力定数
Kt = 0.317				#// DCモータトルク定数
fm = 0.0022				#// 車体とDCモータ間の摩擦係数
fw = 0.001				#// 車輪と路面間の摩擦係数


#///// 線形モデルの導出 /////
a = 2 * Kt / Rm
b = 2 * (Kt * Kb / Rm + fm)

#// 課題1: 下記のパラメータを数字とここまで定義したパラメータで表わせ.
p1 = M * R ** 2 + 3 * m * R ** 2 + 2 * Jm
p2 = M * R * L - 2 * Jm
p3 = M * R * L - 2 * Jm
p4 = 4 / 3 * M * L ** 2 + 2 * Jm
p5 = M * g * L

#print("p1",p1,"\r\np2",p2,"\r\np3",p3,"\r\np4",p4,"\r\np5",p5)

#// 課題2: alpha, beta, p1, p2, p3, p4, p5, fwを用いて以下の行列を定義せよ.
E = np.array([[p1, p2], [p3, p4]])
F = np.array([[b - 2 * fw, -b], [-b, b]])
G = np.array([[0,0], [0,p5]])
H = np.array([[a], [-a]])

#Eの逆行列
Einv = np.linalg.inv(E)


#定数
theta = 0
psi = 5 / 180 * np.pi
thetad = 0
psid = 0
tFin = 5		#終了時刻
h = 0.0001		#刻み幅
N = int(tFin / h)
x = [[theta], [psi], [thetad], [psid]]

#A行列を作る(Scilabは楽)
EG = -np.dot(Einv, G)
FG = np.dot(Einv, F)
A = np.array([[0,0,1,0],[0,0,0,1],[EG[0][0], EG[0][1], FG[0][0], FG[0][1]],[EG[1][0], EG[1][1], FG[1][0], FG[1][1]]])

#B行列を作る(Scilabは)
EH = np.dot(Einv, H)
B = np.array([[0],[0],[EH[0][0]],[EH[1][0]]])

#ランダムに配置を探索
while True:
	n1 = -random.uniform(0,50)
	n2 = -random.uniform(0,50)
	n3 = -random.uniform(0,50)
	j1 = 1j * random.uniform(3,5)
	j2 = 1j * random.uniform(3,5)
	Fx = -np.array(pl(A, B, np.array([n1 + j1,n1 - j1,n2, n3])))
	l = np.linalg.eig(A + np.dot(B, Fx))[0]
	print("固有値 l : " + str(l))
	print("n : ",n1 + j1,n1 - j1,n2 ,n3)
	u = np.dot(Fx, x)
	print("u :", u)
	print("\r\n")
	if np.all(l < 0) and abs(u) < 5:
		break
#End_While

#求めたい関数 x'
def XD(_x):
	#return np.dot(A + np.dot(B, Fx), _x)
	#return np.dot(A, _x) + np.dot(B, np.dot(Fx, _x))
	return np.dot(A, _x) + B * u
#End_Def

#ルンゲクッタ法で解く
def RungeKutta():
	global u
	num = [0] * N
	num[0] = x

	for i in range(0, N - 1):
		cn = num[i]
		u = np.dot(Fx, cn)
		k1 = h * XD(cn)
		k2 = h * XD(cn + k1 / 2)
		k3 = h * XD(cn + k2 / 2)
		k4 = h * XD(cn + k3)
		num[i + 1] = cn + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
	#End_For
	return num
#End_Def

#θとΨについて
result = RungeKutta()
thetaResult = [0] * len(result)
psiResult = [0] * len(result)
for i in range(len(result)):
	thetaResult[i] = result[i][0]
	psiResult[i] = result[i][1]
#End_For

#Plot
plt.figure(figsize=(12, 12), dpi = 130)
plt.rcParams["font.size"] = 25
plt.rcParams['font.family'] = 'Times new Roman'
plt.plot(np.arange(0, tFin, h)[0:len(thetaResult)], thetaResult, label="θ")
plt.plot(np.arange(0, tFin, h)[0:len(psiResult)], psiResult, label="Ψ")
plt.xlabel("Time [second]", fontsize=32)
plt.ylabel("θ Ψ [rad]", fontsize=32)
#plt.legend(bbox_to_anchor=(1.03, 1.13), loc="upper right",borderaxespad=1,
#fontsize=16)
plt.legend(fontsize=32)

#save
os.makedirs("./Result", exist_ok = True)
plt.savefig("./Result/ResultImage.png")
plt.show()
with open("./Result/Variables.txt", 'w') as f:
	f.write("n1 : " + str(n1) + "\r\n")
	f.write("n2 : " + str(n2) + "\r\n")
	f.write("n3 : " + str(n3) + "\r\n")
	f.write("j1 : " + str(j1) + "\r\n")
	f.write("j2 : " + str(j2) + "\r\n")
