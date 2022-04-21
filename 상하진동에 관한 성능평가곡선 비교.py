# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:24:10 2021

@author: hwlee
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from matplotlib import rc


x1 = np.arange(3,8,0.01)
x2 = np.arange(8,30,0.01)

y1 = np.zeros(len(x1))+0.81
y2 = 0.101*x2

x3 = np.concatenate((x1,x2))
y3 = np.concatenate((y1,y2))



x11 = np.arange(3,8,0.01)
x21 = np.arange(8,30,0.01)

y11 = np.zeros(len(x1))+1.36
y21 = 0.17*x2

x31 = np.concatenate((x11,x21))
y31 = np.concatenate((y11,y21))



x12 = np.arange(3,8,0.01)
x22 = np.arange(8,30,0.01)

y12 = np.zeros(len(x1))+ 2
y22 = 0.25*x2

x32 = np.concatenate((x12,x22))
y32 = np.concatenate((y12,y22))




x13 = np.arange(3,8,0.01)
x23 = np.arange(8,30,0.01)

y13 = np.zeros(len(x1))+2.9
y23 = 0.363*x2

x33 = np.concatenate((x13,x23))
y33 = np.concatenate((y13,y23))



x14 = np.arange(3,8,0.01)
x24 = np.arange(8,30,0.01)

y14 = np.zeros(len(x1))+4.92
y24 = 0.615*x2

x34 = np.concatenate((x14,x24))
y34 = np.concatenate((y14,y24))


u0 = np.arange(1,4,3)
u1 = np.arange(4,8,0.01)
u2 = np.arange(8,80,0.01)

v0 = (0.5-1)/(4-1)*u0+(7/6)
v1 = np.zeros(len(u1))+0.5
v2 = (5-0.5)/(80-8)*u2

u3 = np.concatenate((u0,u1,u2))
v3 = np.concatenate((v0,v1,v2))


u01 = np.arange(1,4,3)
u11 = np.arange(4,8,0.01)
u21 = np.arange(8,80,0.01)

v01 = (1-2)/(4-1)*u0+7/3
v11 = np.zeros(len(u1))+1
v21 = (10-1)/(80-8)*u2

u31 = np.concatenate((u01,u11,u21))
v31 = np.concatenate((v01,v11,v21))


u02 = np.arange(1,4,3)
u12 = np.arange(4,8,0.01)
u22 = np.arange(8,80,0.01)

v02 = (2-4)/(4-1)*u0+14/3
v12 = np.zeros(len(u1))+2
v22 = (20-2)/(80-8)*u2

u32 = np.concatenate((u02,u12,u22))
v32 = np.concatenate((v02,v12,v22))



u03 = np.arange(1,4,3)
u13 = np.arange(4,8,0.01)
u23 = np.arange(8,80,0.01)

v03 = (4-8)/(4-1)*u0+28/3
v13 = np.zeros(len(u1))+4
v23 = (40-4)/(80-8)*u2

u33 = np.concatenate((u03,u13,u23))
v33 = np.concatenate((v03,v13,v23))





matplotlib.rcParams['axes.unicode_minus'] = False
font_name = fm.FontProperties(fname='c:\\windows\\fonts\\malgun.ttf').get_name()
rc('font', family=font_name)

plt.figure(1)


v90 = plt.loglog(x34,y34)
v70 = plt.loglog(x33,y33)
v50 = plt.loglog(x32,y32)
v30 = plt.loglog(x31,y31)
v10 = plt.loglog(x3,y3)

iso_t = ax.loglog(u33,v33,'--')
iso_r = ax.loglog(u32,v32,'--')
iso_o = ax.loglog(u31,v31,'--')
iso_w = ax.loglog(u3,v3,'--')


#slab_180 = ax.plot(16.18, 2.74,'cs')
#ax.text(16.18, 2.74, '   16.18 Hz, 2.74cm/$s^2$', fontweight='bold')

#slab_210 = ax.plot(18.77, 2.27,'bo')
#ax.text(18.77, 2.267, '   18.77 Hz, 2.27cm/$s^2$', fontweight='bold')

#slab_240 = ax.plot(21.21, 1.88,'r^')
#ax.text(21.21, 1.88, '   21.21 Hz, 1.88cm/$s^2$', fontweight='bold')




plt.xlabel('주파수(Hz)')
plt.ylabel('가속도(cm/$s^2$)')
plt.title('AIJ와 ISO의 건물 사용성 평가 기준')
plt.legend(["V-90 (AIJ)","V-70 (AIJ)","V-50 (AIJ)","V-30 (AIJ)","V-10 (AIJ)","Workshop (ISO)","Office (ISO)","Residential (ISO)","Operating Theatre (ISO)", "건전부", "균열부"], loc='upper right', bbox_to_anchor=(1.58,1))


ax.grid(True, which="both", ls="-")

plt.xlim([1,50])
plt.ylim([0.5,50])


ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.0f'))
ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))

ax.set_xticks([1,2,3,4,5,6,7,8,9,10,20,30,40,50])
ax.set_yticks([0.5,0.8,1,2,3,4,5,10,20,50])

fig.set_size_inches(7,5)
plt.savefig('슬래브 두께별 사용성 평가.png', dpi=300, bbox_inches = 'tight')

plt.show()