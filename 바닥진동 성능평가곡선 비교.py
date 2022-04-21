# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:24:10 2021

@author: hwlee
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter


from matplotlib import rc

#%% 사용자 경로 설정

# Input(최대가속도, rms, 진동수 입력)
theoretical_acc_max_s =  0.1538
theoretical_acc_rms_s = 0.0411 # 이론(단순지지)
theoretical_freq_s = 8.74

theoretical_acc_max_f = 0.6310
theoretical_acc_rms_f = 0.1231 # 이론(고정지지)
theoretical_freq_f = 16.41

measured_acc_max = 0.6844
measured_acc_rms = 0.0982 # 계측
measured_freq = 11.36

# fea_acc_rms = 아래에서 구함
fea_freq = 14.057 # 마이다스


# Input 경로 설정 (마이다스 가속도응답 불러와서 rms 구하기위함)
acc_data_dir = r'D:\이형우\바닥판 진동 사용성 검토\MIDAS GEN 모델링'
acc_data_file = 'acceleration.xlsx'
acc_data_sheet = 'acc_data'

#%% fea 가속도 정보 load
acc_data = pd.read_excel(acc_data_dir+'\\'+acc_data_file,
                     sheet_name=acc_data_sheet, skiprows=9, usecols=[1,4], header=0)
acc_data.columns = ['time(s)', 'acceleration(m/s^2)']

new_acc_data_time = []
new_acc_data_displ = []

for i, j in zip(acc_data['time(s)'], acc_data['acceleration(m/s^2)']):
    if np.isnan(i):
        break
    else:
        new_acc_data_time.append(i)
        new_acc_data_displ.append(j)

new_acc_data = pd.DataFrame()
new_acc_data['time(s)'], new_acc_data['acceleration(m/s^2)'] = new_acc_data_time, new_acc_data_displ
new_acc_data.reset_index(inplace=True, drop=True)

#%% r.m.s function
def rms(x_list):
    x_array = np.array(x_list)
    return np.sqrt(x_array.dot(x_array)/x_array.size)

#%% 최대가속도, rms값 (유한요소해석)
acc_max = new_acc_data['acceleration(m/s^2)'].max()
acc_rms = rms(new_acc_data['acceleration(m/s^2)']) # m/s^2
acc_rms = round(acc_rms, 3)

# cm로 변환
acc_rms_cm = acc_rms*100 # cm/s^2

#%% 한국어 나오게하는 코드(이해는 못함)
matplotlib.rcParams['axes.unicode_minus'] = False
font_name = fm.FontProperties(fname='c:\\windows\\fonts\\malgun.ttf').get_name()
rc('font', family=font_name)

#%% ISO Graph

# Trade-off Factor
N = 15
T = 0.5
d = 0

F1 = 1.7 * N**(-0.5) * T **(-d) 
if F <= 2: F = 2
F=2

# AISC에서 제시하는 factor
F=30

# base curve
base_x = np.array([1, 4, 8, 80]) # Hz
base_y = np.array([0.01, 0.005, 0.005, 0.05]) # m/s^2

# Weighted Base Curve
weighted_base_y = F * base_y

# Graph
plt.figure(1, dpi=150)
plt.xlim([1,100])
plt.ylim([0.03, 2.20])
# plt.ylim([0.002,0.2])

plt.loglog(base_x, weighted_base_y, label='Residential(ISO)')

# 이론값(단순지지)
plt.plot(theoretical_freq_s, theoretical_acc_rms_s,'cs', label='이론값(4변 단순)')
plt.text(theoretical_freq_s, theoretical_acc_rms_s, '   f='+str(theoretical_freq_s)+' Hz\n   a='+str(theoretical_acc_rms_s)+' m/$s^2$',\
         fontsize='small', horizontalalignment='left', verticalalignment='center')
    
# 이론값(고정지지)
plt.plot(theoretical_freq_f, theoretical_acc_rms_f,'ms', label='이론값(4변 고정)')
plt.text(theoretical_freq_f, theoretical_acc_rms_f, '   f='+str(theoretical_freq_f)+' Hz\n   a='+str(theoretical_acc_rms_f)+' m/$s^2$',\
         fontsize='small', horizontalalignment='left')

# 마이다스 해석값
plt.plot(fea_freq, acc_rms,'bo', label='해석값')
plt.text(fea_freq, acc_rms, 'f='+str(fea_freq)+' Hz\n   a='+str(acc_rms)+' m/$s^2$',\
         fontsize='small', horizontalalignment='right', verticalalignment='bottom')

# 계측값
plt.plot(measured_freq, measured_acc_rms,'r^', label='계측값')
plt.text(measured_freq, measured_acc_rms, '   f='+str(measured_freq)+' Hz\n   a='+str(measured_acc_rms)+' m/$s^2$',\
         fontsize='small', horizontalalignment='left', verticalalignment='top')

plt.xlabel('주파수, Hz')
plt.ylabel('가속도(r.m.s.), m/$s^2$')
plt.title('ISO 사용성 평가(마감 전)')
# plt.legend(["V-90 (AIJ)","V-70 (AIJ)","V-50 (AIJ)","V-30 (AIJ)","V-10 (AIJ)","Workshop (ISO)","Office (ISO)","Residential (ISO)","Operating Theatre (ISO 2631-2)", "균열부", "건전부", "슬래브 두께 240mm"], loc='upper right', bbox_to_anchor=(1.58,1))
plt.legend(loc='upper left', fontsize='small')
plt.grid(True, which="both", ls="-")

plt.xticks([1, 4, 8, 80],['1','4','8','80'])
plt.yticks([0.03, 0.05, 0.1, 0.3, 2.0],['0.03','0.05','0.10','0.30','2.00'])

#%% AIJ Graph
AIJ_weight_factor = 15

# base curve
base_x_AIJ = np.array([3, 8, 30]) # Hz
V10_y = np.array([0.81, 0.81, 0.81 + 0.101*(30-8)]) / 100 # m/s^2

# Other Curves
V30_y = np.array([1.36, 1.36, 1.36 + 0.170*(30-8)]) / 100
V50_y = np.array([2.00, 2.00, 2.00 + 0.250*(30-8)]) / 100
V70_y = np.array([2.90, 2.90, 2.90 + 0.363*(30-8)]) / 100
V90_y = np.array([4.92, 4.92, 4.92 + 0.615*(30-8)]) / 100

# Weighted Curve
weighted_y_AIJ = AIJ_weight_factor * V50_y # 사용자 입력!!!

# Graph
plt.figure(2, dpi=150)
plt.xlim([1,50])
plt.ylim([0.1, 1.4])
# plt.ylim([0.002,0.2])

plt.loglog(base_x_AIJ, weighted_y_AIJ, label='V-10 (AIJ)')

# 이론값(단순지지)
plt.plot(theoretical_freq_s, theoretical_acc_max_s,'cs', label='이론값(4변 단순)')
plt.text(theoretical_freq_s, theoretical_acc_max_s, '   f='+str(theoretical_freq_s)+' Hz\n   a='+str(theoretical_acc_max_s)+' m/$s^2$',\
         fontsize='small', horizontalalignment='left', verticalalignment='center')
    
# 이론값(고정지지)
plt.plot(theoretical_freq_f, theoretical_acc_max_f,'ms', label='이론값(4변 고정)')
plt.text(theoretical_freq_f, theoretical_acc_max_f, '   f='+str(theoretical_freq_f)+' Hz\n   a='+str(theoretical_acc_max_f)+' m/$s^2$',\
         fontsize='small', horizontalalignment='left')

# 마이다스 해석값
plt.plot(fea_freq, acc_max,'bo', label='해석값')
plt.text(fea_freq, acc_max, '   f='+str(fea_freq)+' Hz\n   a='+str(acc_max)+' m/$s^2$',\
         fontsize='small', horizontalalignment='left', verticalalignment='top')

# 계측값
plt.plot(measured_freq, measured_acc_max,'r^', label='계측값')
plt.text(measured_freq, measured_acc_max, 'f='+str(measured_freq)+' Hz    \n   a='+str(measured_acc_max)+' m/$s^2$',\
         fontsize='small', horizontalalignment='right', verticalalignment='top')

plt.xlabel('주파수, Hz')
plt.ylabel('최대가속도, m/$s^2$')
plt.title('AIJ 사용성 평가(마감 전)')
# plt.legend(["V-90 (AIJ)","V-70 (AIJ)","V-50 (AIJ)","V-30 (AIJ)","V-10 (AIJ)","Workshop (ISO)","Office (ISO)","Residential (ISO)","Operating Theatre (ISO 2631-2)", "균열부", "건전부", "슬래브 두께 240mm"], loc='upper right', bbox_to_anchor=(1.58,1))
plt.legend(loc='upper left', fontsize='small')
plt.grid(True, which="both", ls="-")

plt.xticks([1, 3, 8, 30],['1','3','8','30'])
plt.yticks([0.1, 0.3, 0.5, 1.0, 1.2],['0.1', '0.3', '0.5', '1.0', '1.2'])

plt.show()

#%% ISO Graph(all)

# Graph
plt.figure(3, dpi=150)
plt.xlim([1,100])
plt.ylim([0.003, 0.7])
# plt.ylim([0.002,0.2])

plt.loglog(base_x, base_y, label='Operating Theatre(ISO)')
plt.loglog(base_x, base_y*2, label='Residential(ISO)')
plt.loglog(base_x, base_y*4, label='Office(ISO)')
plt.loglog(base_x, base_y*8, label='Workshop(ISO)')

plt.xlabel('주파수, Hz')
plt.ylabel('가속도(r.m.s.), m/$s^2$')
plt.title('ISO 사용성 평가기준')
# plt.legend(["V-90 (AIJ)","V-70 (AIJ)","V-50 (AIJ)","V-30 (AIJ)","V-10 (AIJ)","Workshop (ISO)","Office (ISO)","Residential (ISO)","Operating Theatre (ISO 2631-2)", "균열부", "건전부", "슬래브 두께 240mm"], loc='upper right', bbox_to_anchor=(1.58,1))
plt.legend(loc='upper left', fontsize='small')
plt.grid(True, which="both", ls="-")

plt.xticks([1, 4, 8, 80],['1','4','8','80'])
plt.yticks([0.005, 0.01, 0.02, 0.04, 0.1, 0.5],['0.005','0.01','0.02','0.04','0.1','0.5'])

#%% AIJ Graph(all)

# base curve
base_x_AIJ = np.array([3, 8, 30]) # Hz
V10_y = np.array([0.81, 0.81, 0.81 + 0.101*(30-8)]) / 100 # m/s^2

# Other Curves
V30_y = np.array([1.36, 1.36, 1.36 + 0.170*(30-8)]) / 100
V50_y = np.array([2.00, 2.00, 2.00 + 0.250*(30-8)]) / 100
V70_y = np.array([2.90, 2.90, 2.90 + 0.363*(30-8)]) / 100
V90_y = np.array([4.92, 4.92, 4.92 + 0.615*(30-8)]) / 100


# Graph
plt.figure(4, dpi=150)
plt.xlim([1,50])
plt.ylim([0.006, 0.4])
# plt.ylim([0.002,0.2])

plt.loglog(base_x_AIJ, V10_y, label='V-10(AIJ)')
plt.loglog(base_x_AIJ, V30_y, label='V-30(AIJ)')
plt.loglog(base_x_AIJ, V50_y, label='V-50(AIJ)')
plt.loglog(base_x_AIJ, V70_y, label='V-70(AIJ)')
plt.loglog(base_x_AIJ, V90_y, label='V-90(AIJ)')

plt.xlabel('주파수, Hz')
plt.ylabel('최대가속도, m/$s^2$')
plt.title('AIJ 사용성 평가기준')
# plt.legend(["V-90 (AIJ)","V-70 (AIJ)","V-50 (AIJ)","V-30 (AIJ)","V-10 (AIJ)","Workshop (ISO)","Office (ISO)","Residential (ISO)","Operating Theatre (ISO 2631-2)", "균열부", "건전부", "슬래브 두께 240mm"], loc='upper right', bbox_to_anchor=(1.58,1))
plt.legend(loc='upper left', fontsize='small')
plt.grid(True, which="both", ls="-")

plt.xticks([1, 3, 8, 30],['1','3','8','30'])
plt.yticks([0.006, 0.008, 0.010, 0.020, 0.030, 0.050, 0.100, 0.200],['0.006','0.008','0.010','0.020', '0.030', '0.050', '0.100','0.200'])

plt.show()

#%% AIJ Graph (all)

# base curve
base_x_AIJ = np.array([3, 8, 30]) # Hz
V10_y = np.array([0.81, 0.81, 0.81 + 0.101*(30-8)]) / 100 # m/s^2

# Other Curves
V30_y = np.array([1.36, 1.36, 1.36 + 0.170*(30-8)]) / 100
V50_y = np.array([2.00, 2.00, 2.00 + 0.250*(30-8)]) / 100
V70_y = np.array([2.90, 2.90, 2.90 + 0.363*(30-8)]) / 100
V90_y = np.array([4.92, 4.92, 4.92 + 0.615*(30-8)]) / 100


# Graph
fig, ax = plt.subplots()
ax.axis([1, 50, 0.5/100, 50/100])

ax.loglog(base_x_AIJ, V10_y, label='V-10(AIJ)')
ax.loglog(base_x_AIJ, V30_y, label='V-30(AIJ)')
ax.loglog(base_x_AIJ, V50_y, label='V-50(AIJ)')
ax.loglog(base_x_AIJ, V70_y, label='V-70(AIJ)')
ax.loglog(base_x_AIJ, V90_y, label='V-90(AIJ)')


for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())

ax.ticklabel_format(axis='both', style='plain')



# plt.figure(2, dpi=150)
# plt.xlim([1,50])
# # plt.ylim([0.008 * AIJ_weight_factor, 0.3 * AIJ_weight_factor])
# plt.ylim([0.5/100,50/100])



plt.xlabel('주파수, Hz')
plt.ylabel('최대가속도, m/$s^2$')
plt.title('AIJ 사용성 평가(마감 전)')
# plt.legend(["V-90 (AIJ)","V-70 (AIJ)","V-50 (AIJ)","V-30 (AIJ)","V-10 (AIJ)","Workshop (ISO)","Office (ISO)","Residential (ISO)","Operating Theatre (ISO 2631-2)", "균열부", "건전부", "슬래브 두께 240mm"], loc='upper right', bbox_to_anchor=(1.58,1))
plt.legend(loc='upper left', fontsize='small')
plt.grid(True, which="both", ls="-")




# plt.xticks([3, 8, 30],['3','8','30'])
# plt.yticks([0.0081*2, 0.0081*2, 0.03032*2],['0.0162','0.0162','0.0606'])

plt.show()



#%% 나중에 바꾸면 될듯(지금 노필요)
plt.ticklabel_format(style='plain')
plt.ticklabel_format(style='plain', axis='both', useOffset=False)
# plt.ylim([0.001,0.2])



plt.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.0f'))
plt.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))



fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig('슬래브 두께별 사용성 평가.png', dpi=300, bbox_inches = 'tight')



#%% 필요없음
x1 = np.arange(3,8,0.01)
x2 = np.arange(8,30,0.01)

y1 = np.zeros(len(x1))+0.81*2
y2 = 0.101*x2*2

x3 = np.concatenate((x1,x2))
y3 = np.concatenate((y1,y2))



x11 = np.arange(3,8,0.01)
x21 = np.arange(8,30,0.01)

y11 = np.zeros(len(x1))+1.36*2
y21 = 0.17*x2*2

x31 = np.concatenate((x11,x21))
y31 = np.concatenate((y11,y21))



x12 = np.arange(3,8,0.01)
x22 = np.arange(8,30,0.01)

y12 = np.zeros(len(x1))+ 2*2
y22 = 0.25*x2*2

x32 = np.concatenate((x12,x22))
y32 = np.concatenate((y12,y22))




x13 = np.arange(3,8,0.01)
x23 = np.arange(8,30,0.01)

y13 = np.zeros(len(x1))+2.9*2
y23 = 0.363*x2*2

x33 = np.concatenate((x13,x23))
y33 = np.concatenate((y13,y23))



x14 = np.arange(3,8,0.01)
x24 = np.arange(8,30,0.01)

y14 = np.zeros(len(x1))+4.92*2
y24 = 0.615*x2*2

x34 = np.concatenate((x14,x24))
y34 = np.concatenate((y14,y24))


u0 = np.arange(1,4,3)
u1 = np.arange(4,8,0.01)
u2 = np.arange(8,80,0.01)

v0 = (0.5-1)/(4-1)*u0+(7/6)*2
v1 = np.zeros(len(u1))+0.5*2
v2 = (5-0.5)/(80-8)*u2*2

u3 = np.concatenate((u0,u1,u2))
v3 = np.concatenate((v0,v1,v2))


u01 = np.arange(1,4,3)
u11 = np.arange(4,8,0.01)
u21 = np.arange(8,80,0.01)

v01 = (1-2)/(4-1)*u0+7/3*2
v11 = np.zeros(len(u1))+1*2
v21 = (10-1)/(80-8)*u2*2

u31 = np.concatenate((u01,u11,u21))
v31 = np.concatenate((v01,v11,v21))


u02 = np.arange(1,4,3)
u12 = np.arange(4,8,0.01)
u22 = np.arange(8,80,0.01)

v02 = (2-4)/(4-1)*u0+14/3*2
v12 = np.zeros(len(u1))+2*2
v22 = (20-2)/(80-8)*u2*2

u32 = np.concatenate((u02,u12,u22))
v32 = np.concatenate((v02,v12,v22))



u03 = np.arange(1,4,3)
u13 = np.arange(4,8,0.01)
u23 = np.arange(8,80,0.01)

v03 = (4-8)/(4-1)*u0+28/3*2
v13 = np.zeros(len(u1))+4*2
v23 = (40-4)/(80-8)*u2*2

u33 = np.concatenate((u03,u13,u23))
v33 = np.concatenate((v03,v13,v23))





fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

#v90 = ax.loglog(x34,y34)
v70 = ax.loglog(x33,y33)
#v50 = ax.loglog(x32,y32)
#v30 = ax.loglog(x31,y31)
#v10 = ax.loglog(x3,y3)

#iso_t = ax.loglog(u33,v33,'--')
#iso_r = ax.loglog(u32,v32,'--')
iso_o = ax.loglog(u31,v31,'--')
#iso_w = ax.loglog(u3,v3,'--')


model_1_before_finish = ax.plot(16.5253, acc_rms_cm,'cs')
ax.text(16.5253, acc_rms_cm, '   16.53 Hz,'+acc_rms_cm+'cm/$s^2$', fontweight='bold')

slab_210 = ax.plot(37.89, 11.9115,'bo')
ax.text(37.89, 11.9115, '   37.89 Hz, 11.9115cm/$s^2$', fontweight='bold')

#slab_240 = ax.plot(10.84, 15.1952,'r^')
#ax.text(21.21, 1.88, '   21.21 Hz, 1.88cm/$s^2$', fontweight='bold')




plt.xlabel('주파수(Hz)')
plt.ylabel('가속도(cm/$s^2$)')
plt.title('[슬래브 8100 x 7500] (위치 4번) 사용성 평가')
#plt.legend(["V-90 (AIJ)","V-70 (AIJ)","V-50 (AIJ)","V-30 (AIJ)","V-10 (AIJ)","Workshop (ISO)","Office (ISO)","Residential (ISO)","Operating Theatre (ISO 2631-2)", "균열부", "건전부", "슬래브 두께 240mm"], loc='upper right', bbox_to_anchor=(1.58,1))
plt.legend(["V-70 (AIJ)","Office (ISO)", "균열부", "건전부", "슬래브 두께 240mm"], loc='upper right', bbox_to_anchor=(1.58,1))


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