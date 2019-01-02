'''
Evaluation version 3
Technique 1: Weighted Least Squares estimation through hyperbolic technique
Author: Nasreen Mohsin
USAGE= python SFU_Localization_eval_v5.py TX MAX_NUM_pkts offline read_data.txt Position_no. sv save_file.txt
       python SFU_localization_eval_v5.py TX MAX_NUM_pkts online IP_addr Port  sv save_file.txt
Eg: python SFU_Localization_eval_v5.py -8 5 offline data_eval_pos2.txt 2  sv eval_v5_pos2.txt
'''

import os, sys, time
import Tkinter
from PIL import Image, ImageTk
from math import *
from collections import deque
#from scipy import optimize
import math
import socket
import numpy as np
from numpy import linalg
from scipy.stats.distributions import t as tt
import lmfit
from lmfit import Minimizer, Parameters, conf_interval


def button_click_exit_mainloop (event):
    event.widget.quit() # this will cause mainloop to unblock.
def distance_RSS (RSS,TX,n,PL1):
    return 10 ** (float(float(TX - RSS) - PL1) / (10 * n))

def cricle_distance(x1,y1,x2,y2):
    return math.sqrt((pow((x1-x2),2.0)+pow((y1-y2),2.0)))

def find_nearest(array, value):
    idx = (np.abs(np.asarray(array) - value)).argmin()
    return idx


def VAR_dist(d):
    return d ** 2


def se(params, locations, distances):
    x=[params['x'].value,params['y'].value]
    se = []
    cnt=0
    for location, distance,esp in zip(locations, distances,data_esp):
        if distance !=0:
            distance_calculated = cricle_distance(x[0], x[1], location[0], location[1])
            se.append(float(math.pow(distance_calculated - distance, 2.0)))
            cnt+=1

    return se

def M2pixel(x,y):
    grid_v=77
    grid_h=75
    return x*grid_v, y*grid_h

def create_grid(event=None):
    w = cv.winfo_width() # Get current width of canvas
    h = cv.winfo_height() # Get current height of canvas
    cv.delete('grid_line') # Will only remove the grid_line
    grid_int_v = 77
    grid_int_h = 75
    # Creates all vertical lines at intevals of 100
    for i in range(0, w, grid_int_v):
        cv.create_line([(i, 0), (i, h)], fill='light blue', tag='grid_line',dash=(3,5) )

    # Creates all horizontal lines at intevals of 100
    for i in range(0, h, grid_int_h):
        cv.create_line([(0, i), (w, i)], fill='light blue', tag='grid_line',dash=(3,5) )

# Create SFU Map
root = Tkinter.Tk()
#root.bind("<Button>", button_click_exit_mainloop)

## TX power

TX = int(sys.argv[1])  # -8,2

## Read from Fixed Scanners list
f1 = open("C:\Users\Nachu\Documents\MITACS\SFU_Lab_Analysis\Localization_Data\FixedScanners2.txt", "r")
data = f1.readline()
FS_list = []  # Fixed scanner list
nodeList = []
FS_posX = []  # Position of FS
FS_posY = []
FS_pos = []
FS_region = [];
FS_reg_dist = [];
FS_Ref_dist=[]
print "Fixed scanners ID and positions"
grid_int_v = 77
grid_int_h = 75
while (data):
    nodeList.append(data.strip().split(","))
    data = f1.readline()
for node_i in range(len(nodeList)):
    FS_list.append(nodeList[node_i][0])
    FS_posX.append((float(nodeList[node_i][1])))
    FS_posY.append(float(nodeList[node_i][2]))
    FS_pos.append([float(nodeList[node_i][1]), float(nodeList[node_i][2])])
    FS_region.append([])
    FS_reg_dist.append([])
    FS_Ref_dist.append([])
# print nodeList
f1.close()  # close the file

## Reference points posiitons
## Read from Ref points posiitons
f3 = open("C:\Users\Nachu\Documents\MITACS\SFU_Lab_Analysis\Localization_Data\Ref_pos_2.txt", "r")
data = f3.readline()
Ref_list = []  # Fixed scanner list
RList = []
Ref_posX = []  # Position of FS
Ref_posY = []
Ref_region = [];
if TX == 2:
    clus = 3
else:
    clus = 4
while (data):
    RList.append(data.strip().split(","))
    data = f3.readline()
for node_i in range(len(RList)):
    Ref_list.append(RList[node_i][0])
    Ref_posX.append(float(RList[node_i][1]))
    Ref_posY.append(float(RList[node_i][2]))
    for fs_id in range(len(FS_list)):
        FS_Ref_dist[fs_id].append(cricle_distance(FS_posX[fs_id], FS_posY[fs_id], float(RList[node_i][1]), float(RList[node_i][2])))

    if int(RList[node_i][clus]) != 0:
        Ref_region.append(int(RList[node_i][clus]) - 1)
        FS_region[int(RList[node_i][clus]) - 1].append(RList[node_i][0])
        tp = int(RList[node_i][clus]) - 1
        FS_reg_dist[tp].append(
            cricle_distance(FS_posX[tp], FS_posY[tp], float(RList[node_i][1]), float(RList[node_i][2])))

f3.close()  # close the file
# print FS_region
# print FS_reg_dist

#FS_list=['fbbbccddeeaa','fabbccddeeaa'];
beacon_list='ffbbccddeeaa';

# ////////////Initilization visualization window/////////////#

image1 = Image.open('C:\Users\Nachu\Documents\MITACS\SFU_LAB_Plan_2.png')
root.geometry('%dx%d' % (image1.size[0],image1.size[1]))

tkpi = ImageTk.PhotoImage(image1)
cv = Tkinter.Canvas(root, width=image1.size[0], height=image1.size[1], background="white", bd=1, relief=Tkinter.RAISED)
cv.grid(row=0, column=0)
cv.create_image(0, 0, image=tkpi, anchor='nw')
cv.bind('<Configure>', create_grid)
sz_beacon = 15  # size of beacon

# Draw scanners
scanners = []
sz_scanner = 15  # size of scanner

for i in range(len(FS_list)):
    xp, yp = M2pixel(abs(FS_posX[i]), abs(FS_posY[i]))
    scanner1 = cv.create_polygon(xp - sz_scanner, yp + sz_scanner,
                                 xp + sz_scanner, yp + sz_scanner,
                                 xp, yp - sz_scanner,
                                 fill='dark blue')
    cv.create_text(xp, yp, fill='white', text=str(i + 1))
    scanners.append(scanner1)

#   Draw Reference points
Refs = []
for i in range(len(Ref_list)):
    xp, yp = M2pixel(abs(Ref_posX[i]), abs(Ref_posY[i]))
    Ref1 = cv.create_oval(xp - sz_scanner, yp - sz_beacon,
                          xp + sz_beacon, yp + sz_beacon,
                          fill='white', outline='green')
    cv.create_text(xp, yp, fill='black', text=str(i + 1))
    Refs.append(Ref1)

# initializing beacons and confidence region
beacons = []
bufsz=3#no. of previous beacon positions to be displayed
beacon_color=['#ffdddd','#ffaaaa','#ff5555','#ff0101','#ff0000'] # fading colour for previous beacon positions
buffX=deque(maxlen=bufsz) # previous beacon positions in pixeco-ordinates
buffY=deque(maxlen=bufsz)

for cnt in range(bufsz):
    beacon1 = cv.create_oval(0 - sz_beacon, 0 - sz_beacon,
                             0 + sz_beacon, 0 + sz_beacon,
                             fill=beacon_color[cnt])
    beacons.append(beacon1) # intialize current and previously estimated positions of user
bec_txt = cv.create_text(0, 0, fill='white', text="user")#write text "User" on Map at estimated position

b_conf = cv.create_oval(-1, -1,
                        1, 1,
                        fill='#aaffaa') # initial estimated confidence region
b_conf1 = cv.create_oval(-1, -1,
                        1, 1,
                        fill='#aaffaa') # initial estimated confidence region
beacon_real=cv.create_oval(0 - sz_beacon, 0 - sz_beacon,
                             0 + sz_scanner, 0 + sz_scanner,
                            fill='#aaffdd')
real_bec_txt=cv.create_text(0, 0, fill='white', text="real")# write text "Real" on Map at actual position
#chart_1.grid(row=0, column=0)

ini_pos_beacon=M2pixel(Ref_posX[1],Ref_posY[1])
xini = ini_pos_beacon[0]
yini = ini_pos_beacon[1]
buffX.append(xini)
buffY.append(yini)
#i=0
root.title('SFU LAB')
#root.mainloop()

width_canvas = image1.size[0]  # Get current width of canvas
height_canvas = image1.size[1]



#packets = contents.splitlines()
old_time = 0 #int(contents[0].split(",")[0])

#f1_rss=[]
f1_rss = [[] for _ in range(len(FS_list))]
f1_d = [[] for _ in range(len(FS_list))]

force_rss=[0 for x in range(len(FS_list))]
totForce=0
temp_rss= [0 for x in range(len(FS_list))]
temp_rssN= [0 for x in range(len(FS_list))]
max_rss=-120
max_i=0
temp_ble= [0 for x in range(len(FS_list))]
avg=[[] for _ in range(len(FS_list))]

#wndw=7
if TX == 2:
    n = 1.38117035090007  #
    PL1 = 70.0021591930693  #
else:
    n = 1.26585551765074  #
    PL1 = 66.5407184253001  #

buffX1 = deque(maxlen=bufsz)
buffY1 = deque(maxlen=bufsz)
buff_time = deque(maxlen=bufsz)

t=0

if sys.argv[3]=='online':

    #Socket connection
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    host=sys.argv[4]
    port=int(sys.argv[5])
    s.bind((host,port))
    s.listen(1)
    c, addr = s.accept()
    print 'Got connection from', addr
    c.send('Thank you for connecting')
else:
    ## Read from data txt file##
    #data_txt='C:\Users\Nachu\Documents\MITACS\SFU_Lab_Analysis\Ble_eval_v7\data_eval_pos%d.txt' % int(sys.argv[2])
    #f2=open('C:\Users\Nachu\Documents\MITACS\SFU_Lab_Analysis\Localization_Data\data_test4_pos16.txt', 'r')
    data_txt=sys.argv[4]
    f2=open(data_txt,'r')
    contents = f2.readlines()
    f2.close()
    real_pos = int(sys.argv[5])
pkt_idx=0
###//Evaluation//###
if sys.argv[6]=='sv':
    f4 = open(sys.argv[7], 'w')

# No. of pkts
#MAX_NUM=int(sys.argv[2])
wndw=int(sys.argv[2])
#for packet in contents:


while True:
    if sys.argv[3]=='online':
        packet = c.recv(1024)
    else:
        if pkt_idx!=len(contents):
            packet =contents[pkt_idx]
            pkt_idx+=1
        else:
            break

    if packet in ['\n', '\r\n']:
        continue
    # print (packet)
    item = packet.split(",")
    #if item[1] != 'f3bbccddeeaa' and item[1] != 'f2bbccddeeaa' and item[1]!='f4bbccddeeaa': # no. of fixed scanners evaluation
    #    continue

    # Evaluation
    if sys.argv[3]=='online':
        if len(item) == 1: # from online
            print item[0]
            real_pos = int(item[0])
            if sys.argv[6]=='sv':
                f4.write("\n")  # evaluation
            continue
    if old_time == 0:
        old_time=int(item[0])
    new_time = int(item[0])
    if new_time-old_time >wndw:
    #if any(vt >= MAX_NUM for vt in temp_ble):
        fs_cnt=0 # no. of FS which detected beacon
        print temp_ble

        for ind,fs in enumerate(FS_list):
            if temp_ble[ind]!=0:
                fs_cnt += 1
                avg[ind] = float(temp_rss[ind] )/ float(temp_ble[ind])
                totForce += temp_rssN[ind]
            else:
                avg[ind]= -120

            if avg[ind] > max_rss:
                max_rss=avg[ind]
                max_i=ind

        #print avg
        #print max_rss

        #print max_i
        # Trilateration
        init_est_pos= [0 for x in range(2)]#FS_pos[max_i][:] # closest scanner

        d=[0 for x in range(len(FS_list))]
        d_ci = [0 for x in range(len(FS_list))]
        for ind in range(len(FS_list)):
            if temp_ble[ind]!=0:
                if temp_ble[ind]==1:
                    d[ind]=distance_RSS(avg[ind], TX, n, PL1)
                    d[ind] = distance_RSS(avg[ind], TX, n, PL1)
                    force_rss[ind] = (float(temp_rssN[ind]) / float(totForce))
                    d_ci[ind]=d[ind]
                else:
                    d[ind]=distance_RSS(avg[ind], TX, n, PL1)
                    force_rss[ind]=(float(temp_rssN[ind])/float(totForce))
                    # find confidence interval for distance estimation
                    f2_d = np.asarray(f1_d[ind])
                    SE =float(np.sqrt(np.sum((d[ind]-f2_d)**2)))/float(sqrt(temp_ble[ind]))
                    alpha = 0.05  # 100*(1 - alpha) confidence level
                    sT = tt.ppf(1.0 - alpha / 2.0, temp_ble[ind]-1)
                    d_ci[ind]=SE*sT

            else:
                d[ind]=0 # putting distance infinity
                d_ci[ind]=0
                force_rss[ind] = 0.0
        # Sort
        #imp_order = np.argsort(avg)
        first_fs = 0
        print "est_distances"
        print d
        A = []
        b = []
        S1=[]
        first_fs = -1
        if fs_cnt > 3:
            for ind in range(len(FS_list)):
                if d[ind] != 0:
                    if first_fs != -1:
                        # print d[first_fs]
                        #    print d[ind]
                        A.append([FS_posX[ind] - FS_posX[first_fs], FS_posY[ind] - FS_posY[first_fs]])
                        b.append([d[first_fs] ** 2 - d[ind] ** 2 + FS_posX[ind] ** 2 + FS_posY[ind] ** 2 - FS_posX[
                            first_fs] ** 2 - FS_posY[first_fs] ** 2])
                        S1.append(VAR_dist(d[first_fs] ** 2) + VAR_dist(d[ind] ** 2))

                    else:
                        first_fs = ind
                        continue

            As = np.asarray(A)
            bs = np.asarray(b)
            # Weights
            S2 = np.diag(np.asarray(S1))
            for i, val in np.ndenumerate(S2):
                if i[0] != i[1]:
                    S2[i] = VAR_dist(d[first_fs] ** 2)
            # print S2
            W = linalg.inv(S2)

            ATwA = np.dot(np.dot(As.T, W), As)
            ATwb = np.dot(np.dot(As.T, W), bs)
            # print ATwb
            b_pos, residuals, rank, sv = linalg.lstsq(ATwA, ATwb,rcond=-1)
            # compute the confidence intervals
            nr = len(bs)
            kr = len(b_pos)

            sigma2 = np.sum((bs - np.dot(As, b_pos)) ** 2) / (nr - kr)  # MSE


            C = sigma2 * np.linalg.inv(np.dot(As.T, As))  # covariance matrix
            se = np.sqrt(np.diag(C))  # standard error

            alpha = 0.05  # 100*(1 - alpha) confidence level

            sT = tt.ppf(1.0 - alpha / 2.0, max(1, nr - kr))  # student T multiplier
            tCI = sT * se  # confidence interval
            b_pos = b_pos / 2
            tCI = tCI / 2
            CI = [[tCI[0], tCI[0]], [tCI[1], tCI[1]]]

            # print b_pos

            '''if abs(b_pos[0]) > 6.87 or abs(b_pos[1]) > 7.6:
                stk = FS_reg_dist[max_i][:]
                idf = int(find_nearest(stk, d[max_i]))

                ref_id = int(FS_region[max_i][idf])
                #print ref_id
                b_pos = [Ref_posX[ref_id - 1], Ref_posY[ref_id - 1]]'''
        else:
                if d[max_i] != 0:
                    stk = FS_reg_dist[max_i][:]
                    idf1 = int(find_nearest(stk, d[max_i]))
                    idf2 = int(find_nearest(stk, d[max_i] - d_ci[max_i]))
                    idf3 = int(find_nearest(stk, d[max_i] + d_ci[max_i]))

                    ref_id1 = int(FS_region[max_i][idf1])
                    ref_id2 = int(FS_region[max_i][idf2])
                    ref_id3 = int(FS_region[max_i][idf3])
                # print ref_id
                    b_pos = [(Ref_posX[ref_id1 - 1] + Ref_posX[ref_id2 - 1] + Ref_posX[ref_id3 - 1]) / 3, (
                            Ref_posY[ref_id1 - 1] + Ref_posY[ref_id2 - 1] + Ref_posY[
                            ref_id3 - 1]) / 3]  # estimated beacon position
                # CI = [[d_ci[max_i],d_ci[max_i]], [d_ci[max_i],d_ci[max_i]]]
                    CI = [[d[max_i] - d_ci[max_i], d[max_i] + d_ci[max_i]],
                          [d[max_i] , d_ci[max_i]]]
                else:
                    b_pos = [buffX1[-1], buffY1[-1]]

        b_LU = []
        for beta, ci in zip(b_pos, CI):
            print ('%f [%f,%f]' % (
                beta, beta - ci[0], beta + ci[1]))  # {2: 1.2e} [{0: 1.4e} {1: 1.4e}]'.format(beta - ci, beta + ci, beta)
            b_LU.append([beta - ci[0], beta + ci[1]])
        buffX1.append(b_pos[0])
        buffY1.append(b_pos[1])
        # convert meters to pixel
        xnew, ynew = M2pixel(abs(b_pos[0]), abs(b_pos[1]))  # FS_posY[t])
        xy_c1 = M2pixel(CI[0][0], CI[1][0])
        xy_c2 = M2pixel(CI[0][1], CI[1][1])
        # for the case when fs_cnt <=2
        d_pix= M2pixel(d[max_i],d[max_i])
        fs_mx=M2pixel(abs(FS_posX[max_i]),abs(FS_posY[max_i]))

        buffX.append(int(xnew))
        buffY.append(int(ynew))

        # Error between real position and estimated position

        est_err = cricle_distance(Ref_posX[real_pos - 1], Ref_posY[real_pos - 1], b_pos[0], b_pos[1])
        print est_err

        # Drawing the beacon current position iwht its previous positions (darker red to lighter shade)

        if len(buffX) > 1:
            for cnt in range(len(buffX) - 1):
                cv.delete(beacons[cnt])
                beacons[cnt] = cv.create_oval(buffX[cnt] - sz_beacon, buffY[cnt] - sz_beacon,
                                              buffX[cnt] + sz_beacon, buffY[cnt] + sz_beacon,
                                              fill=beacon_color[cnt], outline='white')

        cv.delete(beacons[-1])
        beacons[-1] = cv.create_oval(buffX[-1] - sz_beacon, buffY[-1] - sz_beacon,
                                     buffX[-1] + sz_beacon, buffY[-1] + sz_beacon,
                                     fill=beacon_color[-1], outline='white')
        cv.delete(bec_txt)
        bec_txt = cv.create_text(buffX[-1], buffY[-1], fill='white', text="user")
        # Drawing confidence region
        cv.delete(b_conf)

        if all(vt!=float('Inf') and vt!=-float('Inf') for vt in CI[0]) and all(vt!=float('Inf') and vt!=-float('Inf') for vt in CI[1]) :
            if fs_cnt>3:
                b_conf = cv.create_oval(buffX[-1] - int(xy_c1[0]), buffY[-1] - int(xy_c1[1]),
                                            buffX[-1] + int(xy_c2[0]), buffY[-1] + int(xy_c2[1]),
                                    fill='', outline='red')
            else:
                b_conf = cv.create_oval(fs_mx[0] - int(xy_c1[1]), fs_mx[1] - int(xy_c1[1]),
                                    fs_mx[0] + int(xy_c1[1]), fs_mx[1] + int(xy_c1[1]),
                                    fill='', outline='red')
                # b_conf = cv.create_oval(fs_mx[0] - (d_pix[0]-int(xy_c1[0])), fs_mx[1] - (d_pix[1]-int(xy_c1[1])),
                #                        fs_mx[0] + (d_pix[0]-int(xy_c2[0])), fs_mx[1] + (d_pix[1]-int(xy_c2[1])),
                #                        fill='', outline='red')
                #cv.delete(b_conf1)
                #b_conf1 = cv.create_oval(fs_mx[0] - (d_pix[0]+int(xy_c1[0])), fs_mx[1] - (d_pix[1]+int(xy_c1[1])),
                #                        fs_mx[0] + (d_pix[0]+int(xy_c2[0])), fs_mx[1] + (d_pix[1]+int(xy_c2[1])),
                #                       fill='', outline='black')
        cv.delete(beacon_real)
        # Drawing real beacon positon
        xp, yp = M2pixel(abs(Ref_posX[real_pos - 1]), abs(Ref_posY[real_pos - 1]))
        beacon_real = cv.create_oval(xp - sz_beacon, yp - sz_beacon,
                                     xp + sz_beacon, yp + sz_beacon,
                                     fill='dark green', outline='green')
        cv.delete(real_bec_txt)
        real_bec_txt = cv.create_text(xp, yp, fill='white', text='real')


        cv.update()
        if sys.argv[6]=='sv':
            f4.write("%d,%d,%f,%f,%f,%f,%f,%f,%f,%d,%d\n" % (new_time, real_pos, b_pos[0], b_pos[1], CI[0][0], CI[0][1],CI[1][0],CI[1][1], est_err, sum(temp_ble),fs_cnt))  # evaluation
        temp_rss = [0 for x in range(len(FS_list))]
        temp_ble = [0 for x in range(len(FS_list))]
        f1_rss = [[] for _ in range(len(FS_list))]
        f1_d = [[] for _ in range(len(FS_list))]
        max_rss=-120
        max_i=0
        t+=1
        avg=[[] for _ in range(len(FS_list))]
        #print buffX
        #print buffY
        old_time = new_time
        buff_time.append(new_time)
    else:
        for ind, fs in enumerate(FS_list):
             if item[1] == fs:
                 f1_rss[ind].append(int(item[4]))
                 f1_d[ind].append(distance_RSS(int(item[4]), TX, n, PL1))
                 temp_rss[ind] += int(item[4])
                 temp_rssN[ind] += 120 + int(item[4])
                 temp_ble[ind] += 1
                 continue


    #i=i+1
    #x=x+25
    #y=y+25
    time.sleep(0.1)
#root.mainloop() # wait until user clicks the window
f4.close()
#s.hutdown()
#s.close()
#except Exception, e:
        # This is used to skip anything not an image.
        # Image.open will generate an exception if it cannot open a file.
        # Warning, this will hide other errors as well.
        #print('error')
        #pass
#        print('e')

