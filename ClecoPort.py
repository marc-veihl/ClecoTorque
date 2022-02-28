#Imports
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from fpdf import FPDF
from datetime import datetime
#Declarables

file_path = r'C:\Cleco Dumps\2-28-22 Gateway Optimus TTF'
torque_prom = 1 #ftlbs (used for peak detection of prevailing and max)
peak_distance = 800 #degrees (used for peak detection of prevailing and max)
Align_Torque = 1 #ftlbs (used to superimpose data)
Look_4_Peaks = 1 #set to 1 if you want to look for just max tq - 2 for prevailing study
TTF = 1 #Set to 1 if Torque to Failure Test
all_plots = 1 # Set to 1 to generate report with all plots attached (slow)
Low_pass_filter = 1 #set to 1 for noisy low torque data, set to 0 for no filter
trim = 0 #If set to zero no trimming

#Main Body

csv_list = glob.glob(file_path +"/*.csv") #Create list of csv's found in file_path dir
super_impose = np.empty(0) #Initialize this array for later
num_of_files = len(csv_list) # Find the number of files to iter. thru
# Bunch of Init. of lists used to build arrays later
plot_namer = []
file_num = []
prevailing_tqs = []
max_tqs = []
Angle_1 = []
Angle_2 = []
n = 0
# Main Loop for Plotting each of the individual rundown plots
for i,s in enumerate(csv_list):
    file_name = csv_list[i]
    file_number = file_name[-12:-4]
    file_num.append((file_number))
    my_data = np.empty(0)
    f = open(file_name,"r")
    my_data = f.readlines()
    # These two lines strip out the weird format data at the beg & end
    my_data = my_data[:-5]
    my_data = my_data [9:]
    x = []
    y = []
    for i,s in enumerate(my_data):
        split_it = my_data[i].split(";")
        for i,s in enumerate(split_it):
            split_it[i] = split_it[i].replace(",",".")
        x.append(float(split_it[0])) # Angle in degrees
        y.append(float(split_it[1])) # Tq in ftlbs.
    if Low_pass_filter == 1: # If active pass data thru LP Butter Filter
        sos = signal.butter(3,10/60,'lp',fs=3, output='sos')
        y = signal.sosfilt(sos,y)
        y = y.tolist() # Convert signal back to a list
    plt.plot(x,y) # Plot
    plt.xlabel('Angle in degrees')
    plt.ylabel('Torque in ftlbs.')
    plt.title(file_number)
    # Logic for different types of runs
    if Look_4_Peaks >= 1 and TTF == 1:
        max_y = max(y)
        max_x = y.index(max_y)
        max_tqs.append(max_y)
        plt.plot(max_x,max_y,'ro',fillstyle='none') # Red circle on Max
    else:
        max_y = y[-1]
        max_x = x[-1]
        max_tqs.append(max_y)
        plt.plot(max_x,max_y,'ro',fillstyle='none') # Red circle on Max
    if Look_4_Peaks == 2:
        peaks = []
        peaks, _ = signal.find_peaks(y,prominence = torque_prom, distance = peak_distance)
        peaks = peaks.tolist()
        if len(peaks) >= 2: # This is to deal with different length lists, 
            max_list = []
            for i,s in enumerate(peaks):
                peak_val = y[peaks[i]]
                max_list.append(peak_val)
            local_max = max(max_list)
            local_ind = y.index(local_max)
            prevailing_tqs.append(local_max)
            # Green circle on Prevailing Max
            plt.plot(local_ind,local_max,'go',fillstyle='none')
        else:
            prevailing_tqs.append(y[int(peaks[0])])
            # Green circle on Prevailing Max
            plt.plot(int(peaks[0]),y[int(peaks[0])],'go',fillstyle='none')
    plt.grid()
    plt.figure
    plot_namer.append(file_path + '\plot_' + file_number + '.png')
    plt.savefig(plot_namer[n], dpi = 300)
    n+= 1
    plt.show()
#SuperImpose Plot
plt.figure()
for i,s in enumerate(csv_list):
    file_name = csv_list[i]
    file_number = file_name[-12:-4]
    my_data = np.empty(0)
    f = open(file_name,"r")
    my_data = f.readlines()
    my_data = my_data[:-5]
    my_data = my_data [9:]
    x = []
    y = []
    for i,s in enumerate(my_data):
        split_it = my_data[i].split(";")
        for i,s in enumerate(split_it):
            split_it[i] = split_it[i].replace(",",".")
        x.append(float(split_it[0]))
        y.append(float(split_it[1]))
    if Low_pass_filter == 1:
        sos = signal.butter(6,10/60,'lp',fs=3, output='sos')
        y = signal.sosfilt(sos,y)
        y = y.tolist()
    if Look_4_Peaks == 2:
        peaks = []
        peaks, _ = signal.find_peaks(y,prominence = torque_prom, distance = peak_distance)
        peaks = peaks.tolist()
        if len(peaks) >= 2:
            max_list = []
            for i,s in enumerate(peaks):
                peak_val = y[peaks[i]]
                max_list.append(peak_val)
            local_max = max(max_list)
            local_ind = y.index(local_max)
            align_data = y
            align_data = align_data[1:local_ind]
            diff_list = []
            for i,s in enumerate(align_data):
                diff_list.append(abs(align_data[i]-Align_Torque))
            min_dist = min(diff_list)
            indexofMin = diff_list.index(min_dist)
            shifted_data = []
            for i,s in enumerate(y):
                shifted_data.append(x[i] - indexofMin)
            Angle_1.append(indexofMin)
            Angle_2.append(len(align_data)-indexofMin)
            plt.plot(shifted_data,y)
        else:
            local_max = y[int(peaks[0])]
            local_ind = int(peaks[0])
            align_data = y
            align_data = align_data[1:local_ind]
            diff_list = []
            for i,s in enumerate(align_data):
                diff_list.append(abs(align_data[i]-Align_Torque))
            min_dist = min(diff_list)
            indexofMin = diff_list.index(min_dist)
            shifted_data = []
            for i,s in enumerate(y):
                shifted_data.append(x[i] - indexofMin)
            Angle_1.append(indexofMin)
            Angle_2.append(len(align_data)-indexofMin)
            plt.plot(shifted_data,y)
            
    if Look_4_Peaks == 1:
        max_y = max(y)
        max_x = y.index(max_y)
        align_data = y
        align_data = align_data[1:max_x]
        diff_list = []
        for i,s in enumerate(align_data):
            diff_list.append(abs(align_data[i]-Align_Torque))
        min_dist = min(diff_list)
        indexofMin = diff_list.index(min_dist)
        shifted_data = []
        for i,s in enumerate(y):
            shifted_data.append(x[i] - indexofMin)
        Angle_1.append(indexofMin)
        Angle_2.append(len(align_data)-indexofMin)
        plt.plot(shifted_data,y)

plt.xlabel('Angle in degrees')
plt.ylabel('Torque in ftlbs.')
plt.title('Angle Zeroed at '+str(Align_Torque)+' ftlbs.')
plt.legend(file_num, fontsize = 'x-small')
plt.grid()
plt.savefig(file_path + '\SuperImpose.png', dpi = 300)
plt.show

#Stats
Stat_list = np.zeros((7,1))
Stat_list = np.array(['N','Mean','STD DEV','X -3STD','X +3STD','MIN','MAX'])
Stat_list = np.transpose(Stat_list)    
if Look_4_Peaks == 2:
    
    head = ['Files','PrevTorque','MaxTorque','Angle1','Angle2']
    table = np.array([file_num,prevailing_tqs,max_tqs,Angle_1,Angle_2])
    table = np.transpose(table)
    data_table = np.array([prevailing_tqs,max_tqs,Angle_1,Angle_2])
    results_table = np.zeros((7,4))
    results_table[0,:] = num_of_files
    results_table[1,:] = np.mean(data_table,axis = 1)
    results_table[2,:] = np.std(data_table, axis = 1)
    results_table[3,:] = results_table[1,:] - 3*results_table[2,:]
    results_table[4,:] = results_table[1,:] + 3*results_table[2,:]
    results_table[5,:] = np.min(data_table,axis = 1)
    results_table[6,:] = np.max(data_table,axis = 1)
    results_table = np.around(results_table,3)
    new_results = np.column_stack((Stat_list,results_table))

if Look_4_Peaks == 1:
    head = ['Files','MaxTorque','Angle1','Angle2']
    table = np.array([file_num,max_tqs,Angle_1,Angle_2])
    table = np.transpose(table)
    data_table = np.array([max_tqs,Angle_1,Angle_2])
    results_table = np.zeros((7,3))
    results_table[0,:] = num_of_files
    results_table[1,:] = np.mean(data_table,axis = 1)
    results_table[2,:] = np.std(data_table, axis = 1)
    results_table[3,:] = results_table[1,:] - 3*results_table[2,:]
    results_table[4,:] = results_table[1,:] + 3*results_table[2,:]
    results_table[5,:] = np.min(data_table,axis = 1)
    results_table[6,:] = np.max(data_table,axis = 1)
    results_table = np.around(results_table,3)
    new_results = np.column_stack((Stat_list,results_table))
    
# PDF Report Generator
pdf = FPDF()
pdf_target = file_path + '\Torque_Report.pdf'
pdf.add_page()
pdf.set_xy(0, 0)
pdf.set_font('arial', 'B', 12)
pdf.cell(60)
pdf.cell(75, 7, file_path + " Torque Report", 0, 2, 'C')
now = datetime.now()
pdf.cell(90, 7, now.strftime("%m/%d/%Y, %H:%M:%S"), 0, 2, 'C')
pdf.cell(-30)
pdf.image(file_path + '\SuperImpose.png', x = 18, y = None, w = 180, type = '', link = '')
pdf.cell(160, 7, "", 0, 2, 'C')
pdf.cell(145, 7, "Torque Study Results:", 0, 2, 'C')
pdf.cell(-15)
pdf.set_font('arial','b', 10)
if Look_4_Peaks == 2:
    pdf.cell(35, 5, 'Files', 1, 0, 'C',)
    pdf.cell(35, 5, 'Prev Torque', 1, 0, 'C')
    pdf.cell(35, 5, 'Max/Seat Torque', 1, 0, 'C')
    pdf.cell(35, 5, 'Angle1', 1, 0, 'C')
    pdf.cell(35, 5, 'Angle2', 1, 1, 'C')
    pdf.set_font('arial','', 10)
    for i,j in enumerate(table):
        pdf.cell(5)
        pdf.cell(35, 5, table[i,0], 1, 0, 'C')
        pdf.cell(35, 5, "{:.3f}".format(float(table[i,1])), 1, 0, 'C')
        pdf.cell(35, 5, "{:.3f}".format(float(table[i,2])), 1, 0, 'C')
        pdf.cell(35, 5, table[i,3], 1, 0, 'C')
        pdf.cell(35, 5, table[i,4], 1, 1, 'C')
    pdf.set_font('arial', 'B', 12)
    pdf.cell(185, 7, "Statistical Data:", 0, 2, 'C')
    for i,j in enumerate(new_results):
        pdf.cell(5)
        pdf.set_font('arial','', 10)    
        pdf.cell(35, 5, new_results[i,0], 1, 0, 'C')
        pdf.cell(35, 5, new_results[i,1], 1, 0, 'C')
        pdf.cell(35, 5, new_results[i,2], 1, 0, 'C')
        pdf.cell(35, 5, new_results[i,3], 1, 0, 'C')
        pdf.cell(35, 5, new_results[i,4], 1, 1, 'C')
        
if Look_4_Peaks == 1:
    pdf.cell(43.75, 5, 'Files', 1, 0, 'C',)
    pdf.cell(43.75, 5, 'Max/Seat Torque', 1, 0, 'C')
    pdf.cell(43.75, 5, 'Angle1', 1, 0, 'C')
    pdf.cell(43.75, 5, 'Angle2', 1, 1, 'C')
    pdf.set_font('arial','', 10)
    for i,j in enumerate(table):
        pdf.cell(5)
        pdf.cell(43.75, 5, table[i,0], 1, 0, 'C')
        pdf.cell(43.75, 5, "{:.3f}".format(float(table[i,1])), 1, 0, 'C')
        pdf.cell(43.75, 5, table[i,2], 1, 0, 'C')
        pdf.cell(43.75, 5, table[i,3], 1, 1, 'C')
    pdf.set_font('arial', 'B', 12)
    pdf.cell(185, 7, "Statistical Data:", 0, 2, 'C')
    for i,j in enumerate(new_results):
        pdf.cell(5)
        pdf.set_font('arial','', 10)    
        pdf.cell(43.75, 5, new_results[i,0], 1, 0, 'C')
        pdf.cell(43.75, 5, new_results[i,1], 1, 0, 'C')
        pdf.cell(43.75, 5, new_results[i,2], 1, 0, 'C')
        pdf.cell(43.75, 5, new_results[i,3], 1, 1, 'C')

if all_plots == 1:
    pdf.add_page()
    for i,j in enumerate(plot_namer):
        pdf.image(plot_namer[i], x = 18, y = None, w = 180, type = '', link = '')
print()
print("Report Generated @ " + pdf_target)
pdf.output(pdf_target, 'F')
