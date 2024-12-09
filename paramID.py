import mcap.stream_reader
import numpy as np
import mcap 
from mcap.records import Header, Schema, Channel, Message, Attachment, Statistics
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear
from scipy.optimize import minimize
from scipy.linalg import expm
import pymap3d
import matplotlib.pyplot as plt
import math
import csv
import pandas as pd




def read_in_mcap(file_path, measurement, Topic):
    # open the file from the given path
    reader = make_reader(open(file_path, "rb"), decoder_factories=[DecoderFactory()])
    # with open(file_path, "rb") as file:
    #     reader = make_reader(file,decoder_factories=[DecoderFactory()])

    # initialize the empty output array
    number_of_measurements = len(measurement)
    output = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        output[i-1] = []

    # read the desired data into output array
    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=Topic):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        output[0].append(time)
        # read all the other data
        for i in range(1,number_of_measurements + 1):
            if len(measurement[i-1])>1:
                tmp = 0
                for j in range(len(measurement[i-1])):
                    if j==0:
                        tmp = getattr(dec_msg,measurement[i-1][0])
                    else:
                        tmp = getattr(tmp,measurement[i-1][j])
            else:
                tmp = measurement[i-1][0]
                tmp = getattr(dec_msg,tmp)

            output[i].append(tmp)
                
    return output

def get_derivative(data):
    length = len(data[:,0])
    derivative = np.zeros((length,2))
    derivative[0,0] = data[0,0]
    for i in range(1,length):
        derivative[i,0] = data[i,0]
        delta_t = data[i,0] - data[i-1,0]
        if delta_t < 0.00001:
            derivative[i,1] = 0.0
        else:
            derivative[i,1] = (data[i,1] - data[i-1,1]) / delta_t

    return derivative 

def get_imu_in_body(imu_data, rpy):
    ENU_angles = np.zeros((len(imu_data[:,0]),4))
    ENU_rot_vel = np.zeros((len(imu_data[:,0]),4))
    ENU_angles[:,0] = imu_data[:,0]
    ENU_rot_vel[:,0] = imu_data[:,0]
    rot_imu_to_body = Rotation.from_euler('xyz', rpy) # bno055 0, 0, -np.pi/2 xsense: np.pi, 0.0, -0.5*np.pi
    # rot_imu_to_body = rot_body_to_imu.inv()

    # quats = np.array([np.array(imu_data[1]).T,np.array(imu_data[2]).T,np.array(imu_data[3]).T,np.array(imu_data[4]).T]).T
    quats = np.array([imu_data[:,1], imu_data[:,2], imu_data[:,3], imu_data[:,4]]).T
    imu_absolute_orientation = Rotation.from_quat(quats)
    rot_enu_to_body = imu_absolute_orientation * rot_imu_to_body
    # rot_body_to_enu = rot_enu_to_body.inv()
    
    # ENU_angles[:,1:] = (imu_absolute_orientation * rot_imu_to_body).as_euler('xyz')
    ENU_angles[:,1:] = rot_enu_to_body.as_euler('xyz')

    rot_vel = np.array([imu_data[:,5], imu_data[:,6], imu_data[:,7]]).T
    ENU_rot_vel[:,1:] = rot_imu_to_body.apply(np.squeeze(rot_vel)) 

    return ENU_angles, ENU_rot_vel

def GPS_to_ENU(gps_data):
    lat_0 = gps_data[1][0]
    lon_0 = gps_data[2][0]
    h_0 = gps_data[3][0]

    length = len(gps_data[0])
    ENU_coord = np.zeros((length,4))
    for i in range(length):
        ENU_coord[i,0] = gps_data[0][i]
        e,n,u = pymap3d.enu.geodetic2enu(gps_data[1][i], gps_data[2][i], gps_data[3][i], lat_0, lon_0, h_0)
        ENU_coord[i,1] = e
        ENU_coord[i,2] = n
        ENU_coord[i,3] = u

    return ENU_coord

def GPS_to_vel(gps_data):
    length = len(gps_data[0])
    x_vel = np.zeros((length,2))
    y_vel = np.zeros((length,2))

    for i in range(length):
        time = gps_data[0][i]
        x_vel[i,0] = time
        y_vel[i,0] = time
        x_vel[i,1] = gps_data[1][i]
        y_vel[i,1] = gps_data[2][i]

    return x_vel, y_vel

def convert_enu_to_body(imu_data, gps_pos, gps_vel, rpy_imu):
    
    # print(len(imu_data[:,0]))
    # print(len(gps_pos[:,0]))
    # print(len(gps_vel[0]))


    length = len(imu_data[:,0])
    GPS_body = np.zeros((length,5))
    start = 0
    if length < len(gps_pos[:,0]):
        start = len(gps_pos[:,0]) - length


    # print(gps_pos[start+100,0])
    # print(gps_vel[0][start+101])
    # print(imu_data[100,0])

    trans_body_to_gps = np.array([-0.145, 0.015, 0.16])
    rot_imu_to_body = Rotation.from_euler('xyz', rpy_imu)
    # rot_imu_to_body = rot_body_to_imu.inv()
    r_gps_body = trans_body_to_gps

    # print(gps_vel[0][0+start])
    # print(gps_pos[0+start,0])
    # print(imu_data[0,0])
    # print(length)
    # print(len(gps_vel[1]))
    # print(start)
    offset = 0
    for i in range(length):
        GPS_body[i,0] = imu_data[i,0]

        ##############################################################################
        # Position transformation
        imu_abs_orientation = Rotation.from_quat(imu_data[i,1:5])
        rot_body_to_enu = imu_abs_orientation * rot_imu_to_body
        gps_offset_enu = rot_body_to_enu.apply(trans_body_to_gps)

        GPS_body[i,1] = gps_pos[i+start,1] - gps_offset_enu[0]
        GPS_body[i,2] = gps_pos[i+start,2] - gps_offset_enu[1]

        ##############################################################################
        # Velocity transformation
        # print("----------------------------")
        # print(f"GPS time with offset: {gps_vel[0][i+offset+start]}")
        # print(f"IMU time:             {imu_data[i,0]}")
        gps_to_enu_rotation = rot_body_to_enu
        velocity_gps_frame = np.array([gps_vel[1][i+offset+start], gps_vel[2][i+offset+start], gps_vel[3][i+offset+start]])
        gps_to_body_rotation = gps_to_enu_rotation.inv()
        velocity_body_frame = gps_to_body_rotation.apply(velocity_gps_frame)

        # Adjusting the linear velocity by the rotational part
        angular_velocity_imu_frame = np.array([imu_data[i,5], imu_data[i,6], imu_data[i,7]])
        angular_velocity_body_frame = rot_imu_to_body.apply(angular_velocity_imu_frame)
        
        rotational_velocity_gps_body = np.cross(angular_velocity_body_frame, r_gps_body)
        velocity_gps_translational_body = velocity_body_frame
        true_velocity_body_frame = velocity_gps_translational_body - rotational_velocity_gps_body

        GPS_body[i,3] = true_velocity_body_frame[0]
        GPS_body[i,4] = true_velocity_body_frame[1]

    return GPS_body

def get_force(m):
    if m>0.13:
        f = (m-0.13)/0.6437048*65
    elif m<(-0.13):
        f = (m+0.13)/0.6437048*65
    else:
        f = 0
    
    return f

def get_force_and_moment_from_motors(motor,radius):
    length = len(motor[0])
    force_x = np.zeros((length,2))
    force_y = np.zeros((length,2))
    moment_z = np.zeros((length,2))

    for i in range(length):
        time = motor[0][i]
        force_x[i,0] = time
        force_y[i,0] = time
        moment_z[i,0] = time
        # F_i = ((m_i -  0.13)/0.6437048) * 65 N

        f1 = get_force(motor[1][i])
        f2 = get_force(motor[2][i])
        f3 = get_force(motor[3][i])
        f4 = get_force(motor[4][i])
        force_x[i,1] = f2 - f4
        force_y[i,1] = f3 - f1
        moment_z[i,1] = (f1 + f2 + f3 + f4)*radius
        # fig, axs = plt.subplots(4,1, sharex=True)
        # axs[0].plot()
        # print(f"f1: {f1}")
        # print(f"f2: {f2}")
        # print(f"f3: {f3}")
        # print(f"f4: {f4}")

    return force_x, force_y, moment_z

def get_force_and_moment_from_motors_new(motor,radius):
    length = len(motor[0])
    force_x = np.zeros((length,2))
    force_y = np.zeros((length,2))
    moment_z = np.zeros((length,2))

    for i in range(length):
        time = motor[0][i]
        force_x[i,0] = time
        force_y[i,0] = time
        moment_z[i,0] = time
        # F_i = ((m_i -  0.13)/0.6437048) * 65 N

        # f1 = get_force(motor[1][i])
        # f2 = get_force(motor[2][i])
        # f3 = get_force(motor[3][i])
        # f4 = get_force(motor[4][i])
        force_x[i,1] = 65*2*motor[1][i]
        force_y[i,1] = 65*2*motor[2][i]
        moment_z[i,1] = 65*radius*motor[6][i]

    
    return force_x, force_y, moment_z

def get_vel_from_imu(imu_data):
    lenght = len(imu_data[0])
    time_vec = np.zeros((lenght))
    vel_x = np.zeros((lenght,2))
    vel_y = np.zeros((lenght,2))
    delta_t = 0

    for i in range(lenght):
        time = imu_data[0][i]
        time_vec[i] = time
        vel_x[i,0] = time
        vel_y[i,0] = time
    
    for i in range(1,lenght):
        delta_t = time_vec[i] - time_vec[i-1]
        vel_x[i,1] = vel_x[i-1,1] + imu_data[6][i] * delta_t
        vel_y[i,1] = vel_y[i-1,1] + imu_data[7][i] * delta_t

    return vel_x, vel_y

def get_acc_from_imu(imu_data):
    lenght = len(imu_data[0])
    time_vec = np.zeros((lenght))
    acc_x = np.zeros((lenght,2))
    acc_y = np.zeros((lenght,2))
    yaw_rate = np.zeros((lenght,2))

    for i in range(lenght):
        time = imu_data[0][i]
        time_vec[i] = time
        acc_x[i,0] = time
        acc_y[i,0] = time
        yaw_rate[i,0] = time

    for i in range(1,lenght):
        acc_x[i,1] = imu_data[6][i]
        acc_y[i,1] = imu_data[7][i]
        yaw_rate[i,1] = imu_data[5][i] # math.radians(imu_data[5][i])

    return acc_x, acc_y, yaw_rate

def get_same_size(min_data,bigger_data):
    # min => GPS
    # big => IMU

    small = 0
    while min_data[small,0] < (bigger_data[0,0]):
        small += 1

    length = len(min_data[:,0]) - small # length to which the bigger data has to be reduced
    big_length = len(bigger_data[:,0])  
    sec_dim = len(bigger_data[0,:])
    if (length>big_length):
        return

    # min_data => gps
    # bigger_data => imu
    reduced_data = np.zeros((length,sec_dim))
    offset = small
    counter = 0
    for i in range(big_length):
        if bigger_data[i,0] >= min_data[small,0]:
            reduced_data[counter,:] = bigger_data[i,:]
            counter += 1
            small += 1
        
        if (small >= length):
            break
        

    return reduced_data

def synchronize_imu_gps(gps_data, imu_data, threshold):
    """
    Synchronizes IMU data to GPS data based on the closest timestamp within a threshold.
    Skips timestamps until both IMU and GPS data are available.
    
    Parameters:
        imu_data (np.ndarray): Array of IMU data with timestamps in the first column.
        gps_data (np.ndarray): Array of GPS data with timestamps in the first column.
        threshold (float): Maximum allowable time difference for synchronization.
        
    Returns:
        np.ndarray: Reduced IMU data aligned with GPS timestamps.
    """
    # Extract timestamps
    imu_timestamps = imu_data[:, 0]
    gps_timestamps = gps_data[:, 0]
    
    # Initialize a list to store synchronized IMU data
    reduced_imu_data = []
    
    # Flag to start synchronization once both datasets have overlapping data
    synchronization_started = False
    
    # Initialize an index to keep track of where we are in IMU data
    imu_index = 0
    
    # Iterate through GPS timestamps
    for gps_time in gps_timestamps:
        # Skip until we have both IMU and GPS data within the threshold
        if not synchronization_started:
            closest_imu_time = imu_timestamps[np.argmin(np.abs(imu_timestamps - gps_time))]
            time_diff = abs(closest_imu_time - gps_time)
            if time_diff <= threshold:
                synchronization_started = True  # Start synchronization
                
        # If synchronization hasn't started, skip this GPS timestamp
        if not synchronization_started:
            continue
        
        # Ensure IMU timestamps are available
        while imu_index < len(imu_timestamps) and imu_timestamps[imu_index] < gps_time - threshold:
            imu_index += 1
        
        # If we've run out of IMU data, break
        if imu_index >= len(imu_timestamps):
            break
        
        # Find the closest IMU timestamp to the current GPS timestamp
        time_diff = np.abs(imu_timestamps[imu_index] - gps_time)
        if time_diff <= threshold:
            reduced_imu_data.append(imu_data[imu_index])
    
    # Convert the result list to a numpy array
    reduced_imu_data = np.array(reduced_imu_data)
    
    return reduced_imu_data

def synchronize_input_gps(gps_data, input_data):

    gps_start = 0
    input_start = 0
    while gps_data[0,0] - input_data[input_start,0] > 0.08:
        input_start += 1
    while input_data[0,0] - gps_data[gps_start,0] > 0.08:
        gps_start += 1

    length = len(gps_data[:,0]) - gps_start 
    reduced_input_data = np.zeros((length,2))
    # print(input_start)
    # print(length)
    input_counter = 0
    for i in range(length): # looping in gps data
        while input_data[input_counter + input_start,0] <= gps_data[i + gps_start,0]:
            input_counter += 1
            if (input_counter + input_start) == len(input_data[:,0]):
                break
        if (input_counter + input_start) >= len(input_data[:,0]):
            break
        else:
            reduced_input_data[i,:] = input_data[input_counter + input_start - 1,:]


    return reduced_input_data

def get_matrices_for_ls_all(u_r,v_r,u_r_dot,v_r_dot,x,y,psi,x_dot,y_dot,psi_dot,x_ddot,y_ddot,psi_ddot,F_x,F_y,M_z,m):
    length = len(u_r[:,0])
    weigth = np.array(np.zeros((8*length,18)))
    target = np.zeros((length*8,1))
    # print(length)
    # print(target.shape)
    r = psi_dot
    r_dot = psi_ddot
    
    for i in range(length):
        #                        0         1         2        3         4        5     6    7    8     9    10        11        12        13        14        15          16           17
        #                        X_u_dot,  Y_v_dot,  Y_r_dot, N_v_dot,  N_r_dot, X_u,  Y_v, Y_r, N_v,  N_r, omega_01, omega_02, omega_03, zeta_1,   zeta_2,   zeta_3,     u_c,         v_c
        # weight   = np.matrix([ [-u_r_dot, v_r*r,    r**2,    0,        0,       -u_r, 0,   0,   0,    0,   0,        0,        0,        0,        0,        0,          0,           0],
        #                  1     [-u_r*r,   -v_r_dot, -r_dot,  0,        0,       0,    v_r, r,   0,    0,   0,        0,        0,        0,        0,        0,          0,           0],
        #                  2     [u_r*v_r,  -v_r*u_r, -u_r*r,  -v_r_dot, r_dot,  0,    0,   0,   -v_r, -r,  0,        0,        0,        0,        0,        0,          0,           0],
        #                  3     [0,        0,        0,       0,        0,       0,    0,   0,   0,    0,   0,        0,        0,        0,        0,        0,          np.cos(psi), -np.sin(psi)],
        #                  4     [0,        0,        0,       0,        0,       0,    0,   0,   0,    0,   0,        0,        0,        0,        0,        0,          np.sin(psi), np.cos(psi)],
        #                  5     [0,        0,        0,       0,        0,       0,    0,   0,   0,    0,   -x,       0,        0,        -2*x_dot, 0,        0,          0,           0],
        #                  6     [0,        0,        0,       0,        0,       0,    0,   0,   0,    0,   0,        -y,       0,        0,        -2*y_dot, 0,          0,           0],
        #                  7     [0,        0,        0,       0,        0,       0,    0,   0,   0,    0,   0,        0,        -psi,     0,        0,        -2*psi_dot, 0,           0]])
        
        
        # first row
        weigth[8*i,0] = -u_r_dot[i,1]
        weigth[8*i,1] = v_r[i,1]*r[i,1]
        weigth[8*i,2] = (r[i,1])**2
        weigth[8*i,5] = -u_r[i,1]
        # second row
        weigth[8*i+1,0] = -u_r[i,1]*r[i,1]
        weigth[8*i+1,1] = -v_r_dot[i,1]
        weigth[8*i+1,2] = -r_dot[i,1]
        weigth[8*i+1,6] = -v_r[i,1]
        weigth[8*i+1,7] = -r[i,1]
        # third row
        weigth[8*i+2,0] = u_r[i,1]*v_r[i,1]   # X_u_dot
        weigth[8*i+2,1] = -v_r[i,1]*u_r[i,1]  # Y_v_dot 
        weigth[8*i+2,2] = -u_r[i,1]*r[i,1]    # Y_r_dot
        weigth[8*i+2,3] = -v_r_dot[i,1]       # N_v_dot
        weigth[8*i+2,4] = r_dot[i,1]          # N_r_dot
        weigth[8*i+2,8] = -v_r[i,1]           # N_v
        weigth[8*i+2,9] = -r[i,1]             # N_r
        # fourth row
        weigth[8*i+3,16] = np.cos(psi[i,1])
        weigth[8*i+3,17] = -np.sin(psi[i,1])
        # fifth row
        weigth[8*i+4,16] = np.sin(psi[i,1])
        weigth[8*i+4,17] = np.cos(psi[i,1])
        # sixth row
        weigth[8*i+5,10] = -x[i,1]
        weigth[8*i+5,13] = -2*x_dot[i,1]
        # seventh row
        weigth[8*i+6,11] = -y[i,1]
        weigth[8*i+6,14] = -2*y_dot[i,1]
        # eigth row 
        weigth[8*i+7,12] = -psi[i,1]
        weigth[8*i+7,15] = -2*psi_dot[i,1]

        
        # Build the target vector
        # target = np.array([[F_x+m*(-u_r_dot+r*v_r), F_y-m*(v_r+r*u_r), M_z-I_z*r_dot, x_dot-u_r*np.cos(psi)+v_r*np.sin(psi), y_dot-u_r*np.sin(psi)-v_r*np.cos(psi), x_dot_dot, y_dot_dot, psi_dot_dot]])
        target[8*i,0] = F_x[i,1]+m*(-u_r_dot[i,1]+r[i,1]*v_r[i,1])
        target[8*i+1,0] = F_y[i,1]-m*(v_r_dot[i,1]+r[i,1]*u_r[i,1])
        target[8*i+2,0] = M_z[i,1]
        target[8*i+3,0] = x_dot[i,1]-u_r[i,1]*np.cos(psi[i,1])+v_r[i,1]*np.sin(psi[i,1])
        target[8*i+4,0] = y_dot[i,1]-u_r[i,1]*np.sin(psi[i,1])-v_r[i,1]*np.cos(psi[i,1])
        target[8*i+5,0] = x_ddot[i,1]
        target[8*i+6,0] = y_ddot[i,1]
        target[8*i+7,0] = psi_ddot[i,1]
    


    return weigth, target

def get_matrices_for_ls_xy(u_r,v_r,u_r_dot,v_r_dot,psi,x_dot,y_dot,F_x,F_y,M_z,m):

    length = len(u_r[:,0])
    weigth = np.array(np.zeros((7*length,12)))
    target = np.zeros((length*7,1))
    # print(length)
    # print(target.shape)
    
    for i in range(length):
        #                        0          1         2         3     4    5     6         7         8         9         10           11 
        #                        X_u_dot,   Y_v_dot,  N_v_dot,  X_u,  Y_v, N_v,  omega_01, omega_02, zeta_1,   zeta_2,   u_c,         v_c

        # weight   =       0     [-u_r_dot, 0,        0,        -u_r, 0,   0,    0,        0,        0,        0,        0,           0],
        #                  1     [0,        -v_r_dot, 0,        0,    v_r, 0,    0,        0,        0,        0,        0,           0],
        #                  2     [u_r*v_r,  -v_r*u_r, -v_r_dot, 0,    0,   -v_r, 0,        0,        0,        0,        0,           0],
        #                  3     [0,        0,        0,        0,    0,   0,    0,        0,        0,        0,        np.cos(psi), -np.sin(psi)],
        #                  4     [0,        0,        0,        0,    0,   0,    0,        0,        0,        0,        np.sin(psi), np.cos(psi)],
        #                  5     [0,        0,        0,        0,    0,   0,    -x,       0,        -2*x_dot, 0,        0,           0],
        #                  6     [0,        0,        0,        0,    0,   0,    0,        -y,       0,        -2*y_dot, 0,           0],
        
        
         # first row
        weigth[5*i,0] = -u_r_dot[i,1]
        weigth[5*i,3] = -u_r[i,1]
        # second row
        weigth[5*i+1,1] = -v_r_dot[i,1]
        weigth[5*i+1,4] = -v_r[i,1]
        # third row
        weigth[5*i+2,0] = u_r[i,1]*v_r[i,1]   # X_u_dot
        weigth[5*i+2,1] = -v_r[i,1]*u_r[i,1]  # Y_v_dot 
        weigth[5*i+2,2] = -v_r_dot[i,1]       # N_v_dot
        weigth[5*i+2,5] = -v_r[i,1]           # N_v
        # fourth row
        weigth[5*i+3,6] = np.cos(psi[i,1])
        weigth[5*i+3,7] = -np.sin(psi[i,1])
        # fifth row
        weigth[5*i+4,6] = np.sin(psi[i,1])
        weigth[5*i+4,6] = np.cos(psi[i,1])
        # # sixth row
        # weigth[7*i+5,6] = -x[i,1]
        # weigth[7*i+5,8] = -2*x_dot[i,1]
        # # seventh row
        # weigth[7*i+6,7] = -y[i,1]
        # weigth[7*i+6,9] = -2*y_dot[i,1]


        
        # Build the target vector
        # target = np.array([[F_x+m*(-u_r_dot+r*v_r), F_y-m*(v_r+r*u_r), M_z-I_z*r_dot, x_dot-u_r*np.cos(psi)+v_r*np.sin(psi), y_dot-u_r*np.sin(psi)-v_r*np.cos(psi), x_dot_dot, y_dot_dot, psi_dot_dot]])
        target[5*i,0] = F_x[i,1]-m*u_r_dot[i,1]
        target[5*i+1,0] = F_y[i,1]-m*v_r_dot[i,1]
        target[5*i+2,0] = M_z[i,1]
        target[5*i+3,0] = x_dot[i,1]-u_r[i,1]*np.cos(psi[i,1])+v_r[i,1]*np.sin(psi[i,1])
        target[5*i+4,0] = y_dot[i,1]-u_r[i,1]*np.sin(psi[i,1])-v_r[i,1]*np.cos(psi[i,1])
        # target[7*i+5,0] = x_ddot[i,1]
        # target[7*i+6,0] = y_ddot[i,1]

    


    return weigth, target

def get_matrices_for_ls_r(psi, psi_dot, psi_dot_dot,M_z):
    length = len(psi[:,0]) 
    weigth = np.array(np.zeros((length,4)))
    target = np.zeros((length,1))

    r = psi_dot
    r_dot = psi_dot_dot
    # for i in range(length):               uncomment if only the deccelaration part should be considered
    #     if (psi[i,0] > 1728918351):
    #         new_length = i
    #         break

    for i in range(length):   
        # first row
        weigth[i,0] = r_dot[i,1]
        weigth[i,1] = -r[i,1]

        # second row
        # weigth[2*i+1,2] = -psi[i,1]
        # weigth[2*i+1,3] = -2*psi_dot[i,1]

        # target vector
        target[2*i,0] = M_z[i,1]
        # target[2*i+1,0] = psi_dot_dot[i,1] 

    return weigth, target

def get_matrices_for_ls_fossen_only(u_r,v_r,u_r_dot,v_r_dot,psi,x_dot,y_dot,psi_dot,psi_ddot,F_x,F_y,M_z,m):
    length = len(u_r[:,0])
    weigth = np.array(np.zeros((5*length,12)))
    target = np.zeros((length*5,1))
    r = psi_dot
    r_dot = psi_ddot

    for i in range(length):
        #                         0         1         2        3         4        5     6    7    8     9    10           11
        #                         X_u_dot,  Y_v_dot,  Y_r_dot, N_v_dot,  N_r_dot, X_u,  Y_v, Y_r, N_v,  N_r, u_c,         v_c
        # weight   = np.matrix([ [-u_r_dot, v_r*r,    r**2,    0,        0,      -u_r, 0,   0,   0,    0,   0,           0],
        #                  1     [-u_r*r,   -v_r_dot, -r_dot,  0,        0,      0,    v_r, r,   0,    0,   0,           0],
        #                  2     [u_r*v_r,  -v_r*u_r, -u_r*r,  -v_r_dot, r_dot,  0,    0,   0,   -v_r, -r,  0,           0],
        #                  3     [0,        0,        0,       0,        0,      0,    0,   0,   0,    0,   np.cos(psi), -np.sin(psi)],
        #                  4     [0,        0,        0,       0,        0,      0,    0,   0,   0,    0,   np.sin(psi), np.cos(psi)],
       
        
        # first row
        weigth[5*i,0] = -u_r_dot[i,1]
        weigth[5*i,1] = v_r[i,1]*r[i,1]
        weigth[5*i,2] = (r[i,1])**2
        weigth[5*i,5] = -u_r[i,1]
        # second row
        weigth[5*i+1,0] = -u_r[i,1]*r[i,1]
        weigth[5*i+1,1] = -v_r_dot[i,1]
        weigth[5*i+1,2] = -r_dot[i,1]
        weigth[5*i+1,6] = -v_r[i,1]
        weigth[5*i+1,7] = -r[i,1]
        # third row
        weigth[5*i+2,0] = u_r[i,1]*v_r[i,1]   # X_u_dot
        weigth[5*i+2,1] = -v_r[i,1]*u_r[i,1]  # Y_v_dot 
        weigth[5*i+2,2] = -u_r[i,1]*r[i,1]    # Y_r_dot
        weigth[5*i+2,3] = -v_r_dot[i,1]       # N_v_dot
        weigth[5*i+2,4] = r_dot[i,1]          # N_r_dot
        weigth[5*i+2,8] = -v_r[i,1]           # N_v
        weigth[5*i+2,9] = -r[i,1]             # N_r
        # fourth row
        weigth[5*i+3,10] = np.cos(psi[i,1])
        weigth[5*i+3,11] = -np.sin(psi[i,1])
        # fifth row
        weigth[5*i+4,10] = np.sin(psi[i,1])
        weigth[5*i+4,11] = np.cos(psi[i,1])

        
        # Build the target vector
        # target = np.array([[F_x+m*(-u_r_dot+r*v_r), F_y-m*(v_r+r*u_r), M_z-I_z*r_dot, x_dot-u_r*np.cos(psi)+v_r*np.sin(psi), y_dot-u_r*np.sin(psi)-v_r*np.cos(psi), x_dot_dot, y_dot_dot, psi_dot_dot]])
        target[5*i,0] = F_x[i,1]+m*(-u_r_dot[i,1]+r[i,1]*v_r[i,1])
        target[5*i+1,0] = F_y[i,1]-m*(v_r_dot[i,1]+r[i,1]*u_r[i,1])
        target[5*i+2,0] = M_z[i,1]
        target[5*i+3,0] = x_dot[i,1]-u_r[i,1]*np.cos(psi[i,1])+v_r[i,1]*np.sin(psi[i,1])
        target[5*i+4,0] = y_dot[i,1]-u_r[i,1]*np.sin(psi[i,1])-v_r[i,1]*np.cos(psi[i,1])


    return weigth, target

def get_matrices_for_ls_fossen_only_xy(u_r,v_r,u_r_dot,v_r_dot,psi,x_dot,y_dot,F_x,F_y,M_z,m):

    length = len(u_r[:,0])
    weigth = np.array(np.zeros((5*length,8)))
    target = np.zeros((length*5,1))
    # print(length)
    # print(target.shape)
    
    for i in range(length):
        #                        0          1         2         3     4    5     6            7       
        #                        X_u_dot,   Y_v_dot,  N_v_dot,  X_u,  Y_v, N_v,  u_c,         v_c

        # weight   =       0     [-u_r_dot, 0,        0,        -u_r, 0,   0,    0,           0],
        #                  1     [0,        -v_r_dot, 0,        0,    v_r, 0,    0,           0],
        #                  2     [u_r*v_r,  -v_r*u_r, -v_r_dot, 0,    0,   -v_r, 0,           0],
        #                  3     [0,        0,        0,        0,    0,   0,    np.cos(psi), -np.sin(psi)],
        #                  4     [0,        0,        0,        0,    0,   0,    np.sin(psi), np.cos(psi)],
        
         # first row
        weigth[5*i,0] = -u_r_dot[i,1]
        weigth[5*i,3] = -u_r[i,1]
        # second row
        weigth[5*i+1,1] = -v_r_dot[i,1]
        weigth[5*i+1,4] = -v_r[i,1]
        # third row
        weigth[5*i+2,0] = u_r[i,1]*v_r[i,1]   # X_u_dot
        weigth[5*i+2,1] = -v_r[i,1]*u_r[i,1]  # Y_v_dot 
        weigth[5*i+2,2] = -v_r_dot[i,1]       # N_v_dot
        weigth[5*i+2,5] = -v_r[i,1]           # N_v
        # fourth row
        weigth[5*i+3,6] = np.cos(psi[i,1])
        weigth[5*i+3,7] = -np.sin(psi[i,1])
        # fifth row
        weigth[5*i+4,6] = np.sin(psi[i,1])
        weigth[5*i+4,7] = np.cos(psi[i,1])

        
        # Build the target vector
        # target = np.array([[F_x+m*(-u_r_dot+r*v_r), F_y-m*(v_r+r*u_r), M_z-I_z*r_dot, x_dot-u_r*np.cos(psi)+v_r*np.sin(psi), y_dot-u_r*np.sin(psi)-v_r*np.cos(psi), x_dot_dot, y_dot_dot, psi_dot_dot]])
        target[5*i,0] = F_x[i,1]-m*u_r_dot[i,1]
        target[5*i+1,0] = F_y[i,1]-m*v_r_dot[i,1]
        target[5*i+2,0] = M_z[i,1]
        target[5*i+3,0] = x_dot[i,1]-u_r[i,1]*np.cos(psi[i,1])+v_r[i,1]*np.sin(psi[i,1])
        target[5*i+4,0] = y_dot[i,1]-u_r[i,1]*np.sin(psi[i,1])-v_r[i,1]*np.cos(psi[i,1])

    return weigth, target

def get_matrices_for_ls_fossen_only_r(r,r_dot, M_z):
    length = len(r[:,0]) 
    weigth = np.array(np.zeros((length,2)))
    target = np.zeros((length,1))


    for i in range(length):   
        # first row
        weigth[i,0] = r_dot[i,1]   # N_r_dot
        weigth[i,1] = -r[i,1]      # N_r

        # target vector
        target[i,0] = M_z[i,1]

    return weigth, target

def get_matrices_for_ls_fossen_simplyfied(u_r, v_r, r, F_x, F_y, M_z, m):
    length = len(u_r[:,0])
    number_of_eq = 3
    number_of_params = 6
    weigth = np.zeros((number_of_eq*length,number_of_params))
    target = np.zeros((length*number_of_eq,1))
    gps_counter = 0
    imu_counter = 0
    input_counter = 0
    while u_r[gps_counter,0] < 0.01:
        gps_counter += 1
    while r[imu_counter,0] < 0.01:
        imu_counter += 1
    while (abs(F_x[input_counter,0] - u_r[gps_counter,0]) > 0.08):
        input_counter += 1
    # print(f"input counter: {input_counter}")
    # print(u_r.shape)
    # print(r.shape)
    # print(F_x.shape)
    # print(f"u_r: {u_r[gps_counter+100,0]}")
    # print(f"r: {r[imu_counter+100,0]}")
    # print(f"F_X: {F_x[input_counter+100,0]}")
    # plt.plot(M_z[:,0], M_z[:,1])
    # plt.show()
    
    for i in range(1,length- max(gps_counter,imu_counter,input_counter)):

        ur_dot = (u_r[i+gps_counter,1] - u_r[i+gps_counter-1,1]) / (u_r[i+gps_counter,0] - u_r[i+gps_counter-1,0])
        vr_dot = (v_r[i+gps_counter,1] - v_r[i+gps_counter-1,1]) / (v_r[i+gps_counter,0] - v_r[i+gps_counter-1,0])
        r_dot = (r[i+imu_counter,1] - r[i+imu_counter-1,1]) / (r[i+imu_counter,0] - r[i+imu_counter-1,0])

        # first row
        weigth[number_of_eq*i,0] = -ur_dot   # X_u_dot
        weigth[number_of_eq*i,1] = v_r[i+gps_counter,1]*r[i+imu_counter,1] # Y_v_dot
        weigth[number_of_eq*i,3] = -u_r[i+gps_counter,1]       # X_u

        # second row
        weigth[number_of_eq*i+1,0] = -u_r[i+gps_counter,1]*r[i+imu_counter,1] # X_u_dot
        weigth[number_of_eq*i+1,1] = -vr_dot    # Y_v_dot
        weigth[number_of_eq*i+1,4] = -v_r[i+gps_counter,1]        # Y_v

        # third row
        weigth[number_of_eq*i+2,0] = u_r[i+gps_counter,1]*v_r[i+gps_counter,1]  # X_u_dot
        weigth[number_of_eq*i+2,1] = -v_r[i+gps_counter,1]*u_r[i+gps_counter,1] # Y_v_dot
        weigth[number_of_eq*i+2,2] = r_dot         # N_r_dot
        weigth[number_of_eq*i+2,5] = -r[i+imu_counter,1]            # N_r

        # fourth row
        # weigth[number_of_eq*i+3,6] = 1 # u_c

        # fifth row
        # weigth[number_of_eq*i+4,7] = 1 # v_c

        # sixth row
        # all zeros

        target[number_of_eq*i,0] = F_x[i+input_counter,1] - m*ur_dot + m*v_r[i+gps_counter,1]*r[i+imu_counter,1]
        target[number_of_eq*i+1,0] = F_y[i+input_counter,1] - m*vr_dot - m*u_r[i+gps_counter,1]*r[i+imu_counter,1]
        target[number_of_eq*i+2,0] = M_z[i+input_counter,1]
        # target[number_of_eq*i+3,0] = x_dot[i,1] - u_r[i,1]*np.cos(psi[i,1]) + v_r[i,1]*np.sin(psi[i,1])
        # target[number_of_eq*i+4,0] = y_dot[i,1] - u_r[i,1]*np.sin(psi[i,1]) - v_r[i,1]*np.cos(psi[i,1])
        # target[number_of_eq*i+5,0] = psi_dot[i,1] - r[i,1]
        # break

    # print(weigth[3:,:])
    # print(target[:200,0])
    return weigth, target

def reduce_all_data(x, u_r, v_r, u_r_dot, v_r_dot, psi, psi_dot, psi_dot_dot, F_x, F_y, M_z):
    # get the same size of the measured data
    u_r_reduced = get_same_size(x,u_r)
    v_r_reduced = get_same_size(x,v_r)
    # r_reduced = get_same_size(x,r) => same as psi_dot_reduced

    u_r_dot_reduced = get_same_size(x,u_r_dot)
    v_r_dot_reduced = get_same_size(x,v_r_dot)
    # r_dot_reduced = get_same_size(x,r_dot) => same as psi_dot_dot_reduced

    psi_reduced = get_same_size(x,psi)
    psi_dot_reduced = get_same_size(x,psi_dot)
    psi_dot_dot_reduced = get_same_size(x,psi_dot_dot)

    F_x_reduced = get_same_size(x,F_x)
    F_y_reduced = get_same_size(x,F_y)
    M_z_reduced = get_same_size(x,M_z)
    
    return u_r_reduced, v_r_reduced, u_r_dot_reduced, v_r_dot_reduced, psi_reduced, psi_dot_reduced, psi_dot_dot_reduced, F_x_reduced, F_y_reduced, M_z_reduced

def read_all_data_to_var(file_path,rpy_IMU):

    # IMU               1                   2                   3                   4                       5                          6                         7
    measurement = [['orientation','x'],['orientation','y'],['orientation','z'],['orientation','w'],['angular_velocity','x'],['angular_velocity','y'],['angular_velocity','z']]
    imu_data = read_in_mcap(file_path, measurement, ["/alpine_amanita/bno055/imu"])
    imu_data_arr = np.array(imu_data).T #get_array_form(imu_data)
    ENU_angles, ENU_rot_vel = get_imu_in_body(imu_data_arr,rpy_IMU)     # orientation and angular velocity in body frame
    psi = ENU_angles[:,[0,3]]                                       # heading of the craft in body frame
    r = ENU_rot_vel[:,[0,3]]                                        # angular velocity around local body z-axis

    # GPS position      1             2             3
    measurement = [['latitude'],['longitude'],['altitude']] 
    gps_fix = read_in_mcap(file_path, measurement, ["/alpine_amanita/ublox_gps_node/fix"])
    measurement = [['twist','twist', 'linear', 'x'], ['twist','twist', 'linear', 'y'], ['twist','twist', 'linear', 'z']]
    gps_vel_in_enu = read_in_mcap(file_path, measurement, ["/alpine_amanita/ublox_gps_node/fix_velocity"])

    # Reduce IMU data for the transformation of the GPS data
    imu_data_red = synchronize_imu_gps(np.array(gps_fix).T[:,[0,1]], imu_data_arr, 0.009)


    # GPS in body frame
    gps_fix_enu = GPS_to_ENU(gps_fix)       # returns time, x, y, z
    gps_in_body = convert_enu_to_body(imu_data_red, gps_fix_enu, gps_vel_in_enu, rpy_IMU) # changes the GPS data into the local body frame
    x = gps_in_body[:,[0,1]]                # x-position in ENU frame from GPS
    y = gps_in_body[:,[0,2]]                # y-position in ENU frame from GPS
    x_dot = get_derivative(x)               # change in x-position in ENU frame 
    y_dot = get_derivative(y)               # change in y-position in ENU frame 
    u_r = gps_in_body[:,[0,3]]              # x-velocity in body frame
    v_r = gps_in_body[:,[0,4]]              # y-velocity in body frame
    u_r_dot = get_derivative(u_r)           # change in x-velocity in body frame
    v_r_dot = get_derivative(v_r)           # change in y-velocity in body frame
    

    # Motor commands
    # measurement = [['m1'],['m2'],['m3'],['m4']]
    # motors = read_in_mcap(file_path,measurement,['/alpine_amanita/motor_commands'])
    # F_x, F_y, M_z = get_force_and_moment_from_motors(motors, radius)

    #new method
    measurement = [['linear', 'x'], ['linear', 'y'], ['linear', 'z'], ['angular', 'x'], ['angular', 'y'], ['angular', 'z']]
    motor_twist = read_in_mcap(file_path, measurement, ['/alpine_amanita/motor_twist'])
    F_x_new, F_y_new, M_z_new = get_force_and_moment_from_motors_new(motor_twist, radius)

    F_x_new_red = synchronize_input_gps(np.array(gps_fix).T[:,[0,1]], F_x_new)
    F_y_new_red = synchronize_input_gps(np.array(gps_fix).T[:,[0,1]], F_y_new)
    M_z_new_red = synchronize_input_gps(np.array(gps_fix).T[:,[0,1]], M_z_new)
    # F_x_red = synchronize_input_gps(np.array(gps_fix).T[:,[0,1]], F_x) 
    # F_y_red = synchronize_input_gps(np.array(gps_fix).T[:,[0,1]], F_y) 
    # M_z_red = synchronize_input_gps(np.array(gps_fix).T[:,[0,1]], M_z) 
    # plt.plot(M_z[:,0], M_z[:,1])
    # plt.plot(M_z_new_red[:,0], M_z_new_red[:,1])
    # plt.plot(F_x_new_red[:,0], F_x_new_red[:,1])
    # plt.plot(F_y_new_red[:,0], F_y_new_red[:,1])
    # print(F_x_new[1,0])
    # print(F_x_new_red[1,0])
    # print(F_x_new.shape)
    # print(F_x_new_red.shape)

    

    # reduce size
    psi_red = synchronize_imu_gps(np.array(gps_fix).T[:,[0,1]],psi, 0.009)
    r_red = synchronize_imu_gps(np.array(gps_fix).T[:,[0,1]],r, 0.009)
    psi_dot_red = get_derivative(psi_red)
    r_dot_red = get_derivative(r_red)

    # fig, axs = plt.subplots(3,1, sharex=True)
    # axs[0].plot(F_y[:,0], F_y[:,1], color='g', label='F_y original')
    # axs[0].plot(F_y_red[:,0], F_y_red[:,1], color='r', label='F_y reduced')
    # axs[0].legend()
    # axs[1].plot(F_x[:,0], F_x[:,1], color='g', label='F_x original')
    # axs[1].plot(F_x_red[:,0], F_x_red[:,1], color='r', label='F_x reduced')
    # axs[1].legend()
    # axs[2].plot(M_z[:,0], M_z[:,1], color='g', label='M_z original')
    # axs[2].plot(M_z_red[:,0], M_z_red[:,1], color='r', label='M_z reduced')
    # axs[2].legend()

    # fig, axs = plt.subplots(2,1, sharex=True)
    # axs[0].plot(psi[:,0], psi[:,1], color='g', label='psi original')
    # axs[0].plot(psi_red[:,0], psi_red[:,1], color='r', label='psi reduced')
    # axs[1].plot(r[:,0], r[:,1], color='g', label='r original')
    # axs[1].plot(r_red[:,0], r_red[:,1], color='r', label='r reduced')

    # fig, axs = plt.subplots(2,1, sharex=True)
    # axs[0].plot(u_r[:,0], u_r[:,1], color='g', label='u_r original')
    # # axs[0].plot(psi_red[:,0], psi_red[:,1], color='r', label='psi reduced')
    # axs[1].plot(v_r[:,0], v_r[:,1], color='g', label='v_r original')
    # # axs[1].plot(gps_in_body[:,0], gps_in_body[:,4], label='hallo')
    # # axs[1].plot(r_red[:,0], r_red[:,1], color='r', label='r reduced')
    # axs[1].legend()

    



    # print(F_x[100,:])
    # print(F_x_red[100,:])

    
    # plt.plot(F_x[:,0], F_x[:,1], color='g')
    # plt.plot(F_x_red[:,0], F_x_red[:,1], color='r')
    # plt.show()

    return x, y, x_dot, y_dot, u_r, v_r, u_r_dot, v_r_dot, psi_red, psi_dot_red, r_red, r_dot_red, F_x_new_red, F_y_new_red, M_z_new_red, np.array(gps_vel_in_enu).T

def do_all(file_path,option,rpy_IMU):
    # x, y, x_dot, y_dot, x_dot_dot, y_dot_dot, u_r, v_r, u_r_dot, v_r_dot, psi, psi_dot, psi_dot_dot, F_x, F_y, M_z = read_all_data_to_var(file_path)
    # x, y and its derivatives don't have to be reduced, as the GPS already has the smallest frequency
    # u_r_reduced, v_r_reduced, u_r_dot_reduced, v_r_dot_reduced, psi_reduced, psi_dot_reduced, psi_dot_dot_reduced, F_x_reduced, F_y_reduced, M_z_reduced = reduce_all_data(x, u_r, v_r, u_r_dot, v_r_dot, psi, psi_dot, psi_dot_dot, F_x, F_y, M_z)

    x_dot, y_dot, x_vel, y_vel, x_vel_dot, y_vel_dot, psi, psi_dot, r, r_dot, F_x_red, F_y_red, M_z_red = read_all_data_to_var(file_path,rpy_IMU)

    # get the weight matrices
    if option==1:
        gradient, target = get_matrices_for_ls_all(u_r_reduced,v_r_reduced,u_r_dot_reduced,v_r_dot_reduced,x,y,psi_reduced,x_dot,y_dot,psi_dot_reduced,x_dot_dot,y_dot_dot,psi_dot_dot_reduced,F_x_reduced,F_y_reduced,M_z_reduced,m)
    elif option==2:
        gradient, target = get_matrices_for_ls_xy(u_r_reduced,v_r_reduced,u_r_dot_reduced,v_r_dot_reduced,psi_reduced,x_dot,y_dot,F_x_reduced,F_y_reduced,M_z_reduced,m)
    elif option==3:
        gradient, target = get_matrices_for_ls_r(psi_reduced, psi_dot_reduced, psi_dot_dot_reduced,M_z_reduced)
    elif option==4:
        gradient, target = get_matrices_for_ls_fossen_only(u_r_reduced,v_r_reduced,u_r_dot_reduced,v_r_dot_reduced,psi_reduced,x_dot,y_dot,psi_dot_reduced,psi_dot_dot_reduced,F_x_reduced,F_y_reduced,M_z_reduced,m)
    elif option==5:
        gradient, target = get_matrices_for_ls_fossen_only_xy(u_r_reduced,v_r_reduced,u_r_dot_reduced,v_r_dot_reduced,psi_reduced,x_dot,y_dot,F_x_reduced,F_y_reduced,M_z_reduced,m)
    elif option==6:
        gradient, target = get_matrices_for_ls_fossen_only_r(psi_dot_reduced, psi_dot_dot_reduced, M_z_reduced)
    elif option==7:
        gradient, target = get_matrices_for_ls_fossen_simplyfied(x_vel, y_vel, r, F_x_red, F_y_red, M_z_red, m)

    return gradient,target

def rot_dyn(x,theta,dt):
    
    psi = x[2,0]
    u_r = x[3,0]
    v_r = x[4,0]

    R_psi = np.matrix([[np.cos(psi), -np.sin(psi)],
                       [np.sin(psi),  np.cos(psi)]])
    
    A_cont = R_psi 

    A_d = expm(A_cont * dt) 
    
    xp = np.squeeze(A_d @ np.array([[u_r], [v_r]]) + np.array([[theta[0]], [theta[1]]]))

    return xp

def fossen_x(x,u,theta,dt):
    m = 10.835
    u_r = x[0]

    X_u_dot = theta[0]
    X_u = theta[1]

    Mass = m + X_u_dot

    Mass_inv = 1/Mass

    N = X_u * u_r

    A_cont = -Mass_inv * N
    B_cont = Mass_inv

    A_d = expm(A_cont * dt)
    B_d = 1/A_cont * (A_d - 1) * B_cont

    xp =  A_d * x + B_d * u

    return xp

def fossen_r(x,u,theta,dt):
    m = 10.835
    r = x[0]

    I_comb = theta[0]
    N_r = theta[1]

    Mass = I_comb

    Mass_inv = 1/Mass

    N = N_r * r

    A_cont = -Mass_inv * N
    B_cont = Mass_inv

    A_d = expm(A_cont * dt)
    B_d = 1/A_cont * (A_d - 1) * B_cont

    xp =  A_d * x + B_d * u

    return xp

def dynamics_fossen(x,u,theta,dt):
    m = 10.835
    n_states = 3


    # x 3,1
    # u 3,1
    # theta 4,
    # u[2,0] *= 0.2835
    u_r = x[0,0]
    v_r = x[1,0]
    r = x[2,0]
    
    # fx = u[0,0]
    # fy = u[1,0]
    # mz = u[2,0]

    # x = np.expand_dims(x,1) # 3x1
    # input = np.array([[fx], [fy], [mz]])

    X_u_dot = theta[0] 
    Y_v_dot = theta[0]
    I_comb = theta[1]
    X_u = theta[2]
    Y_v = theta[2]
    N_r = theta[3]


    # Mass matrix
    Mass_matrix = np.matrix([[m+X_u_dot, 0,         0],
                             [0,         m+Y_v_dot, 0],
                             [0,         0,         I_comb]],dtype=np.float64)

    # Inverse of the mass matrix
    Mass_inv = np.linalg.inv(Mass_matrix)


    # Sum of Coriolis and Damping matrix
    N = np.matrix([[-X_u,         -m*r,          Y_v_dot*v_r],  
                   [ m*r,         -Y_v,         -X_u_dot*u_r],  
                   [-Y_v_dot*v_r,  X_u_dot*u_r, -N_r]],dtype=np.float64)
    

    A_cont = -Mass_inv @ N
    B_cont = Mass_inv

    # Discretize A and B
    # A_d = np.eye(A_cont.shape[0],dtype=np.float64) + dt * A_cont 
    # B_d = dt * B_cont
    A_d = expm(A_cont * dt)
    B_d = np.linalg.inv(A_cont) @ (A_d - np.eye(A_cont.shape[0])) @ B_cont
    
    # Calculate new discrete state 
    xp =  np.squeeze(A_d @ x + B_d @ u) 

    return  xp

def objective(theta):

    prediction_error = 0
    regularization = 0
    regularization_weight = 0.0
    delta = 0.5 # bigger => more quadratic
    
    for i in range(len(state)-1):
        # x_k = Jan_state[i].reshape(-1,1)
        # u_k = Jan_input[i].reshape(-1,1)
        # dt = Jan_dt[i]
        # x_k_plus_1_pred = dynamics_fossen(x_k, u_k, theta, dt).reshape(-1,1)
        # x_k_plus_1 = Jan_state_plus[i].reshape(-1,1)
        x_k = state[i].reshape(-1,1)
        # x_k = x_k[0] # only u_r
        # x_k = x_k[2] # only r
        u_k = u[i].reshape(-1,1)
        # u_k = u_k[0] # only F_x
        # u_k = u_k[2] # only M_z
        dt = dt_array[i]
        x_k_plus_1_pred = dynamics_fossen(x_k, u_k, theta, dt).reshape(-1,1)
        # x_k_plus_1_pred = fossen_x(x_k, u_k, theta, dt).reshape(-1,1) # only fossen for velocity in x
        # x_k_plus_1_pred = fossen_r(x_k, u_k, theta, dt).reshape(-1,1) # only fossen for moment around z
        x_k_plus_1 = state[i+1].reshape(-1,1)
        # x_k_plus_1 = x_k_plus_1[0,0] # only u_r
        # x_k_plus_1 = x_k_plus_1[2,0] # only r

        # Squared error
        # current_error = np.linalg.norm(x_k_plus_1 - x_k_plus_1_pred)**2

        # Huber loss
        residuals = x_k_plus_1 - x_k_plus_1_pred
        huber = np.sum(np.where(abs(residuals) <= delta, 
                                0.5 * np.square(residuals),  # Quadratic region
                                delta * (np.abs(residuals) - 0.5 * delta)  # Linear region
        ))
        prediction_error += huber

    # Adding a regularization term for penalysing the values of the parameter
    regularization = np.sum(np.square(theta))
    total_error = prediction_error + regularization_weight * regularization
    
    return total_error

def objective_rot(theta):

    prediction_error = 0
    regularization = 0
    regularization_weight = 0.01
    delta = 1.5 # bigger => more quadratic

    for i in range(len(state)-1):
        x_k = state[i].reshape(-1,1)
        dt = dt_array[i]
        x_k_plus_1_pred = rot_dyn(x_k, theta, dt).reshape(-1,1)
        x_k_plus_1 = state[i+1].reshape(-1,1)
        x_k_plus_1 = x_k_plus_1[:2,0]
        # print(x_k_plus_1)
        
        # Squared error
        # current_error = np.linalg.norm(x_k_plus_1 - x_k_plus_1_pred)**2

        # Huber loss
        residuals = x_k_plus_1 - x_k_plus_1_pred
        huber = np.sum(np.where(abs(residuals) <= delta, 
                                0.5 * np.square(residuals),  # Quadratic region
                                delta * (np.abs(residuals) - 0.5 * delta)  # Linear region
        ))
        prediction_error += huber

    total_error = prediction_error

    return total_error


# rigid body
radius = 0.2835 # in m
m = 10.835 # in kg


# file_path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Measurements\rosbags_2024_10_14_mythenquai\rosbags_2024_10_14_mythenquai\rosbag2_2024_10_14-14_46_56_mythenquai_rectangle_1\rosbag2_2024_10_14-14_46_56_0.mcap")
# gradient1, target1 = do_all(file_path,2) # rectangle 1
# file_path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Measurements\rosbags_2024_10_14_mythenquai\rosbags_2024_10_14_mythenquai\rosbag2_2024_10_14-14_50_02_mythenquai_rectangle_2\rosbag2_2024_10_14-14_50_02_0.mcap")
# gradient2, target2 = do_all(file_path,2) # rectangle 2
# file_path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Measurements\rosbags_2024_10_14_mythenquai\rosbags_2024_10_14_mythenquai\rosbag2_2024_10_14-15_38_42_mythenquai_rectange_3\rosbag2_2024_10_14-15_38_42_0.mcap")
# gradient3, target3 = do_all(file_path,2) # rectangle 3
# file_path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Measurements\rosbags_2024_10_14_mythenquai\rosbags_2024_10_14_mythenquai\rosbag2_2024_10_14-14_56_21_mythenquai_straight_line_1\rosbag2_2024_10_14-14_56_21_0.mcap")
# gradient4, target4 = do_all(file_path,2) # straight line 1
# file_path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Measurements\rosbags_2024_10_14_mythenquai\rosbags_2024_10_14_mythenquai\rosbag2_2024_10_14-15_00_49_mythenquai_straight_line_2\rosbag2_2024_10_14-15_00_49_0.mcap")
# gradient5, target5 = do_all(file_path,2) # straight line 2
# file_path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Measurements\rosbags_2024_10_14_mythenquai\rosbags_2024_10_14_mythenquai\rosbag2_2024_10_14-15_02_23_mythenquai_straight_line_3\rosbag2_2024_10_14-15_02_23_0.mcap")
# gradient6, target6 = do_all(file_path,2) # straight line 3
# file_path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Measurements\rosbags_2024_10_14_mythenquai\rosbags_2024_10_14_mythenquai\rosbag2_2024_10_14-15_04_54_mythenquai_turn_counter_clockwise\rosbag2_2024_10_14-15_04_54_0.mcap")
# gradient7, target7 = do_all(file_path,3) # turn counter clockwise
# file_path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Measurements\rosbags_2024_10_14_mythenquai\rosbags_2024_10_14_mythenquai\rosbag2_2024_10_14-15_05_30_mythenquai_turn_clockwise\rosbag2_2024_10_14-15_05_30_0.mcap")
# gradient8, target8 = do_all(file_path,3) # turn clockwise

# x, y, x_dot, y_dot, x_dot_dot, y_dot_dot, u_r, v_r, u_r_dot, v_r_dot, psi, psi_dot, psi_dot_dot, F_x, F_y, M_z = read_all_data_to_var(file_path)
# u_r_reduced, v_r_reduced, u_r_dot_reduced, v_r_dot_reduced, psi_reduced, psi_dot_reduced, psi_dot_dot_reduced, F_x_reduced, F_y_reduced, M_z_reduced = reduce_all_data(x, u_r, v_r, u_r_dot, v_r_dot, psi, psi_dot, psi_dot_dot, F_x, F_y, M_z)
# gradient, target = get_matrices_for_ls_r(psi_reduced, psi_dot_reduced, psi_dot_dot_reduced, M_z_reduced)

# gradient1,gradient2,gradient3,gradient4,gradient5,gradient6,gradient7,gradient8
# target1,target2,target3,target4,target5,target6,target7,target8
# gradient = np.vstack([gradient1,gradient2,gradient3,gradient4,gradient5,gradient6])
# target = np.vstack([target1,target2,target3,target4,target5,target6])

rpy_IMU = np.array([0.0, 0.0, -np.pi/2]) # bno055 0, 0, -np.pi/2 xsense: np.pi, 0.0, -0.5*np.pi
file_path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Measurements\rosbags_15_11_24_xsense_bno\rosbags_15_11_24_xsense_bno\7.1_random_movement\rosbag2_2024_11_15-16_10_29_0.mcap")
# gradient, target = do_all(file_path,7,rpy_IMU) 

################################################################################
########## Options => 1:all, 2:xy, 3:r, 4:fossen,all, 5:fossen,xy, 6:fossen,r 7:fossen,simpflyfied
################################################################################

# x, y, x_dot, y_dot, x_dot_dot, y_dot_dot, u_r, v_r, u_r_dot, v_r_dot, psi, psi_dot, psi_dot_dot, F_x, F_y, M_z = read_all_data_to_var(file_path)
# u_r_reduced, v_r_reduced, u_r_dot_reduced, v_r_dot_reduced, psi_reduced, psi_dot_reduced, psi_dot_dot_reduced, F_x_reduced, F_y_reduced, M_z_reduced = reduce_all_data(x, u_r, v_r, u_r_dot, v_r_dot, psi, psi_dot, psi_dot_dot, F_x, F_y, M_z)
# plt.plot(F_x[:,0], F_y[:,1])
# plt.show()

###################################
### Perform linear least square ###
###################################


# parameter vector => containing all unknowns
# parameter = ['X_u_dot', 'Y_v_dot', 'Y_r_dot', 'N_v_dot', 'N_r_dot', 'X_u', 'Y_v', 'Y_r', 'N_v', 'N_r', 'omega_01', 'omega_02', 'omega_03', 'zeta_1', 'zeta_2', 'zeta_3', 'u_c', 'v_c']
# parameter_xy = ['X_u_dot', 'Y_v_dot', 'N_v_dot', 'X_u', 'Y_v', 'N_v', 'omega_01', 'omega_02', 'zeta_1', 'zeta_2', 'u_c', 'v_c']
# parameter_r = ['N_r_dot', 'N_r', 'omega_03', 'zeta_3']
# parameter_fossen = ['X_u_dot', 'Y_v_dot', 'Y_r_dot', 'N_v_dot', 'N_r_dot', 'X_u', 'Y_v', 'Y_r', 'N_v', 'N_r', 'u_c', 'v_c']
# parameter_fossen_simplyfied = ['X_u_dot', 'Y_v_dot', 'N_r_dot', 'X_u', 'Y_v', 'N_r']#, 'u_c', 'v_c']
# parameter_fossen_xy = ['X_u_dot', 'Y_v_dot', 'N_v_dot', 'X_u', 'Y_v', 'N_v', 'u_c', 'v_c']
# parameter_fossen_r = ['N_r_dot','N_r']


# model = LinearRegression(fit_intercept=False)
# reg = model.fit(gradient,target)
# coefficient = reg.coef_
# print(coefficient[0][5] / coefficient[0][2])


x, y, x_dot, y_dot, u_r, v_r, u_r_dot, v_r_dot, psi, psi_dot, r, r_dot, F_x, F_y, M_z, GPS_vel = read_all_data_to_var(file_path,rpy_IMU)

# Jan_state = np.load(Path(r"C:\Users\safre\Downloads\state_data 1.npy"))
# Jan_state_plus = np.load(Path(r"C:\Users\safre\Downloads\state_new_data.npy"))
# Jan_input = np.load(Path(r"C:\Users\safre\Downloads\control_data.npy"))
# Jan_dt = np.load(Path(r"C:\Users\safre\Downloads\dt_data.npy"))
# fig, axs = plt.subplots(3,1, sharex=True)
# axs[0].plot(np.linspace(0,len(Jan_input[:,0]), len(Jan_input[:,0])), Jan_input[:,0])
# axs[0].plot(np.linspace(0,len(F_x[:,1]), len(F_x[:,1])), F_x[:,1])
# axs[1].plot(np.linspace(0,len(Jan_input[:,1]), len(Jan_input[:,1])), Jan_input[:,1])
# axs[1].plot(np.linspace(0,len(F_y[:,1]), len(F_y[:,1])), F_y[:,1])
# axs[2].plot(np.linspace(0,len(Jan_input[:,2]), len(Jan_input[:,2])), Jan_input[:,2])
# axs[2].plot(np.linspace(0,len(M_z[:,1]), len(M_z[:,1])), M_z[:,1])
# plt.show()


# fig, axs = plt.subplots(6,1, sharex=True)
# axs[0].plot(np.linspace(0, len(x[:,1]), len(x[:,1])), x[:,1])
# axs[1].plot(np.linspace(0, len(y[:,1]), len(y[:,1])), y[:,1])
# axs[2].plot(np.linspace(0, len(psi[:,1]), len(psi[:,1])), np.degrees(psi[:,1]))
# axs[3].plot(np.linspace(0, len(u_r[:,1]), len(u_r[:,1])), u_r[:,1])
# # axs[3].plot(np.linspace(0, len(GPS_vel[:,1]), len(GPS_vel[:,1])), GPS_vel[:,1])
# axs[3].plot(np.linspace(0, len(Jan_state[:,0]), len(Jan_state[:,0])), Jan_state[:,0])
# axs[4].plot(np.linspace(0, len(v_r[:,1]), len(v_r[:,1])), v_r[:,1])
# # axs[4].plot(np.linspace(0, len(GPS_vel[:,2]), len(GPS_vel[:,2])), GPS_vel[:,2])
# axs[4].plot(np.linspace(0, len(Jan_state[:,1]), len(Jan_state[:,1])), Jan_state[:,1])
# axs[5].plot(np.linspace(0, len(r[:,1]), len(r[:,1])), r[:,1])
# axs[5].plot(np.linspace(0, len(Jan_state[:,2]), len(Jan_state[:,2])), Jan_state[:,2])
# plt.show()


# print(u_r.shape)
# print(r.shape)
# print(F_x.shape)
# print(u_r[0,0])
# print(r[0,0])
# print(F_x[0,0])
# print(F_x[1,0])

u = np.array([F_x[:,1], F_y[:,1], M_z[:,1]]).T  
state = np.array([u_r[:,1], v_r[:,1], r[:,1]]).T
dt_array = np.zeros((len(state[:,0])))
dt_array[0] = 0.0
for i in range(1,len(dt_array)):
    dt_array[i] = u_r[i,0] - u_r[i-1,0]



# print(u_r.shape)
# print(x.shape)
# print(psi.shape)
# print(u_r[50,0])
# print(x[50,0])
# print(psi[50,0])
# print(F_x[1,0])

# fig, axs = plt.subplots(2,1, sharex=True)
# axs[0].plot(psi[:,0], psi[:,1])
# axs[1].plot(x[:,0], x[:,1])
# plt.show()

# state = np.array([x[:,1], y[:,1], psi[:,1], u_r[:,1], v_r[:,1]]).T
# dt_array = np.zeros((len(state[:,0])))
# dt_array[0] = 0.0
# for i in range(1,len(dt_array)):
#     dt_array[i] = u_r[i,0] - u_r[i-1,0]


# target = np.squeeze(target)
# ub = np.array([-0, -0, np.inf, 0,0,0])
# lb = np.array([-np.inf, -np.inf, 0.7, -np.inf, -np.inf, -np.inf])
# res = lsq_linear(A=gradient,b=target,bounds=(lb, ub), max_iter=1000, lsq_solver='lsmr')
# print(res)
# bars = plt.bar(np.arange(len(res.x)), res.x, tick_label=parameter_fossen_simplyfied)
# plt.bar_label(bars)
# plt.show()
# Bounds = ((-np.inf, 0), (-np.inf, 0), (0.7, np.inf), (-np.inf, 0), (-np.inf, 0), (-np.inf, 1),)
# x0 = np.array([-30, -30, 0.7, -110, -100, 0.6])
# x0 = np.array([1, 1, 0.7, -20, -20, -2])
x0 = np.array([1.0, 0.7, -70.0, -2.0])
# x0 = np.array([-1, 0.7]) # only u_r
# x0 = np.array([0.7, -20]) # only r
options = {'xatol': 1e-8, 'maxiter': 2000, 'fatol': 1e-8}
res = minimize(objective, x0, method='nelder-mead', options=options) 

# x0 = np.array([0.0, 0.0])
# res = minimize(objective_rot, x0, method='nelder-mead', options=options) 

print(res.x)

# print(f"the coefficients are: {coefficient}")



# df_out = pd.DataFrame(gradient)
# path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Code\gradient_turns_only.csv")
# df_out.to_csv(path, index=False, header=False)
# df_out = pd.DataFrame(target)
# path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Code\target_turns_only.csv")
# df_out.to_csv(path, index=False, header=False)
# print("finished")


########################################################################################
# Uncomment if you want to save the coefficients
########################################################################################
df_out = pd.DataFrame(res.x)
Save_path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Code\coef\coef.csv")
df_out.to_csv(Save_path, index=False)




# same with np.linalg
# reg, residuals, rank, s = np.linalg.lstsq(gradient, target, rcond=None)
# print("Solution for the unknown parameters:", reg)







 