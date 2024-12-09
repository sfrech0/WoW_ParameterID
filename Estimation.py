import numpy as np
import pandas as pd
import math
import pymap3d
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.linalg import expm
from filterpy.kalman import ExtendedKalmanFilter




def read_in_mcap(file_path):
    # open the file from the given path
    reader = make_reader(open(file_path, "rb"), decoder_factories=[DecoderFactory()])

    # IMU               1                       2                   3               4                       5                       6                           7                       8                           9
    measurement = [['orientation','x'],['orientation','y'],['orientation','z'],['orientation','w'],['angular_velocity','x'],['angular_velocity','y'],['angular_velocity','z'],['linear_acceleration','x'],['linear_acceleration','y']]
    number_of_measurements = len(measurement)
    imu = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        imu[i-1] = []

    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=["/alpine_amanita/bno055/imu"]):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        imu[0].append(time)
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

            imu[i].append(tmp)

    # GPS 
    measurement = [['longitude'],['latitude'],['altitude']] 
    number_of_measurements = len(measurement)
    gps = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        gps[i-1] = []

    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=["/alpine_amanita/ublox_gps_node/fix"]):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        gps[0].append(time)
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

            gps[i].append(tmp)

    # Motor values
    measurement = [['m1'],['m2'],['m3'],['m4']] 
    number_of_measurements = len(measurement)
    motor = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        motor[i-1] = []

    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=["/alpine_amanita/motor_commands"]):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        motor[0].append(time)
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

            motor[i].append(tmp)

    return imu, gps, motor

def read_in_covariance(path):
    reader = make_reader(open(path, "rb"), decoder_factories=[DecoderFactory()])

    # IMU Covariance
    measurement = [['orientation_covariance'],['angular_velocity_covariance'],['linear_acceleration_covariance']]
    number_of_measurements = len(measurement)
    imu_cov = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        imu_cov[i-1] = []

    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=["/alpine_amanita/bno055/imu"]):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        imu_cov[0].append(time)
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

            imu_cov[i].append(tmp)

    # GPS Position Covariance
    measurement = [['position_covariance']] 
    number_of_measurements = len(measurement)
    gps_cov = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        gps_cov[i-1] = []

    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=["/alpine_amanita/ublox_gps_node/fix"]):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        gps_cov[0].append(time)
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

            gps_cov[i].append(tmp)
    
    # GPS Velocity Covariance
    measurement = [['twist','twist','covariance']] 
    number_of_measurements = len(measurement)
    gps_vel_cov = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        gps_vel_cov[i-1] = []

    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=["/alpine_amanita/ublox_gps_node/fix_veloctiy"]):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        gps_vel_cov[0].append(time)
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

            gps_vel_cov[i].append(tmp)

    return imu_cov, gps_cov, gps_vel_cov

def read_in_syn_data(path):
    syn_data_array = pd.read_csv(path).values
    length = len(syn_data_array[:,0])
    syn_data = [(syn_data_array[0,0], [syn_data_array[0,1], syn_data_array[0,2], syn_data_array[0,3], syn_data_array[0,4], syn_data_array[0,5], syn_data_array[0,6]])]
    for i in range(1,length):
        tmp = (syn_data_array[i,0], [syn_data_array[i,1], syn_data_array[i,2], syn_data_array[i,3], syn_data_array[i,4], syn_data_array[i,5], syn_data_array[i,6]])
        syn_data.append(tmp)

    return syn_data

def quat_to_euler(imu_data):
    length = len(imu_data[0])
    euler_angles_ENU = np.zeros((length,4))
    euler_angles_NED = np.zeros((length,4))
    quats = np.zeros((length,5))
    for i in range(length):
        time = imu_data[0][i]
        euler_angles_ENU[i,0] = time
        euler_angles_NED[i,0] = time
        x_quat = imu_data[1][i]
        y_quat = imu_data[2][i]
        z_quat = imu_data[3][i]
        w_quat = imu_data[4][i]
        quats[i,0] = time
        quats[i,1] = x_quat
        quats[i,2] = y_quat
        quats[i,3] = z_quat
        quats[i,4] = w_quat
        r = Rotation.from_quat([x_quat, y_quat, z_quat, w_quat])
        a,b,c = r.as_euler('xyz')
        euler_angles_ENU[i,1] = a
        euler_angles_ENU[i,2] = b
        euler_angles_ENU[i,3] = c
        euler_angles_NED[i,1] = a
        euler_angles_NED[i,2] = -b
        euler_angles_NED[i,3] = np.pi/2 - c

    return euler_angles_ENU, euler_angles_NED, quats

def extract_imu_data(file_path):
    imu, gps, motor = read_in_mcap(file_path)
    euler_ENU, euler_NED, quats = quat_to_euler(imu)

    length = len(imu[0])
    acc_x = np.zeros((length,2))
    acc_y = np.zeros((length,2))
    roll_rate = np.zeros((length,2))
    pitch_rate = np.zeros((length,2))
    yaw_rate = np.zeros((length,2))
    for i in range(length):
        time = imu[0][i]
        acc_x[i,0] = time
        acc_y[i,0] = time
        roll_rate[i,0] = time
        pitch_rate[i,0] = time
        yaw_rate[i,0] = time
        acc_x[i,1] = imu[8][i]
        acc_y[i,1] = imu[9][i]
        roll_rate[i,1] = imu[5][i]
        pitch_rate[i,1] = imu[6][i]
        yaw_rate[i,1] = imu[7][i] 

    return acc_x, acc_y, roll_rate, pitch_rate, yaw_rate, euler_ENU, euler_NED, quats

def extract_gps_data(file_path):
    imu, gps, motor = read_in_mcap(file_path)
    lat_0 = gps[2][0]
    lon_0 = gps[1][0]
    h_0 = gps[3][0]

    length = len(gps[0])
    ENU_coord = np.zeros((length,4))
    for i in range(length):
        ENU_coord[i,0] = gps[0][i] #        lat        lon       alt
        e,n,u = pymap3d.enu.geodetic2enu(gps[2][i], gps[1][i], gps[3][i], lat_0, lon_0, h_0)
        ENU_coord[i,1] = e
        ENU_coord[i,2] = n
        ENU_coord[i,3] = u

    return ENU_coord

def extract_april_tags(file_path): # returns pos => time,x,y,z and orient => time, angle around z in rad
    """
    Returns the array containing the time and the x,y,z position in ENU. 
    Also return the orientation around the z axis from the ENU frame
    """

    reader = make_reader(open(file_path, "rb"), decoder_factories=[DecoderFactory()])

    measurement = [['latitude'],['longitude'],['altitude']]
    number_of_measurements = len(measurement)
    geodetic = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        geodetic[i-1] = []

    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=["/gps_pos_tags"]):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        geodetic[0].append(time)
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

            geodetic[i].append(tmp)

    lat_0 = geodetic[1][0]
    lon_0 = geodetic[2][0]
    h_0 = geodetic[3][0]

    length = len(geodetic[0])
    pos = np.zeros((length,4))
    for i in range(length):
        pos[i,0] = geodetic[0][i] #        lat             lon             alt
        e,n,u = pymap3d.enu.geodetic2enu(geodetic[1][i], geodetic[2][i], geodetic[3][i], lat_0, lon_0, h_0)
        pos[i,1] = e
        pos[i,2] = n
        pos[i,3] = u

    measurement = [['orientation','x'],['orientation','y'],['orientation','z'],['orientation','w']]
    number_of_measurements = len(measurement)
    heading = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        heading[i-1] = []

    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=["/heading_from_tags"]):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        heading[0].append(time)
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

            heading[i].append(tmp)

    orient_ENU, orient_NED, quats = quat_to_euler(heading)

    return pos, orient_ENU 

def extract_gps_vel(path):

    # returns an length x 3 array with time, x_vel, y_vel as columns

    reader = make_reader(open(path, "rb"), decoder_factories=[DecoderFactory()])

    # GPS velocity
    measurement = [['twist', 'twist', 'linear', 'x'], ['twist', 'twist', 'linear', 'y'], ['twist', 'twist', 'linear', 'z']] 
    number_of_measurements = len(measurement)
    gps_vel = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        gps_vel[i-1] = []

    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=["/alpine_amanita/ublox_gps_node/fix_velocity"]):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        gps_vel[0].append(time)
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

            gps_vel[i].append(tmp)
    
    gps_velocity = np.zeros((len(gps_vel[0]),4))
    for i in range(len(gps_vel[0])):
        gps_velocity[i,0] = gps_vel[0][i] # time
        gps_velocity[i,1] = gps_vel[1][i] # x_vel
        gps_velocity[i,2] = gps_vel[2][i] # y_vel
        gps_velocity[i,3] = gps_vel[3][i] # z_vel

    return gps_velocity

def extract_mag_data(file_path): # returns magnetic field strenght in x,y,z as array [time, x, y, z]
    
    reader = make_reader(open(file_path, "rb"), decoder_factories=[DecoderFactory()])

    # Magnetic data
    measurement = [['magnetic_field', 'x'], ['magnetic_field', 'y'], ['magnetic_field', 'z']] 
    number_of_measurements = len(measurement)
    mag = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        mag[i-1] = []

    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=["/alpine_amanita/bno055/mag"]):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        mag[0].append(time)
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

            mag[i].append(tmp)

    mag_data = np.zeros((len(mag[0]),4))
    for i in range(len(mag[0])):
        mag_data[i,0] = mag[0][i]
        mag_data[i,1] = mag[1][i]
        mag_data[i,2] = mag[2][i]
        mag_data[i,3] = mag[3][i]

    return mag_data

def get_force(m):
    if m>0.13: # 0.13 0.1868861121817384
        f = (m-0.13)/0.6437048*65 # 0.6437048
    elif m<(-0.13):
        f = (m+0.13)/0.6437048*65
    else:
        f = 0.0
    
    return f

def extract_motor_values(file_path,radius):
    imu, gps, motor = read_in_mcap(file_path)
    length = len(motor[0])
    u = np.zeros((length,4))
    force_x = np.zeros((length,2))
    force_y = np.zeros((length,2))
    moment_z = np.zeros((length,2))
    # orient = np.zeros((length,2))

    for i in range(length):
        time = motor[0][i]
        u[i,0] = time
        force_x[i,0] = time
        force_y[i,0] = time
        moment_z[i,0] = time

        f1 = get_force(motor[1][i])
        f2 = get_force(motor[2][i])
        f3 = get_force(motor[3][i])
        f4 = get_force(motor[4][i])

        # F_i = ((m_i -  0.13)/0.6437048) * 65 N
        fx = (f2 - f4) #/ 15
        fy = (f3 - f1) #/ 15

        mz = (f1 + f2 + f3 + f4)*radius

        force_x[i,1] = fx
        u[i,1] = fx
        force_y[i,1] = fy
        u[i,2] = fy
        moment_z[i,1] = mz
        u[i,3] = mz
    #     orient[i,0] = time
    #     orient[i,1] = math.atan2(fy,fx)


    return force_x, force_y, moment_z, u

def extract_motor_values_new(path,radius):

    reader = make_reader(open(path, "rb"), decoder_factories=[DecoderFactory()])

    # Motor values
    measurement = [['linear', 'x'], ['linear', 'y'], ['linear', 'z'], ['angular', 'x'], ['angular', 'y'], ['angular', 'z']] 
    number_of_measurements = len(measurement)
    motor = [None]*(number_of_measurements + 1)
    for i in range(number_of_measurements + 1):
        motor[i-1] = []

    for schema, channel, message, dec_msg in reader.iter_decoded_messages(topics=["/alpine_amanita/motor_twist"]):
        # read the time stamp
        time = message.log_time / 1000000000 # to get sek from nanosek
        motor[0].append(time)
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

            motor[i].append(tmp)

    length = len(motor[0])
    F_x = np.zeros((length,2))
    F_y = np.zeros((length,2))
    M_z = np.zeros((length,2))
    u = np.zeros((length,4))
    for i in range(length):
        tmp = motor[0][i]
        F_x[i,0] = tmp
        F_y[i,0] = tmp
        M_z[i,0] = tmp
        u[i,0] = tmp
        F_x[i,1] = 65*2*motor[1][i]
        F_y[i,1] = 65*2*motor[2][i]
        M_z[i,1] = 65*radius*motor[6][i]
        u[i,1] = 65*2*motor[1][i]
        u[i,2] = 65*2*motor[2][i]
        u[i,3] = 65*radius*motor[6][i]


    return F_x, F_y, M_z, u

def extract_covariance(path):
    imu_cov, gps_cov, gps_vel_cov = read_in_covariance(path)
    length_imu = len(imu_cov[0])
    length_gps = len(gps_cov[0])
    length_gps_vel = len(gps_vel_cov[0])
    orientation = np.zeros((length_imu,4))
    angular_vel = np.zeros((length_imu,4))
    linear_acc = np.zeros((length_imu,4))
    pos = np.zeros((length_gps,4))
    linear_vel = np.zeros((length_gps_vel,4))

    # IMU data
    for i in range(length_imu):
        time = imu_cov[0][i]
        orientation[i,0] = time
        angular_vel[i,0] = time
        linear_acc[i,0] = time
        orientation[i,1] = imu_cov[1][i][0]
        orientation[i,2] = imu_cov[1][i][4]
        orientation[i,3] = imu_cov[1][i][8]
        angular_vel[i,1] = imu_cov[2][i][0]
        angular_vel[i,2] = imu_cov[2][i][4]
        angular_vel[i,3] = imu_cov[2][i][8]
        linear_acc[i,1] = imu_cov[3][i][0]
        linear_acc[i,2] = imu_cov[3][i][4]
        linear_acc[i,3] = imu_cov[3][i][8]
    
    # GPS data
    # lon_0 = gps_cov[1][0][0]
    # lat_0 = gps_cov[1][0][4]
    # h_0 = gps_cov[1][0][8]

    for i in range(length_gps):
        pos[i,0] = gps_cov[0][i] #                lat            lon            alt
        # e,n,u = pymap3d.enu.geodetic2enu(gps_cov[1][i][4], gps_cov[1][i][0], gps_cov[1][i][8], lat_0, lon_0, h_0)
        # pos[i,1] = e
        # pos[i,2] = n
        # pos[i,3] = u
        pos[i,1] = gps_cov[1][i][0]
        pos[i,2] = gps_cov[1][i][4]
        pos[i,3] = gps_cov[1][i][8]

    # GPS velocity
    for i in range(length_gps_vel):
        linear_vel[i,0] = gps_vel_cov[0][i]
        linear_vel[i,1] = gps_vel_cov[1][i][0] 
        linear_vel[i,2] = gps_vel_cov[1][i][7]
        linear_vel[i,3] = gps_vel_cov[1][i][14]
        # linear_vel[i,4] = gps_vel_cov[1][i][21]
        # linear_vel[i,5] = gps_vel_cov[1][i][28]
        # linear_vel[i,6] = gps_vel_cov[1][i][35]



    return orientation, angular_vel, linear_acc, pos, linear_vel

def get_integral(data):
    length = len(data[:,0])
    integral = np.zeros((length,2))
    integral[0,0] = data[0,0]
    for i in range(1,length):
        time = data[i,0]
        integral[i,0] = time
        integral[i,1] = integral[i-1,1] + data[i,1]*(time - data[i-1,0])

    return integral

def AJacobian(x,dt):
    psi = x[2]
    u_r = x[3]
    v_r = x[4]
    r = x[5]

    # compute the derivative of the equation of motion w.r.t. ur, vr and r
    derv = np.matrix([[X_u,                                -m*r+Y_v_dot*r,               -m*v_r+Y_v_dot*v_r+2*Y_r_dot*r],
                      [m*r-X_u_dot*r,                      -Y_v,                          m*u_r-X_u_dot*u_r-Y_r],
                      [-Y_v_dot*v_r-Y_r_dot*r+X_u_dot*v_r, -Y_v_dot*u_r+X_u_dot*u_r-N_v, -Y_r_dot*u_r-N_r]])
    
    A_eq = -np.matmul(M_inv,derv) # 3x3
    A_eq = np.hstack((np.zeros((3,3)), A_eq)) # 3x6

    # compute the derivative of the rotation part
    s = np.sin(psi)
    c = np.cos(psi)
    A_r = np.matrix([[0, 0, -(u_r+u_c)*s - (v_r+v_c)*c, c, -s, 0],
                     [0, 0,  (u_r+u_c)*c - (v_r+v_c)*s, s,  c, 0],
                     [0, 0,                          0, 0,  0, 1]])
    
    A = np.vstack((A_eq,A_r)) # 6x6

    # Discretize the transition matrix
    A = np.eye(6) + dt * A
    
    return A
    
def AJacobian_fossen(x,dt):
    psi = x[2]
    u_r = x[3]
    v_r = x[4]
    r = x[5]

    # Mass matrix
    M_sim = np.matrix([[m+X_u_dot, 0,         0],
                       [0,         m+Y_v_dot, 0],
                       [0,         0,         N_r_dot]])
    M_inv_sim = np.linalg.inv(M_sim)

    # compute the derivative of the equation of motion w.r.t. ur, vr and r
    derv = np.matrix([[-X_u,                      Y_v_dot*r-m*r,                Y_v_dot*v_r-m*v_r],   # -m*r in 0,1 and -m*v_r in 0,2
                      [-X_u_dot*r+m*r,               -Y_v,                     -X_u_dot*u_r+m*u_r],   #  +m*r in 1,0 and  +m*u_r in 1,2
                      [-Y_v_dot*v_r+X_u_dot*v_r, -Y_v_dot*u_r+X_u_dot*u_r, -N_r]])
    
    # derv = np.matrix([[-X_u, -m*r,  -m*v_r],   
    #                   [ m*r, -Y_v,   m*u_r],   
    #                   [ 0,    0,    -N_r]])
    
    A_eq = np.hstack((np.zeros((3,3)), derv)) # 3x6
    A_eq_o_m = -np.matmul(M_inv_sim,A_eq) # 3x6
    

    # compute the derivative of the rotation part
    s = np.sin(psi)
    c = np.cos(psi)
    A_r = np.matrix([[0, 0, -u_r*s -v_r*c, c, -s, 0],
                     [0, 0,  u_r*c -v_r*s, s,  c, 0],
                     [0, 0,  0,            0,  0, 1]])
    
    
    # Transition matrix F
    F = np.vstack((A_r,A_eq_o_m)) # 6x6

    # Discretize the transition matrix
    A = np.eye(6) + dt * F

    return A

def Jacobian_focus_group(x,dt):
    psi = x[2]
    u_r = x[3]
    v_r = x[4]
    r = x[5]

    X_u_dot = -3.3
    Y_v_dot = -3.3
    N_r_dot = 2.0

    X_u = -10
    Y_v = -10
    N_r = 2 

    # Mass matrix
    M_sim = np.matrix([[m-X_u_dot, 0,         0],
                       [0,         m-Y_v_dot, 0],
                       [0,         0,         N_r_dot]])
    M_inv_sim = np.linalg.inv(M_sim)

    # compute the derivative of the equation of motion w.r.t. ur, vr and r
    derv = np.matrix([[ X_u, -m*r,  -m*v_r],   
                      [ m*r,  Y_v,   m*u_r],   
                      [ 0,    0,     N_r]])
    
    A_eq = np.hstack((np.zeros((3,3)), derv)) # 3x6
    A_eq_o_m = -np.matmul(M_inv_sim,A_eq) # 3x6
    

    # compute the derivative of the rotation part
    s = np.sin(psi)
    c = np.cos(psi)
    A_r = np.matrix([[0, 0, -u_r*s -v_r*c, c, -s, 0],
                     [0, 0,  u_r*c -v_r*s, s,  c, 0],
                     [0, 0,  0,            0,  0, 1]])
    
    
    # Transition matrix F
    F = np.vstack((A_r,A_eq_o_m)) # 6x6

    # Discretize the transition matrix
    A = np.eye(6) + dt * F

    return A

def Jacobian_simple_model(x,dt):
    psi = x[2]
    u_r = x[3]
    v_r = x[4]
    r = x[5]

    s = np.sin(psi)
    c = np.cos(psi)

    first_part = np.matrix([[0, 0, -u_r*s-v_r*c, c, -s, 0],
                            [0, 0,  u_r*c-v_r*s, s,  c, 0],
                            [0, 0,  0,           0,  0, 1]])
    
    F = np.vstack((first_part, np.zeros((3,6))))

    A = np.eye(6) + dt * F

    return A

def AJacobian_new_model(x,dt): # not used anymore
    psi = x[2]
    u_r = x[3]
    v_r = x[4]
    r = x[5]

    m11 = m + X_u_dot
    m22 = m + Y_v_dot
    m33 = N_r_dot

    # Mass matrix
    M_sim = np.matrix([[m11, 0,   0],
                       [0,   m22, 0],
                       [0,   0,   m33]])

    M_inv_sim = np.linalg.inv(M_sim)

    # compute the derivative of the equation of motion w.r.t. ur, vr and r
    derv = np.matrix([[X_u,            -m22*r,        -m22*v_r],
                      [m11*r,           Y_v,           m11*u_r],
                      [(m22-m11)*v_r,  (m22-m11)*u_r,  N_r]])
    
    A_eq = np.hstack((np.zeros((3,3)), derv)) # 3x6
    A_eq = -np.matmul(M_inv_sim,A_eq) # 3x6
    

    # compute the derivative of the rotation part
    s = np.sin(psi)
    c = np.cos(psi)
    A_r = np.matrix([[0, 0, -u_r*s -v_r*c, c, -s, 0],
                     [0, 0,  u_r*c -v_r*s, s,  c, 0],
                     [0, 0,  0,            0,  0, 1]])
    
    # A_r = np.matrix([[0, 0, -(u_r+u_c)*s - (v_r+v_c)*c, c, -s, 0],
    #                  [0, 0,  (u_r+u_c)*c - (v_r+v_c)*s, s,  c, 0],
    #                  [0, 0,  0,                         0,  0, 1]])
    
    # Transition matrix F
    F = np.vstack((A_r,A_eq)) # 6x6

    # Discretize the transition matrix
    A = np.eye(6) + dt * F

    return A

def dynamics(x,u,dt):
    pos_x = x[0]
    pos_y = x[1]
    psi = x[2]
    u_r = x[3]
    v_r = x[4]
    r = x[5]
    
    fx = u[0]
    fy = u[1]
    mz = u[2]

    # x = np.expand_dims(x,1).copy()

    N = np.matrix([[-X_u,                  -m*r,              Y_v_dot*v_r+Y_r_dot*r],
                   [m*r,                   -Y_v,             -X_u_dot*u_r-Y_r],
                   [-Y_v_dot*v_r-Y_r_dot*r, X_u_dot*u_r-N_v, -N_r]])

    rot_m = np.matrix([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi),  np.cos(psi), 0], 
                       [0,            0,           1]])
    vector = np.array([[u_r + u_c], [v_r + v_c], [r]])  # 3x1

    A = np.matmul(M_inv,N) # 3x3
    B = np.matmul(M_inv,np.array([[fx], [fy], [mz]])) # 3x1
    eq_o_m = -np.matmul(A,np.array([[u_r], [v_r], [r]])) + B # 3x3 * 3x1 + 3x1 = 3x1
    rot = np.matmul(rot_m,vector) # 3x3 * 3x1 => 3x1

    output = np.vstack((rot,eq_o_m)) # 6x1

    # Descritize the output
    output = np.array([[pos_x], [pos_y], [psi], [u_r], [v_r], [r]]) + dt * output # 6x1 + dt* 6x1 = 6x1
  
    return np.squeeze(output) # (6,)

def dynamics_fossen(x,u,dt):
    pos_x = x[0]
    pos_y = x[1]
    psi = x[2]
    u_r = x[3]
    v_r = x[4]
    r = x[5]
    
    fx = u[0]
    fy = u[1]
    mz = u[2]

    x = np.expand_dims(x,1)
    input = np.array([[fx], [fy], [mz]])


    ######################################
    # Model: eta_d = R*nu_r + nu_c       #
    #        nu_d  = M^-1*(tau - N*nu)   #
    ######################################


    # Rotation matrix
    rot_m = np.matrix([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi),  np.cos(psi), 0], 
                       [0,            0,           1]])


    # Mass matrix
    Mass_matrix = np.matrix([[m+X_u_dot, 0,         0],
                             [0,         m+Y_v_dot, 0],
                             [0,         0,         N_r_dot]])

    # Inverse of the mass matrix
    Mass_inv = np.linalg.inv(Mass_matrix)

    # Sum of Coriolis and Damping matrix
    N = np.matrix([[-X_u,         -m*r,          Y_v_dot*v_r],  # -m*r in  0,1
                   [ m*r,         -Y_v,         -X_u_dot*u_r],  # m*r in 1,0
                   [-Y_v_dot*v_r,  X_u_dot*u_r, -N_r]])
    

    A_cont_upper = np.hstack((np.zeros((3,3)), rot_m))
    A_cont_lower = np.hstack((np.zeros((3,3)), - Mass_inv @ N))
    A_cont = np.vstack((A_cont_upper,A_cont_lower))

    B_cont = np.vstack((np.zeros((3,3)), Mass_inv))

    # Discretize A and B
    A_d = np.eye(n_states) + dt * A_cont 
    B_d = dt * B_cont
    # A_d = expm(A_cont * dt)
    # B_d = np.linalg.inv(A_cont) @ (A_d - np.eye(n_states)) @ B_cont
    
    # Calculate new discrete state
    xp = A_d * x + B_d * input + dt * np.array([[u_c], [v_c], [0], [0], [0], [0]]) + np.random.normal(0, np.array([np.diagonal(Q)]).T, size=(n_states,1))

    return np.squeeze(xp) # (6,)

def dynamics_focus_group(x,u,dt):
    pos_x = x[0]
    pos_y = x[1]
    psi = x[2]
    u_r = x[3]
    v_r = x[4]
    r = x[5]
    
    fx = u[0]
    fy = u[1]
    mz = u[2]

    x = np.expand_dims(x,1)
    input = np.array([[fx], [fy], [mz]])

    ######################################
    # Model: eta_d = R*nu_r + nu_c       #
    #        nu_d  = M^-1*(tau - N*nu)   #
    ######################################
    

    # Rotation matrix
    rot_m = np.matrix([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi),  np.cos(psi), 0], 
                       [0,            0,           1]])


    # Mass matrix
    Mass_matrix = np.matrix([[m-X_u_dot, 0,         0],
                             [0,         m-Y_v_dot, 0],
                             [0,         0,         N_r_dot]])

    # Inverse of the mass matrix
    Mass_inv = np.linalg.inv(Mass_matrix)

    # Sum of Coriolis and Damping matrix
    # X_u = -10
    # Y_v = -10
    # N_r = 2   
    N = np.matrix([[X_u, -m*r,   0],  
                   [m*r, Y_v,   0],  
                   [0,    0,    N_r]])
    

    A_cont_upper = np.vstack((np.zeros((3,3)), rot_m))
    A_cont_lower = np.vstack((np.zeros((3,3)), - Mass_inv @ N))
    A_cont = np.hstack((A_cont_upper,A_cont_lower))

    B_cont = np.vstack((np.zeros((3,3)), Mass_inv))

    # Discretize A and B
    A_d = np.eye(n_states) + dt * A_cont
    B_d = dt * B_cont
    

    xp = A_d * x + B_d * input


    return np.squeeze(xp) # (6,)

def dynamics_new_model(x,u,dt): # not used anymore
    pos_x = x[0]
    pos_y = x[1]
    psi = x[2]
    u_r = x[3]
    v_r = x[4]
    r = x[5]
    
    fx = u[0]
    fy = u[1]
    mz = u[2]

    # m11 = m + X_u_dot
    # m22 = m + Y_v_dot
    # m33 = N_r_dot

    ######################################
    # Model: eta_d = R*nu_r + nu_c       #
    #        nu_d  = M^-1*(tau - N*nu)   #
    ######################################

    # Rotation matrix
    rot_m = np.matrix([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi),  np.cos(psi), 0], 
                       [0,            0,           1]])

    
    vector = np.array([[u_r], [v_r], [r]])  # 3x1

    # vector = np.array([[u_r + u_c], [v_r + v_c], [r]])  # 3x1
    
    rot = np.matmul(rot_m,vector) + np.array([[u_c], [v_c], [0]]) # 3x3 * 3x1 => 3x1

    # Mass matrix
    M_sim = np.matrix([[m11, 0,   0],
                       [0,   m22, 0],
                       [0,   0,   m33]])
    
    # Sum of Coriolis and Damping matrix
    N = np.matrix([[X_u,       0,       -m22*v_r],
                   [0,        -Y_v,      m11*u_r],
                   [m22*v_r,  -m11*u_r,  N_r]])

    # Inverse of the mass matrix
    M_inv_sim = np.linalg.inv(M_sim)
    M_inv_N = np.matmul(M_inv_sim,N) # 3x3
    B = np.matmul(M_inv_sim,np.array([[fx], [fy], [mz]])) # 3x1
    eq_o_m = -np.matmul(M_inv_N,np.array([[u_r], [v_r], [r]])) + B # 3x3 * 3x1 + 3x1 = 3x1
    

    output = np.vstack((rot,eq_o_m)) # 6x1

    # Descritize the output and add some noise
    xp = np.array([[pos_x], [pos_y], [psi], [u_r], [v_r], [r]]) + dt * output #+ np.random.normal(0, np.array([np.diagonal(Q)]).T, size=(6,1)) # 6x1 + dt* 6x1 = 6x1

    return np.squeeze(xp)

def dynamics_no_input(x,dt):

    x[0] = x[0] + (x[3]*np.cos(x[2]) - x[4]*np.sin(x[2]))*dt
    x[1] = x[1] + (x[3]*np.sin(x[2]) + x[4]*np.cos(x[2]))*dt
    x[2] = x[2] + x[5]*dt

    x = np.expand_dims(x,1)
    x = x + np.random.normal(0, np.array([np.diagonal(Q)]).T, size=(6,1))

    return np.squeeze(x)

def measurement_imu(data):
    # data consists of psi, roll_rate, pitch_rate, yaw_rate and quaternions
    z = np.zeros((n_states,1))

    rot_imu_to_body = Rotation.from_euler('xyz', [0.0, 0.0, -0.5*np.pi]) # bno055 0, 0, -np.pi/2 xsense: np.pi, 0.0, -0.5*np.pi
    # rot_imu_to_body = rot_body_to_imu.inv()
    imu_absolute_orientation = Rotation.from_quat(data[4])
    rot_enu_to_body = imu_absolute_orientation * rot_imu_to_body
    # rot_body_to_enu = rot_enu_to_body.inv()
    z[2,0] = rot_enu_to_body.as_euler('xyz')[2]

    omega = rot_imu_to_body.apply(np.squeeze(np.array([[data[1]], [data[2]], [data[3]]])))
    z[5,0] = omega[2] 
    


    return np.squeeze(z.T)

def measurement_imu_calc_ang(imu_data, old_data, dt, Mag_calculation):
    # data consists of psi, roll_rate, pitch_rate, yaw_rate and quaternions
    z = old_data
    z = np.expand_dims(z,1)

    rot_imu_to_body = Rotation.from_euler('xyz', [0.0, 0.0, -0.5*np.pi]) # bno055 0, 0, -np.pi/2 xsense: np.pi, 0.0, -0.5*np.pi
    # rot_imu_to_body = rot_body_to_imu.inv()
    angular_velocity_imu = np.array([imu_data[1], imu_data[2], imu_data[3]])
    omega = rot_imu_to_body.apply(angular_velocity_imu)

    z[5,0] = omega[2] 
    if Mag_calculation:
        angular_change = omega[2] * dt
        z[2,0] = angular_change + old_data[2]    
    else:
        imu_absolute_orientation = Rotation.from_quat(imu_data[4])
        rot_enu_to_body = imu_absolute_orientation * rot_imu_to_body
        # rot_body_to_enu = rot_enu_to_body.inv()
        z[2,0] = rot_enu_to_body.as_euler('xyz')[2] 

    return np.squeeze(z.T)

def measurement_gps(z,gps,imu):
    # z input is the z from the imu => (,6)
    z = np.expand_dims(z,1)
    x_pos = gps[0]
    y_pos = gps[1]
    x_vel = gps[2]
    y_vel = gps[3]
    z_vel = gps[4]
    roll_rate = imu[1]
    pitch_rate = imu[2]
    yaw_rate = imu[3]
    quats = imu[4]

    ##############################################################################
    # Position transformation
    trans_body_to_gps = np.array([-0.145, 0.015, 0.16])
    imu_abs_orientation = Rotation.from_quat(quats)
    rot_imu_to_body = Rotation.from_euler('xyz', [0.0, 0.0, -0.5*np.pi]) # bno055: 0, 0, -np.pi/2  xsense: np.pi, 0.0, -0.5*np.pi
    rot_body_to_gps = Rotation.from_euler('xyz', [0.0, 0.0, 0.0])
    # rot_imu_to_body = rot_body_to_imu.inv()
    rot_body_to_enu = imu_abs_orientation * rot_imu_to_body
    gps_offset_enu = rot_body_to_enu.apply(trans_body_to_gps)

    z[0,0] = x_pos - gps_offset_enu[0]
    z[1,0] = y_pos - gps_offset_enu[1]

    ##############################################################################
    # Velocity transformation
    gps_to_enu_rotation = rot_body_to_enu
    velocity_gps_frame = np.array([x_vel, y_vel, z_vel])
    gps_to_body_rotation = gps_to_enu_rotation.inv()
    velocity_body_frame = gps_to_body_rotation.apply(velocity_gps_frame)

    # Adjusting the linear velocity by the rotational part
    angular_velocity_imu_frame = np.array([roll_rate, pitch_rate, yaw_rate])
    angular_velocity_body_frame = rot_imu_to_body.apply(angular_velocity_imu_frame)
    r_gps_body = trans_body_to_gps
    rotational_velocity_gps_body = np.cross(angular_velocity_body_frame, r_gps_body)
    velocity_gps_translational_body = velocity_body_frame
    true_velocity_body_frame = velocity_gps_translational_body - rotational_velocity_gps_body

    z[3,0] = true_velocity_body_frame[0]
    z[4,0] = true_velocity_body_frame[1]


    return np.squeeze(z.T)

def measurement_gps_calc_ang(z,gps,imu,mag,Mag_calculation, old_angle, input):
    # z input is the z from the imu => (,6)
    prev_x_vel = z[3]
    prev_y_vel = z[4]
    x_pos = gps[0]
    y_pos = gps[1]
    x_vel = gps[2]
    y_vel = gps[3]
    z_vel = gps[4]
    roll_rate = imu[1]
    pitch_rate = imu[2]
    yaw_rate = imu[3]
    quats = imu[4]
    z = np.expand_dims(z,1)

    ##############################################################################
    # Position transformation
    trans_body_to_gps = np.array([-0.145, 0.015, 0.16])
    imu_abs_orientation = Rotation.from_quat(quats)
    rot_imu_to_body = Rotation.from_euler('xyz', [0.0, 0.0, -0.5*np.pi]) # bno055: 0, 0, -np.pi/2  xsense: np.pi, 0.0, -0.5*np.pi
    rot_body_to_gps = Rotation.from_euler('xyz', [0.0, 0.0, 0.0])
    # rot_imu_to_body = rot_body_to_imu.inv()
    rot_body_to_enu = imu_abs_orientation * rot_imu_to_body
    gps_offset_enu = rot_body_to_enu.apply(trans_body_to_gps)

    z[0,0] = x_pos - gps_offset_enu[0]
    z[1,0] = y_pos - gps_offset_enu[1]

    ##############################################################################
    # Velocity transformation
    gps_to_enu_rotation = rot_body_to_enu
    velocity_gps_frame = np.array([x_vel, y_vel, z_vel])
    gps_to_body_rotation = gps_to_enu_rotation.inv()
    velocity_body_frame = gps_to_body_rotation.apply(velocity_gps_frame)

    # Adjusting the linear velocity by the rotational part
    angular_velocity_imu_frame = np.array([roll_rate, pitch_rate, yaw_rate])
    angular_velocity_body_frame = rot_imu_to_body.apply(angular_velocity_imu_frame)
    r_gps_body = trans_body_to_gps
    rotational_velocity_gps_body = np.cross(angular_velocity_body_frame, r_gps_body)
    velocity_gps_translational_body = velocity_body_frame
    true_velocity_body_frame = velocity_gps_translational_body - rotational_velocity_gps_body

    z[3,0] = true_velocity_body_frame[0]
    z[4,0] = true_velocity_body_frame[1]

    ##############################################################################
    # Calculation of the angle by fusing the mag data with the velocity vectors
    if Mag_calculation:

        mag_rot = rot_imu_to_body.apply(mag)

        mag_x = mag_rot[0]
        mag_y = mag_rot[1]
        mag_z = mag_rot[2]

        magnetic_declination = math.radians(3.54) # depend on your location => Zürich
        magnitude = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
        mag_x = mag_x / magnitude
        mag_y = mag_y / magnitude

        heading = math.atan2(mag_y, mag_x)
        heading = (heading + 2*np.pi) % (2*np.pi)
        true_heading = (heading + magnetic_declination) % (2*np.pi)

        # Change from NED into ENU frame
        true_heading = np.pi/2 - true_heading

        # absolute_vel = np.sqrt(x_vel**2 + y_vel**2)
        change = math.atan2(z[4,0],z[3,0]) - math.atan2(prev_y_vel,prev_x_vel)

        if true_heading > np.pi:
            true_heading -= 2*np.pi
        elif true_heading < -np.pi:
            true_heading += 2*np.pi

        if change > np.pi:
            change -= 2*np.pi
        elif change < -np.pi:
            change += 2*np.pi

        # if absolute_vel < 0.05: # if the velocity is to small, we don't use it
        if abs(input[0]) < 0.1 and abs(input[1]) < 0.1:    
            z[2,0] = true_heading #+ (true_heading - old_angle)
        else:
            # adjusted = true_heading + change / 2
            adjusted = old_angle + change
            # if adjusted > np.pi:
            #     adjusted -= 2*np.pi
            # elif adjusted < -np.pi:
            #     adjusted += 2*np.pi
            z[2,0] = adjusted
    
    return np.squeeze(z.T)

def prior_update(A,Pm):
    Pp = A @ Pm @ A.T + Q
    return Pp

def posterior_update(xp, Pp, z, gps_on):
    # z => (,6)
    # x => (,6)
    xp = np.expand_dims(xp,1) # x => 6x1
    z = np.expand_dims(z,1) # z => 6x1

    # Compute H, depending on wether there is gps measurement or not
    H = np.eye(n_states)
    # H[0,0] = 0 # set x to zero
    # H[1,1] = 0
    # H[3,3] = 0 # x_vel to zero
    # H[4,4] = 0 # y_vel to zero
    # z[3,0] = 0
    # z[4,0] = 0
    
    if gps_on != True:
        # WITHOUT gps measurement
        H[0,0] = 0 # set x to zero
        H[1,1] = 0 # set y to zero
        # H[2,2] = 0
        H[3,3] = 0 # x_vel to zero
        H[4,4] = 0 # y_vel to zero
        # z[0,0] = 0
        # z[1,0] = 0
        # z[3,0] = 0
        # z[4,0] = 0

 
    # Kalman Gain
    inv = H @ Pp @ H.T + R
    if np.linalg.det(inv) < 0.00001: # avoid singularity => take the pseudo-inverse
        Kk = Pp @ H.T @ np.linalg.pinv(inv) # 6x6
    else:
        Kk = Pp @ H.T @ np.linalg.inv(inv) # 6x6
    
    # mean update
    y = z - H @ xp
    # y[3,0] = 0
    # y[4,0] = 0  
    xm = xp + Kk @ y

    # Variance update
    Pm = (np.eye(n_states) - Kk @ H) @ Pp @ (np.eye(n_states) - Kk @ H).T + Kk @ R @ Kk.T
    xm = xm.T
 
    return np.squeeze(xm), Pm

def get_derivative(data):
    length = len(data[:,0])
    derivative = np.zeros((length,2))
    derivative[0,0] = data[0,0]
    for i in range(1,length):
        derivative[i,0] = data[i,0]
        delta_t = data[i,0] - data[i-1,0]
        derivative[i,1] = (data[i,1] - data[i-1,1]) / delta_t

    return derivative 

def imu(psi, roll_rate, pitch_rate, yaw_rate, quats): # saves the [timestamp, [psi, roll_rate, pitch_rate, yaw_rate, quats]]
    length = len(psi[:,0])
    # imu = [(['timestamp'], ['imu_measurement'])]

    # index               0            1               2               3              4
    imu = [(psi[0,0], [psi[0,1], roll_rate[0,1], pitch_rate[0,1], yaw_rate[0,1], quats[0,1:]])]
    for i in range(1,length):
        # tmp = (u_r_dot[i,0], [psi[i,1], u_r[i,1], v_r[i,1], r[i,1]])
        tmp = (psi[i,0], [psi[i,1], roll_rate[i,1], pitch_rate[i,1], yaw_rate[i,1], quats[i,1:]])
        imu.append(tmp)

    return imu

def gps(GPS_pos, GPS_vel): # saves [time, [x, y, x_vel, y_vel, z_vel, rel_psi]]
    length = len(GPS_pos[:,0])
    # v1 = np.zeros((2,1))
    # v2 = np.zeros((2,1))

    # angle = np.zeros((length,1))
    # for i in range(1,length):
    #     v1[0,0] = GPS_vel[i-1,1]
    #     v1[1,0] = GPS_vel[i-1,2]
    #     v2[0,0] = GPS_vel[i,1]
    #     v2[1,0] = GPS_vel[i,2]
    #     v1v2 = v1[0,0] * v2[0,0] + v1[1,0] * v2[1,0]
    #     norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    #     if norm <= 0.0000001:
    #         angle[i,0] = angle[i-1,0]
    #     else:
    #         angle[i,0] = angle[i-1,0] + math.acos(v1v2/norm)


    # index     time           0,x           1,y           2,x_vel       3,y_vel       4,z_vel       
    gps = [(GPS_pos[0,0], [GPS_pos[0,1], GPS_pos[0,2], GPS_vel[0,1], GPS_vel[0,2], GPS_vel[0,3]])]
    for i in range(1,length):
        tmp = (GPS_pos[i,0], [GPS_pos[i,1], GPS_pos[i,2], GPS_vel[i,1], GPS_vel[i,2], GPS_vel[i,3]])
        gps.append(tmp)

    return gps

def input(u): # saves the timestamp and [Fx, Fy, Mz]
    length = len(u[:,0])
    # input = [(['timestamp'], ['control_input'])]
    input = [(u[0,0], [u[0,1], u[0,2], u[0,3]])]
    for i in range(1,length):
        tmp = (u[i,0], [u[i,1], u[i,2], u[i,3]])
        input.append(tmp)

    return input
    
radius = 0.2835 # in m
m = 10.835 # in kg

# Read in the constant coefficients
path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Code\coef\coef_best.csv")  # has to be changed
c = pd.read_csv(path).values
# X_u_dot = 2.375e+01 #c[0,0] -30 -200 -15 -0.5
# Y_v_dot = 2.375e+01 #c[0,1] -30 -200 -15 -0.5
# # Y_r_dot = c[0,2]
# # N_v_dot = c[0,3]
# N_r_dot = 1.949e+00 #c[0,4] 0.7
# X_u = -9.229e+01 #c[0,5] -110 75 -1.5 -35 -75  0.5
# Y_v = -9.229e+01 #c[0,6] -100 75 -1.5 -35 -30  0.5
# # Y_r = c[0,7]
# # N_v = c[0,8]
# N_r = -2.663e+00 #c[0,9] 0.6 0.05
u_c = 0.0 #c[0,10] 
v_c = 0.0 #c[0,11]
X_u_dot = c[0,0]
Y_v_dot = c[0,0]
N_r_dot = c[1,0]
X_u = c[2,0]
Y_v = c[2,0]
N_r = c[3,0]

##############################################################################################################################################
#################                Complementary Filtering               #######################################################################
##############################################################################################################################################

class IMUComplementaryFilter:
    def __init__(self, alpha_orientation, alpha_angular_velocity):
        """
        Initialize the complementary filter for both orientation and angular velocity around z-axis.
        
        Parameters:
        - alpha_orientation: Weight for the high-frequency IMU orientation data (0 < alpha < 1).
        - alpha_angular_velocity: Weight for the high-frequency IMU angular velocity data.
        """
        self.alpha_orientation = alpha_orientation
        self.alpha_angular_velocity = alpha_angular_velocity
        self.orientation_estimate = Rotation.from_quat([0, 0, 0, 1])  # Initial quaternion as identity
        self.filtered_angular_velocity_z = 0.0

    def update(self, imu_quaternion, imu_angular_velocity_z, dt, reference_orientation=None, reference_angular_velocity_z=None):
        """
        Preprocess orientation and angular velocity using the complementary filter.
        
        Parameters:
        - imu_quaternion: Quaternion orientation from IMU (w, x, y, z).
        - imu_angular_velocity_z: Angular velocity around z-axis from IMU (rad/s).
        - dt: Time interval since the last IMU update.
        - reference_orientation: Optional low-frequency orientation reference as a quaternion (w, x, y, z).
        - reference_angular_velocity_z: Optional low-frequency angular velocity reference around z-axis (rad/s).
        
        Returns:
        - Filtered orientation estimate as a quaternion (w, x, y, z)
        - Filtered angular velocity around z-axis (rad/s)
        """
        # --- Orientation Filtering ---
        imu_rotation = Rotation.from_quat(imu_quaternion)
        
        # Integrate angular velocity for estimated orientation change
        theta = np.linalg.norm(imu_angular_velocity_z) * dt
        delta_rotation = Rotation.from_rotvec([0, 0, theta])  # Rotation around z-axis
        
        orientation_with_gyro = self.orientation_estimate * delta_rotation

        # Blend gyro-integrated and IMU-measured orientations
        if reference_orientation is not None:
            reference_rotation = Rotation.from_quat(reference_orientation)
            blended_orientation = Rotation.from_quat(
                (1 - self.alpha_orientation) * reference_rotation.as_quat() + 
                self.alpha_orientation * orientation_with_gyro.as_quat()
            ).as_quat()
            blended_orientation /= np.linalg.norm(blended_orientation)  # Normalize to ensure it's a unit quaternion
            self.orientation_estimate = Rotation.from_quat(blended_orientation)
        else:
            # Use IMU and gyro-integrated orientation for blending
            blended_orientation = Rotation.from_quat(
                (1 - self.alpha_orientation) * imu_rotation.as_quat() + 
                self.alpha_orientation * orientation_with_gyro.as_quat()
            ).as_quat()
            blended_orientation /= np.linalg.norm(blended_orientation)  # Normalize to ensure it's a unit quaternion
            self.orientation_estimate = Rotation.from_quat(blended_orientation)

        # --- Angular Velocity Filtering ---
        if reference_angular_velocity_z is not None:
            self.filtered_angular_velocity_z = self.alpha_angular_velocity * imu_angular_velocity_z + \
                                               (1 - self.alpha_angular_velocity) * reference_angular_velocity_z
        else:
            self.filtered_angular_velocity_z = self.alpha_angular_velocity * self.filtered_angular_velocity_z + \
                                               (1 - self.alpha_angular_velocity) * imu_angular_velocity_z


        # Return filtered results
        return self.orientation_estimate.as_quat(), self.filtered_angular_velocity_z

##############################################################################################################################################
####################                 EKF                  ####################################################################################
##############################################################################################################################################

# Option 1 => real data
# Option 2 => synthetic data

# CHANGE THE FOR LOOP BEFORE THE SIMULATION
option = 1
april_tags = False
GPS_OFF = False
Mag_calculation = True

# number of states
n_states = 6

if option==1:
    # path to the data that should be used
    path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Measurements\rosbags_15_11_24_xsense_bno\rosbags_15_11_24_xsense_bno\6.1_zick_zack\rosbag2_2024_11_15-16_08_42_0.mcap")
    
    # extract all necessary data
    u_r_dot, v_r_dot, roll_rate, pitch_rate, yaw_rate, euler_angles_enu, euler_angles_ned, quats = extract_imu_data(path)
    psi = euler_angles_enu[:,[0,3]] 
    F_x, F_y, M_z, u = extract_motor_values(path,radius)
    F_x_new, F_y_new, M_z_new, u_new = extract_motor_values_new(path,radius)
    GPS_ENU = extract_gps_data(path) # length x 4 => time, e, n, u
    GPS_vel = extract_gps_vel(path) # length x 4 => time, x_vel, y_vel, z_vel
    mag_data = extract_mag_data(path) # length x 4 => time, x, y, z  
    u_r_from_IMU = np.zeros((len(u_r_dot[:,0]),2))
    v_r_from_IMU = np.zeros((len(v_r_dot[:,0]),2))
    

    # bring it into the needed form
    imu_data = imu(psi,roll_rate,pitch_rate,yaw_rate,quats)
    gps_data = gps(GPS_ENU,GPS_vel)
    control_data = input(u_new)
    # START_imu = imu_data[0][0]
    # START_gps = gps_data[0][0]
    # fig, axs = plt.subplots(3,1)
    # axs[0].plot(u[:,0], F_x[:,1])
    # axs[0].plot(F_x_new[:,0], F_x_new[:,1])
    # axs[1].plot(F_y[:,0], F_y[:,1])
    # axs[1].plot(F_y_new[:,0], F_y_new[:,1])
    # axs[2].plot(M_z[:,0], M_z[:,1])
    # axs[2].plot(M_z_new[:,0], M_z_new[:,1])
    

    # define the number of steps to do the ekf
    number_of_steps = len(imu_data)

    # read out april tags
    if april_tags:
        position_tags, orientation_tags = extract_april_tags(path)
        april_tag_start = position_tags[0,0]
        for i in range(len(u_r_dot[:,0])):
            if u_r_dot[i,0] >= april_tag_start:
                april_tag_start = i
                break
    
    fig, axs = plt.subplots(3,1, sharex=True)
    # axs[0].plot(F_x[:,0], F_x[:,1])
    axs[0].plot(np.linspace(0,number_of_steps, len(F_x_new[:,1])), F_x_new[:,1])
    axs[0].set_ylabel('Force in x [N]',fontsize=14)
    # axs[1].plot(F_y[:,0], F_y[:,1])
    axs[1].plot(np.linspace(0,number_of_steps, len(F_y_new[:,1])), F_y_new[:,1])
    axs[1].set_ylabel('Force in y [N]',fontsize=14)
    # axs[2].plot(M_z[:,0], M_z[:,1])
    axs[2].plot(np.linspace(0,number_of_steps, len(M_z_new[:,1])), M_z_new[:,1])
    axs[2].set_ylabel('Moment around z [Nm]',fontsize=14)
    plt.xlabel('IMU-steps', fontsize=14)
    plt.suptitle('Force and Moment', size='xx-large', weight='bold')


elif option==2:
    path = Path(r"C:\Users\safre\OneDrive - ETH Zurich\ETH\Master\Semesterprojekt\Code\synthetic_data\rectangle.csv")
    syn_data = read_in_syn_data(path)
    # Add noise to the data => syn_data = add_noise(syn_data) => zero mean gaussian noise with Cov R, or maybe also uniform distr.
    number_of_steps = len(syn_data)
    if GPS_OFF:
        GPS_Off_start = 3000 # between 0 and 900 => means between 0s and 90s
        GPS_Off_end = 5300 # between 0 and 900 => means between 0s and 90s


# plt.plot(M_z[:,0], M_z[:,1])
# plt.show()
# u_r_from_input = np.zeros((len(control_data),2))
# v_r_from_input = np.zeros((len(control_data),2))

# for i in range(1,len(control_data)):
#     u_r_from_input[i,0] = control_data[i][0]
#     v_r_from_input[i,0] = control_data[i][0]
#     delta_t = control_data[i][0] - control_data[i-1][0]
#     u_r_from_input[i,1] = control_data[i][1][0] * delta_t / m
#     v_r_from_input[i,1] = control_data[i][1][1] * delta_t / m

# Define state matrix
xp_f = np.zeros((number_of_steps+1,n_states)) # 6 states
xp_s = np.zeros((number_of_steps+1,n_states)) # 6 states

# initialize state
xp0 = np.random.rand(1,n_states)
xp_f[0,:] = np.squeeze(xp0)
xp_s[0,:] = np.squeeze(xp0)

# define estimator parameters
if option==1:
    Q = np.diag([0.001, 0.001, 0.01*np.pi/180, 0.01, 0.01, 0.1*np.pi/180]) # if orient from mag av then [0.02, 0.02, 0.5*np.pi/180, 0.01, 0.01, 0.5*np.pi/180]
    if GPS_OFF:
        R = np.diag([0.0, 0.0, 0.02, 0.0, 0.0, 0.04]) # no GPS measurement
    else:
        R = np.diag([0.02, 0.02, 0.02, 0.017, 0.017, 0.04])  # 0.02, 0.02, 0.0159, 0.017, 0.017, 0.04
elif option==2:
    Q = np.diag([0.0, 0.0, 0.0*np.pi/180, 0.02, 0.02, 0.5*np.pi/180]) # model covariance
    R = np.diag([0.01, 0.01, 0.02, 0.1, 0.1, 0.005]) # measurement covariance

# initialize estimator
xm0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
Pm0 = np.diag([2, 2, 0.5, 5, 5, 0.5])
xm_f = np.zeros((number_of_steps, n_states))
Pm_f = np.zeros((number_of_steps, n_states, n_states))
xm_s = np.zeros((number_of_steps, n_states))
Pm_s = np.zeros((number_of_steps, n_states, n_states))

xm_f[0,:] = np.squeeze(xm0)
Pm_f[0,:,:] = np.squeeze(Pm0)
xm_s[0,:] = np.squeeze(xm0)
Pm_s[0,:,:] = np.squeeze(Pm0)

# initialize error vectors for the GPS data
if option==1:
    error_fossen_gps = np.zeros((len(GPS_ENU[:,0]),4)) # pos x & y, vel x & y
    error_simple_gps = np.zeros((len(GPS_ENU[:,0]),4)) # pos x & y, vel x & y
    GPS_vel_turned = np.zeros((len(GPS_ENU[:,0]),3)) # GPS velocity in body fixed coordinates
    GPS_pos_corrected = np.zeros((len(GPS_ENU[:,0]),3)) # GPS position with correction term from offset

# initialize some counter
start = True
first_run = True
imu_counter = 1
start_gps_counter = 0
gps_counter = 0
mag_counter = 1
control_counter = 0
start_imu = 0
delta_t_control = 0 # for linearizing the control input
last_control_input = np.array([0.0, 0.0, 0.0])
c_input = np.array([0.0, 0.0, 0.0])
mag_average = np.array([0.0, 0.0, 0.0])
z_f = np.zeros(6)
z_s = np.zeros(6)
old_angle = 0.0
if option==1:
    last_time = imu_data[0][0]
    # imu_filter = IMUComplementaryFilter(alpha_orientation=0.99, alpha_angular_velocity=0.99)
elif option==2:
    last_time = syn_data[0][0] 

# simulate
for timestamp, imu_measurement in imu_data: ########## Option 1
# for timestamp, imu_measurement in syn_data: ######### Option 2
    # break     
    
    # if we first only have GPS measurements and no IMU, we wait until IMU data is coming
    if option==1 and start:
        while timestamp > (gps_data[gps_counter][0]):
            gps_counter += 1
            start_gps_counter = gps_counter # only for plotting reasons needed
        while timestamp > (control_data[control_counter][0]):
            # last_control_time = control_data[control_counter][0]
            last_control_input = control_data[control_counter][1]
            c_input = last_control_input
            control_counter += 1
        start = False
            
    # get the time difference between the steps
    dt = timestamp - last_time
    last_time = timestamp
    # delta_t_control += dt

    # terminate if the end of the steps is reached
    if imu_counter == number_of_steps:
        break


    # u_r_from_IMU[imu_counter,0] = u_r_from_IMU[imu_counter-1,0] + dt
    # u_r_from_IMU[imu_counter,1] = u_r_from_IMU[imu_counter-1,1] + u_r_dot[imu_counter,1] * dt
    # v_r_from_IMU[imu_counter,0] = v_r_from_IMU[imu_counter-1,0] + dt
    # v_r_from_IMU[imu_counter,1] = v_r_from_IMU[imu_counter-1,1] + v_r_dot[imu_counter,1] * dt

    
    # get the actual control input => if no new one, the old one will be taken
    if option==1:
        if control_counter < len(control_data) and control_data[control_counter][0] <= timestamp:
            # control_input = control_data[control_counter][1]
            # last_control_time = control_data[control_counter][0]
            # last_control_input = control_data[control_counter][1]
            # c_input = last_control_input
            c_input = control_data[control_counter][1]
            # delta_t_control = 0
            control_counter += 1

    # Linearize between two input points
    # if (control_data[control_counter][0] - last_control_time) < 0.00001:
    #     c_input = last_control_input
    # else:
    #     c_input = (control_data[control_counter][1] - last_control_input) / (control_data[control_counter][0] - last_control_time) * (last_control_time + delta_t_control) + last_control_input

    # Update state
    if option==1:
        xp_f[imu_counter, :] = dynamics_fossen(xm_f[imu_counter - 1, :], c_input, dt)
        xp_s[imu_counter, :] = dynamics_no_input(xm_s[imu_counter - 1, :], dt)
        # xp_s[imu_counter, :] = dynamics_focus_group(xm_s[imu_counter - 1, :],control_input, dt) 
    elif option==2:
        xp_s[imu_counter, :] = dynamics_no_input(xm_s[imu_counter - 1, :], dt)


    # Prior update
    if option==1:

        # Simple model
        A_s = Jacobian_simple_model(xm_s[imu_counter - 1,:],dt)
        Ppk_s = prior_update(A_s,Pm_s[imu_counter - 1, :, :]) # Ppk = Pm + Q

        # Fossen model
        A_f = AJacobian_fossen(xm_f[imu_counter - 1,:],dt) # 6x6 matrix
        Ppk_f = prior_update(A_f,Pm_f[imu_counter - 1, :, :]) # Ppk = Pm + Q
    
    elif option==2:
        A = np.eye(n_states)
        A[0,3] = dt
        A[1,4] = dt
        A[2,5] = dt
        Ppk_s = prior_update(A,Pm_s[imu_counter - 1, :, :]) # Ppk = Pm + Q
    

   
    if option==1:
        # Calculate the orientation from mag data
        if Mag_calculation and imu_counter < len(mag_data[:,1]):
            mag_average = mag_average * (mag_counter-1) / mag_counter + mag_data[imu_counter,1:] / mag_counter
            mag_counter += 1
        
        # get data from the IMU
        z_s = measurement_imu(imu_measurement) # 1x6 vector in form (,6)
        z_f = measurement_imu_calc_ang(imu_measurement, z_f, dt, Mag_calculation) # 1x6 vector in form (,6)

    elif option==2:
        z = np.array([imu_measurement]).T + np.random.normal(0, np.array([np.diagonal(R)]).T, size=(6,1))# gives the complete state => (,6)
        z = np.squeeze(z)
        # z[3] = 0 # xdot => should be estimated by the ekf
        # z[4] = 0 # ydot



    # Measurement update
    if option==1:
        if gps_counter < len(gps_data) and (gps_data[gps_counter][0] - timestamp) < 0.009:
            if first_run:
                # Needed for plotting
                start_imu = imu_counter
                first_run = False

            gps_measurement = gps_data[gps_counter][1]
            z_s = measurement_gps(z_s,gps_measurement, imu_measurement)
            z_f = measurement_gps_calc_ang(z_f,gps_measurement, imu_measurement, mag_average, Mag_calculation, old_angle, c_input)
            mag_average = np.array([0.0, 0.0, 0.0])
            mag_counter = 1
            x_pos = z_f[0]
            y_pos = z_f[1]
            x_vel = z_f[3]
            y_vel = z_f[4]
            
            # print("-------New GPS measurement--------")
            # print(f"IMU time is: {timestamp}")
            # print(f"GPS time is: {gps_data[gps_counter][0]}")
            # print(f"Input time is: {control_data[control_counter-1][0]}")
            # print(f"Predicted velocity in x: {x_vel}")
            # print(f"Used force in x: {control_data[control_counter-1][1]}")
            if GPS_OFF :
                z_f[3] = 0
                z_f[4] = 0
                z_s[3] = 0
                z_s[4] = 0
                z_f[0] = 0
                z_f[1] = 0
                z_s[0] = 0
                z_s[1] = 0
                # xm_f[imu_counter, :], Pm_f[imu_counter, :, :] = posterior_update(xp_f[imu_counter,:], Ppk_f, z_f, False)
                # xm_s[imu_counter, :], Pm_s[imu_counter, :, :] = posterior_update(xp_s[imu_counter,:], Ppk_s, z_s, False)

            xm_f[imu_counter, :], Pm_f[imu_counter, :, :] = posterior_update(xp_f[imu_counter,:], Ppk_f, z_f, not GPS_OFF)
            xm_s[imu_counter, :], Pm_s[imu_counter, :, :] = posterior_update(xp_s[imu_counter,:], Ppk_s, z_s, not GPS_OFF)
            old_angle = z_f[2]  
            error_fossen_gps[gps_counter,0] = np.abs(xp_f[imu_counter,0] - GPS_ENU[gps_counter,1])
            error_simple_gps[gps_counter,0] = np.abs(xp_s[imu_counter,0] - GPS_ENU[gps_counter,1])
            error_fossen_gps[gps_counter,1] = np.abs(xp_f[imu_counter,1] - GPS_ENU[gps_counter,2])
            error_simple_gps[gps_counter,1] = np.abs(xp_s[imu_counter,1] - GPS_ENU[gps_counter,2])
            # error_fossen_gps[gps_counter,0] = np.abs(xp_f[imu_counter,0] - x_pos)
            # error_simple_gps[gps_counter,0] = np.abs(xp_s[imu_counter,0] - x_pos)
            # error_fossen_gps[gps_counter,1] = np.abs(xp_f[imu_counter,1] - y_pos)
            # error_simple_gps[gps_counter,1] = np.abs(xp_s[imu_counter,1] - y_pos)
            GPS_vel_turned[gps_counter,0] = timestamp
            GPS_vel_turned[gps_counter,1] = x_vel
            GPS_vel_turned[gps_counter,2] = y_vel
            GPS_pos_corrected[gps_counter,0] = timestamp
            GPS_pos_corrected[gps_counter,1] = x_pos
            GPS_pos_corrected[gps_counter,2] = y_pos
            error_fossen_gps[gps_counter,2] = np.abs(xp_f[imu_counter,3] - x_vel)
            error_simple_gps[gps_counter,2] = np.abs(xp_s[imu_counter,3] - x_vel)
            error_fossen_gps[gps_counter,3] = np.abs(xp_f[imu_counter,4] - y_vel)
            error_simple_gps[gps_counter,3] = np.abs(xp_s[imu_counter,4] - y_vel)
            gps_counter += 1
        else:
            # posterior update without new GPS measurement
            xm_f[imu_counter, :], Pm_f[imu_counter, :, :] = posterior_update(xp_f[imu_counter,:], Ppk_f, z_f, False)
            xm_s[imu_counter, :], Pm_s[imu_counter, :, :] = posterior_update(xp_s[imu_counter,:], Ppk_s, z_s, False)

    elif option==2:
        if imu_counter%10 != 5: # every 10 measurements steps => position measurement can be taken
            z[0] = 0 # pos x
            z[1] = 0 # pos y
            xm_s[imu_counter, :], Pm_s[imu_counter, :, :] = posterior_update(xp_s[imu_counter,:], Ppk_s, z, False)
        else:
            if GPS_OFF and imu_counter > GPS_Off_start and imu_counter < GPS_Off_end:
                z[0] = 0
                z[1] = 0
                z[3] = 0
                z[4] = 0
                xm_s[imu_counter, :], Pm_s[imu_counter, :, :] = posterior_update(xp_s[imu_counter,:], Ppk_s, z, False)
            else:
                xm_s[imu_counter, :], Pm_s[imu_counter, :, :] = posterior_update(xp_s[imu_counter,:], Ppk_s, z, True)
            
            gps_counter += 1


    # print(f"xm{imu_counter} is {xm[imu_counter, :]}")
    # if imu_counter > 3100 and imu_counter < 3400:
    #     print(f"xm_f{imu_counter} is {xm_f[imu_counter, :]}")
    
    # if imu_counter == 350:
    #     break

    imu_counter += 1
   
# print(f"Mean of the vel from state estimation: {np.mean(xp_f[:,3])}")
print(f"gps counter {gps_counter}")
print(f"imu counter: {imu_counter}")
print(f"control counter: {control_counter}")
print(f"GPS start counter: {start_gps_counter}")
print(f"start of the of GPS in imu_counter: {start_imu}")

if option==1:
    print(f"error fossen in position x: {np.sum(error_fossen_gps[:,0])}")
    print(f"error fossen in position y: {np.sum(error_fossen_gps[:,1])}")
    print(f"error fossen in velocity x: {np.sum(error_fossen_gps[:,2])}")
    print(f"error fossen in velocity y: {np.sum(error_fossen_gps[:,3])}")
    print(f"error simple in position x: {np.sum(error_simple_gps[:,0])}")
    print(f"error simple in position y: {np.sum(error_simple_gps[:,1])}")
    print(f"error simple in velocity x: {np.sum(error_simple_gps[:,2])}")
    print(f"error simple in velocity y: {np.sum(error_simple_gps[:,3])}")



# fig, axs = plt.subplots(2, 1)
# axs[0].plot(F_x[:,0], F_x[:,1], label='force in x')
# axs[1].plot(F_y[:,0], F_y[:,1], label='force in y')
# axs[0].legend()
# axs[1].legend()
# axs[0].plot(np.linspace(0,len(GPS_vel[:,1]),len(GPS_vel[:,1])), GPS_vel[:,1],color='g')
# axs[0].plot(np.linspace(0,len(GPS_vel_turned[:,1]),len(GPS_vel_turned[:,1])), GPS_vel_turned[:,1],color='b')
# axs[1].plot(np.linspace(0,len(GPS_vel[:,2]),len(GPS_vel[:,2])), GPS_vel[:,2],color='g')
# axs[1].plot(np.linspace(0,len(GPS_vel_turned[:,2]),len(GPS_vel_turned[:,2])), GPS_vel_turned[:,2],color='b')
# axs[0].plot(GPS_ENU[start_gps_counter:,0], GPS_ENU[start_gps_counter:,1])
# axs[0].plot(GPS_ENU[:,0], GPS_ENU[:,1])
# axs[1].plot(GPS_ENU[start_gps_counter:,0], GPS_ENU[start_gps_counter:,2])
# axs[1].plot(GPS_ENU[:,0], GPS_ENU[:,2])
# axs[0].plot(GPS_vel_turned[:,0], GPS_vel_turned[:,1])
# axs[1].plot(GPS_vel_turned[:,0], GPS_vel_turned[:,2])


rot_imu_to_body = Rotation.from_euler('xyz', [0.0, 0.0, -0.5*np.pi]) # bno055 0, 0, -np.pi/2   xsense: np.pi, 0.0, -0.5*np.pi
# rot_imu_to_body = rot_body_to_imu.inv()
omega = np.array([roll_rate[:,1], pitch_rate[:,1], yaw_rate[:,1]])
omega = rot_imu_to_body.apply(omega.T)
imu_absolute_orientation = Rotation.from_quat(quats[:,1:])
rot_enu_to_body = imu_absolute_orientation * rot_imu_to_body
# rot_body_to_enu = rot_enu_to_body.inv()
angle = rot_enu_to_body.as_euler('xyz')


# mag_data[:,1:] = rot_imu_to_body.apply(mag_data[:,1:]) 
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(np.linspace(0,number_of_steps, len(mag_data[:,1])), mag_data[:,1], label='mag in x')
# axs[0].plot(np.linspace(0,number_of_steps, len(mag_data[:,2])), mag_data[:,2], label='mag in y')
# axs[0].legend()




#############################################################################################################################################
# PLOTTING
#############################################################################################################################################

# read in the standard deviation from the measurement
if option==1:
    diag_f = np.zeros((len(Pm_f[:,0,0]),6))
    diag_s = np.zeros((len(Pm_s[:,0,0]),6))
    upper_bound_f = np.zeros((len(diag_f),6))
    upper_bound_s = np.zeros((len(diag_s),6))
    lower_bound_f = np.zeros((len(diag_f),6))
    lower_bound_s = np.zeros((len(diag_s),6))

    for i in range(len(Pm_f[:,0,0])):
        diag_f[i,:] = np.sqrt(np.diagonal(Pm_f[i,:,:]))
        diag_s[i,:] = np.sqrt(np.diagonal(Pm_s[i,:,:]))
        upper_bound_f[i,:] = xp_f[i,:] + diag_f[i,:]
        lower_bound_f[i,:] = xp_f[i,:] - diag_f[i,:]
        upper_bound_s[i,:] = xp_s[i,:] + diag_s[i,:]
        lower_bound_s[i,:] = xp_s[i,:] - diag_s[i,:]

    # Error calculation
    error_fossen_imu = np.zeros((number_of_steps,2)) # yaw-angle and yaw-rate
    error_simple_imu = np.zeros((number_of_steps,2)) # yaw-angle and yaw-rate
    for i in range(number_of_steps):
        error_fossen_imu[i,0] = np.abs(xp_f[i,2] - angle[i,2]) # yaw-angle
        error_simple_imu[i,0] = np.abs(xp_s[i,2] - angle[i,2]) # yaw-angle
        error_fossen_imu[i,1] = np.abs(xp_f[i,5] - omega[i,2]) # yaw-rate
        error_simple_imu[i,1] = np.abs(xp_s[i,5] - omega[i,2]) # yaw-rate
elif option==2:
    # Covariance
    diag_s = np.zeros((len(Pm_s[:,0,0]),6))
    for i in range(len(Pm_s[:,0,0])):
        diag_s[i,:] = np.diagonal(Pm_s[i,:,:])
    # Ground truth data and compute the error
    syn_data_plotting = pd.read_csv(path).values
    error = np.zeros((number_of_steps-1,6))
    absolut_error = np.zeros((number_of_steps,6))
    for i in range(1,number_of_steps):
        for j in range(6):
            error[i-1,j] = xp_s[i-1,j] - syn_data_plotting[i-1,j+1]
            # absolut_error[i,j] = absolut_error[i-1,j] + np.abs(error[i-1,j])  # calculate the summed error

###########################################
# Plotting the predicted states
labels = ["$x$ in [m]", "$y$ in [m]", "$Yaw$ in [rad]", "$u_r$ in [m/s]", "$v_r$ in [m/s]", "$r$ in [rad/s]"]
fig, axs = plt.subplots(n_states, 1, sharex=True)

for i in range(6):
    if option==1:
        # if GPS_OFF:
        #     axs[i].fill_betweenx([1.1*np.min([xp_s[1:-1, i],xp_f[1:-1, i]]), 1.1*np.max([xp_s[1:-1, i],xp_f[1:-1, i]])], GPS_Off_start, GPS_Off_end, color='green', alpha=0.2)
        axs[i].plot(np.linspace(1,number_of_steps, len(xp_f[1:,i])), xp_f[1:,i], label="Estimated Fossen", color='red')
        
        # axs[i].plot(np.arange(0, number_of_steps - 1), xp_f[1:-1, i], label="Estimated Fossen", color='orange')
    # elif option == 2:
    #     axs[i].plot(np.arange(0, number_of_steps - 2), syn_data_plotting[1:-1, i+1], label="Actual", color='b')
    #     axs[i].plot(np.arange(0, number_of_steps - 2), error[1:,i], label="Error", color='red', alpha=0.4)
    #     # axs[i].plot(np.arange(0, number_of_steps - 1), absolut_error[1:,i], label="Error", color='green', alpha=0.3)
    #     if GPS_OFF:
    #         axs[i].fill_betweenx([1.1*np.min(xp_s[1:-1, i]), 1.1*np.max(xp_s[1:-1, i])], GPS_Off_start, GPS_Off_end, color='green', alpha=0.2)
    axs[i].plot(np.linspace(1,number_of_steps, len(xp_s[1:,i])), xp_s[1:,i], label="Estimated Simple", color='green')
    axs[i].fill_between(np.linspace(1,number_of_steps,len(xp_s[1:,i])), lower_bound_s[:,i], upper_bound_s[:,i], color='green', alpha=0.2)
    axs[i].fill_between(np.linspace(1,number_of_steps,len(xp_f[1:,i])), lower_bound_f[:,i], upper_bound_f[:,i], color='red', alpha=0.2)


###########################################
# Plotting the actual states
if option == 1:
    axs[0].plot(np.linspace(start_imu,number_of_steps, len(GPS_pos_corrected[start_gps_counter:,0])), GPS_pos_corrected[start_gps_counter:, 1], label="Measurement GPS", color='blue')
    axs[1].plot(np.linspace(start_imu,number_of_steps, len(GPS_pos_corrected[start_gps_counter:,0])), GPS_pos_corrected[start_gps_counter:, 2], label="Measurement GPS", color='blue')
    axs[2].plot(np.linspace(0,number_of_steps, len(angle[:,2])), angle[:,2], label="Measurement IMU", color='blue')
    axs[3].plot(np.linspace(start_imu,number_of_steps, len(GPS_vel_turned[start_gps_counter:,0])), GPS_vel_turned[start_gps_counter:, 1], label="Measurement GPS", color='blue')
    axs[4].plot(np.linspace(start_imu,number_of_steps, len(GPS_vel_turned[start_gps_counter:,0])), GPS_vel_turned[start_gps_counter:, 2], label="Measurement GPS/IMU", color='blue')
    axs[5].plot(np.linspace(0,number_of_steps, len(omega[:,0])), omega[:,2], label="Measurement IMU", color='blue')
    if april_tags:
        axs[0].plot(np.linspace(april_tag_start,number_of_steps, len(position_tags[:,0])), position_tags[:,1], label="April tags", color='red')
        axs[1].plot(np.linspace(april_tag_start,number_of_steps, len(position_tags[:,0])), position_tags[:,2], label="April tags", color='red')
        axs[2].plot(np.linspace(april_tag_start,number_of_steps, len(orientation_tags[:,0])), orientation_tags[:,3], label="April tags", color='red')
    # axs[3].set_ylim([-1.5,1.5])
    # axs[4].set_ylim([-1.5,1.5])
    # axs[5].set_ylim([-3,3])

    # axs[3].plot(np.linspace(0,number_of_steps, len(u_r_from_IMU[:,0])), u_r_from_IMU[:,1], label='vel x from IMU', color='red')
    # axs[4].plot(np.linspace(0,number_of_steps, len(v_r_from_IMU[:,0])), v_r_from_IMU[:,1], label='vel y from IMU', color='red')
    # axs[3].plot(np.linspace(0,number_of_steps, len(u_r_from_input[:,0])), u_r_from_input[:,1], label='vel x from force', color='red')
    # axs[4].plot(np.linspace(0,number_of_steps, len(v_r_from_input[:,0])), v_r_from_input[:,1], label='vel y from force', color='red')
    # axs[3].plot(np.linspace(0,number_of_steps, len(GPS_vel[:,0])), GPS_vel[:,1], label="Velocity from GPS", color='pink')
    # axs[4].plot(np.linspace(0,number_of_steps, len(GPS_vel[:,0])), GPS_vel[:,2], label="Velocity from GPS", color='pink')

for i in range(6):
    axs[i].set_ylabel(labels[i], fontsize=14)
    # axs[i].tick_params(size=19)
    axs[i].yaxis.set_tick_params(size=14)
plt.xlabel("$IMU-steps$", fontsize=14)
plt.xticks(fontsize=14)
# plt.yticks(fontsize=12)
handles, labels = axs[4].get_legend_handles_labels()
rect = Rectangle((0, 0), 1, 1, fc="r", alpha=0.3)
rect2 = Rectangle((0, 0), 1, 1, fc="g", alpha=0.3)
handles.append(rect)
labels.append('Std. Deviation')
handles.append(rect2)
labels.append('Std. Deviation')
fig.legend(handles, labels, loc='center right', fontsize=14)
fig.subplots_adjust(hspace=0.1)
fig.suptitle('Predicted and true States', size='xx-large', weight='bold')


###########################################
# X-Y-Plot
plt.figure(7)
plt.plot(GPS_pos_corrected[start_gps_counter:, 1], GPS_pos_corrected[start_gps_counter:, 2], color='blue', label="Measurement GPS")
plt.plot(xp_f[1:-1,0], xp_f[1:-1,1], label="Estimated Fossen", color='red')
plt.suptitle('X-Y Map', size='xx-large', weight='bold')
plt.xlabel('x-position [m]', fontsize=14)
plt.ylabel('y-position [m]', fontsize=14)
plt.legend()




###########################################
# Plotting the error
if option==1:
    labels = ["$x$ in [m]", "$y$ in [m]", "$\Theta$ in [rad]", "$u_r$ in [m/s]", "$v_r$ in [m/s]", "$r$ in [rad/s]"]
    fig, axs = plt.subplots(n_states, 1, sharex=True)
    plt.suptitle('Absolute error for each state', size='xx-large')

    for i in range(2):
        axs[i].plot(np.linspace(0, number_of_steps, len(error_fossen_gps[:,i])), error_fossen_gps[:,i], label="Error Fossen", color='red')
        axs[i].plot(np.linspace(0, number_of_steps, len(error_simple_gps[:,i])), error_simple_gps[:,i], label="Error Simple", color='green')
    axs[2].plot(np.linspace(0, number_of_steps, len(error_fossen_imu[:,0])-8), error_fossen_imu[8:,0], label="Error Fossen", color='red')
    axs[2].plot(np.linspace(0, number_of_steps, len(error_simple_imu[:,0])-8), error_simple_imu[8:,0], label="Error Simple", color='green')
    axs[3].plot(np.linspace(0, number_of_steps, len(error_fossen_gps[:,2])), error_fossen_gps[:,2], label="Error Fossen", color='red')
    axs[3].plot(np.linspace(0, number_of_steps, len(error_simple_gps[:,2])), error_simple_gps[:,2], label="Error Simple", color='green')
    axs[4].plot(np.linspace(0, number_of_steps, len(error_fossen_gps[:,3])), error_fossen_gps[:,3], label="Error Fossen", color='red')
    axs[4].plot(np.linspace(0, number_of_steps, len(error_simple_gps[:,3])), error_simple_gps[:,3], label="Error Simple", color='green')
    axs[5].plot(np.linspace(0, number_of_steps, len(error_fossen_imu[:,1])-1), error_fossen_imu[1:,1], label="Error Fossen", color='red')
    axs[5].plot(np.linspace(0, number_of_steps, len(error_simple_imu[:,1])-1), error_simple_imu[1:,1], label="Error Simple", color='green')
    
    for i in range(6):
        # axs[i].set_xlabel("$IMU-steps$", fontsize=14)
        axs[i].set_ylabel(labels[i], fontsize=14)
        # axs[i].legend(loc='center right')
    handles, labels = axs[4].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=14)
    plt.xlabel("$IMU-steps$", fontsize=14)
    plt.suptitle('Absolute error for each state', size='xx-large', weight='bold')


# fig, axs = plt.subplots(2, 1)
# plt.suptitle('Position: Estimated vs GPS', size='xx-large')
# axs[0].plot(xp_f[:,0], xp_f[:,1], label="Fossen estimated")
# axs[0].plot(GPS_ENU[:,1], GPS_ENU[:,2], label="GPS")
# axs[0].legend(loc='center right')
# axs[1].plot(xp_s[:,0], xp_s[:,1], label="Jan/Noa estimated")
# axs[1].plot(GPS_ENU[:,1], GPS_ENU[:,2], label="GPS")
# axs[1].legend(loc='center right')

###########################################
# # Plotting the actual covariances
labels = ["$StdD_x$", "$StdD_y$", "$StdD_{\Theta}$", "$StdD_{u_r}$", "$StdD_{v_r}$", "$StdD_r$"]
fig, axs = plt.subplots(n_states,1, sharex=True)
plt.suptitle('Standard Deviation', size='xx-large', weight='bold')
if option==1:
    orientation, angular_vel, linear_acc, pos, vel = extract_covariance(path)
    axs[0].plot(np.linspace(0,number_of_steps, len(pos[:,0])), pos[:,1], label="IMU/GPS", color='blue')
    axs[1].plot(np.linspace(0,number_of_steps, len(pos[:,0])), pos[:,2], label="IMU/GPS", color='blue')
    axs[2].plot(np.linspace(0,number_of_steps, len(orientation[:,0])), orientation[:,3], label="IMU/GPS", color='blue') # Psi
    axs[3].plot(np.linspace(0,number_of_steps, len(vel[:,0])), vel[:,1], label="IMU/GPS", color='blue')
    axs[4].plot(np.linspace(0,number_of_steps, len(vel[:,0])), vel[:,2], label="IMU/GPS", color='blue')
    axs[5].plot(np.linspace(0,number_of_steps, len(angular_vel[:,0])), angular_vel[:,3], label="IMU/GPS", color='blue') # xdot

# Plotting the predicted standard deviation
for i in range(6):
    if option==1:
        axs[i].plot(np.arange(0, number_of_steps), diag_s[:, i], label="Estimated Simple", color='green')
        axs[i].plot(np.arange(0, number_of_steps), diag_f[:, i], label="Estimated Fossen", color='red')
    elif option==2:
        axs[i].plot(np.arange(0, number_of_steps), diag_s[:, i], label="Estimated Simple", color='green')
    # axs[i].set_xlabel("$IMU-steps$")
    axs[i].set_ylabel(labels[i], fontsize=14)
    # axs[i].legend(loc='center right')
plt.xlabel('$IMU-steps$', fontsize=14)
handles, labels = axs[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', fontsize=14)



plt.show()

