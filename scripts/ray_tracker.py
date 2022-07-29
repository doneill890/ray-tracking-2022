
import cv2                      # importing packages to use
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import particle_filter as pf
norm = np.linalg.norm

# 577 and 578: 0.83 cm/pix
# 518 and 519: 1.00 cm/pix
# 586: 1.16 cm/pixel


###################################### PARAMETERS TO CHANGE ######################################


SAVE_VIDEO = True               # whether or not you want to save the annotated video
SAVE_DATA = True                # whether or not you want to save data to a csv

IMAGE_HEIGHT = 2160             # video parameters
IMAGE_WIDTH = 3840

PIX_TO_M = 0.0083               # pixels to meters conversion
FPS = 30.0                      # frames per second

video_name = 'DJI_0578'
filepath = '/Users/Declan/Desktop/HMC/ray-tracking-2022/videos/' + video_name + '.MP4'

data_file = '/Users/Declan/Desktop/HMC/ray-tracking-2022/data/' + video_name + '.csv'
video_file = '/Users/Declan/Desktop/HMC/ray-tracking-2022/data/' + video_name + '.avi'


###################################### HELPER FUNCTIONS ######################################


def on_mouse(event,x,y,flags,param):
    ''' helper function for getting mouse clicks '''
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(resized, (x, y), 5, (320, 159, 22), -1)
        mouseX,mouseY = x,y

def get_click(image):
    ''' waits for mouse clicks and returns the coordinates clicked after you press "g" '''
    while(True):
        cv2.imshow('image', image)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('g'):
            print(mouseX, mouseY)
            break
    return np.array([mouseX, mouseY])

def cnt_dist(prev_coords, cnt):
    ''' helper function to get the distance between a blob in an image and the last known coordinates
        of the ray. it is used to sort the blobs to find the blob closest to the last know spot '''
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = 0
        cY = 0
    coords = np.array([cX, cY])
    return norm(prev_coords-coords)

def process_frame(frame, coords):
    ''' does image processing to get contours (dark blobs, like the ray) from the camera frame '''
    original = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (33,33), 0)
    value = 80  # the intiial guess at a good thresholding value
    for i in range(0, 10):  # this loop programmatically selects an ideal thresholding value
        thresh = cv2.threshold(blur, value, 255, cv2.THRESH_BINARY_INV)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=lambda cnt: cnt_dist(coords, cnt))
        try:
            ray_cnt = cnts[0]
            (x,y),radius = cv2.minEnclosingCircle(ray_cnt)
            if radius < 5: value = value - 1
            elif radius > 20: value = value + 1
            else: 
                print(value)
                cv2.circle(original,(int(x),int(y)),int(radius),(0,255,0),2)
                break
        except:
            value = value - 1
    return [original, ray_cnt]

def plot_tail(image, pos_list):
    ''' plots the last 200 positions of the ray onto the video frame '''
    tail_list = pos_list[-200:len(pos_list)]
    size_increment = 5/len(tail_list)
    for i in range(0, len(tail_list)):
        x, y = tail_list[i]/PIX_TO_M
        x = int(x)
        y = int(IMAGE_HEIGHT - y)
        size = int(size_increment*i)
        cv2.circle(image, (x, y), size, (320, 159, 22), -1)
    return image

def plot_particles(image, filter):
    ''' plots the particles (from the particle filter) onto the video frame '''
    for i in range(0, filter._num_particles):
        x, y, w = filter._particles[i]
        x = int(min(max(x, 0), 960))
        y = int(min(max(y, 0), 540))
        cv2.circle(image, (x, y), 2, (0, 0, 100), -1)
    return image

def lowpass_filter(data):
    ''' a lowpass filter to reduce noise in the data '''
    alpha = .1                              # alpha has to be between 0 and 1
    filtered = [data[0]]                    # alpha = 1 for no filtering, more smooth if alpha is closer to 0
    for i in range(1, len(data)):
        filtered.append(alpha * data[i] + (1 - alpha) * filtered[i-1])
    return filtered

def calculate_data(pos_list):
    ''' calculating velocity, heading, etc. from the list of positions '''
    x_un = [pos[0] for pos in pos_list]     # unfiltered coordinates
    y_un = [pos[1] for pos in pos_list]

    x = lowpass_filter(x_un)                # filtered coordinates
    y = lowpass_filter(y_un)
    
    time = [i*1/FPS for i in range(0, len(pos_list))]
    pos_list = [np.array([x[i], y[i]]) for i in range(0, len(x))]

    vec_dpos = [pos_list[i] - pos_list[i-1] for i in range(1, len(pos_list))]
    linear_velocity = [norm(dpos)*FPS for dpos in vec_dpos]
    linear_velocity = lowpass_filter(linear_velocity)

    heading = [np.arctan2(dpos[1], dpos[0])*180/np.pi for dpos in vec_dpos]
    heading = lowpass_filter(heading)
    angular_velocity = [(heading[i] - heading[i-1])*FPS for i in range(1, len(heading))]
    angular_velocity = lowpass_filter(angular_velocity)

    return [time, x, y, linear_velocity, heading, angular_velocity]

def plot(time, x, y, linear_velocity, heading, angular_velocity):
    ''' makes some useful plots '''

    plt.figure(1)
    plt.plot(x, y)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.axis('scaled')

    plt.figure(2)
    plt.plot(time[1:len(time)], linear_velocity)
    plt.xlabel("time (s)")
    plt.ylabel("linear velocity (m/s)")

    plt.figure(3)
    plt.plot(time[1:len(time)], heading)
    plt.xlabel("time (s)")
    plt.ylabel("heading (degrees)")

    plt.figure(4)
    plt.plot(time[2:len(time)], angular_velocity)
    plt.xlabel("time (s)")
    plt.ylabel("angular velocity (degrees/second)")

    plt.show()

def save_data(savepath, t, x, y, lv, h, av):
    ''' saves all the relevant data to a csv in the data folder '''
    avg_velocity = sum(lv)/len(lv)
    avg_angular_velocity = sum(av)/len(av)
    pos_list = [np.array([x[i], y[i]]) for i in range(0, len(x))]
    scalar_dpos = [norm(pos_list[i] - pos_list[i-1]) for i in range(1, len(pos_list))]
    total_dist = sum(scalar_dpos)

    data = {'t': t,
            'x': x,
            'y': y,
            'linear_velocity': lv,
            'heading': h,
            'angular velocity': av,
            'avg_velocity': [avg_velocity],
            'avg_angular_velocity': [avg_angular_velocity],
            'total_dist': [total_dist]}

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.transpose()
    df.to_csv(savepath, index=False)


#################################### PROCESSING STARTS HERE ####################################


cap = cv2.VideoCapture(filepath)                            # getting a video capture for the video to analyze

if SAVE_VIDEO:                                              # setting up a video file to write to if specified
    out = cv2.VideoWriter(video_file,
    cv2.VideoWriter_fourcc(*"MJPG"), 30, (960,540))

pos_list = []                                               # empty list to store the ray position over time

cv2.namedWindow('image')                                    # displaying the first frame to get the first ray position
cv2.setMouseCallback('image', on_mouse)
ret, frame = cap.read()
resized = cv2.resize(frame, (960, 540))

prev_coords = get_click(resized)
filter = pf.ParticleFilter(prev_coords)

while(True):                                                # main loop over all video frames
    ret, frame = cap.read()
    if (ret and frame is not None):
        frame = cv2.resize(frame, (960, 540))
        image, ray_cnt = process_frame(frame, prev_coords)  # find the contour for the ray
        M = cv2.moments(ray_cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else: 
            cX = cY = 0
        filter._predict()                                   # run prediction step for every frame
        coords = np.array([cX, cY])
        if norm(filter._get_mean() - coords) < 30:          # reject outliers
            prev_coords = coords
            filter._correct(coords)                         # only run correction step on good measurements
        else:
            print("measurement rejected")
        mean_p = filter._get_mean()
        mean_p[1] = IMAGE_HEIGHT - mean_p[1]
        pos_list.append(mean_p*PIX_TO_M)
        image = plot_particles(image, filter)               # plot the particles
        #image = plot_tail(image, pos_list)                 # uncomment this to plot the path of the ray
        resized = cv2.resize(image, (960, 540))
        cv2.imshow('image', resized)

        if SAVE_VIDEO:                                      # write the annotated frame if saving a video
            out.write(resized)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):                                 # p to pause
            prev_coords = get_click(resized)
        elif key == ord('q'):                               # q to quit
            break
    else:
        break

cap.release()                                               # close everything
out.release()
cv2.destroyAllWindows()


t, x, y, lv, h, av = calculate_data(pos_list)       # calculate x, y, linear velocity, heading, etc.
plot(t, x, y, lv, h, av)                            # plot all the data

if SAVE_DATA:                                       # save the data to a csv if desired
    save_data(data_file, t, x, y, lv, h, av)        # (specify if you want to at the top of the file)

