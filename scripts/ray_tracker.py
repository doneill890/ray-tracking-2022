
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
norm = np.linalg.norm

# get the optical flow
# get the mean, subtract off

# event click up/ down to pick starting position

# video parameters
# altitude 35 meters
# 4k
# 2160 by 3840 pixels
# 1.16 cm/pixel

IMAGE_HEIGHT = 2160
IMAGE_WIDTH = 3840

SAVE_VIDEO = False
SAVE_DATA = False

PIX_TO_M = 0.0116            # pixels to meters conversion
ORIGINAL_FPS = 30            # frames per second
SAMPLING = 1
FPS = ORIGINAL_FPS/SAMPLING

def process_frame(frame):
    ''' does image processing to get contours (ray, etc.) from the camera frame '''
    original = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    return [original, cnts]

def clark_filter(data):
    ''' the clark filter '''
    alpha = .01             # alpha has to be between 0 and 1
    filtered = [data[0]]
    for i in range(1, len(data)):
        filtered.append(alpha * data[i] + (1 - alpha) * filtered[i-1])
    return filtered

def calculate_data(pos_list):
    x_un = [pos[0] for pos in pos_list]  # unfiltered coordinates
    y_un = [pos[1] for pos in pos_list]

    x = clark_filter(x_un)  # filtered coordinates
    y = clark_filter(y_un)
    
    time = [i*1/FPS for i in range(0, len(pos_list))]
    pos_list = [np.array([x[i], y[i]]) for i in range(0, len(x))]

    vec_dpos = [pos_list[i] - pos_list[i-1] for i in range(1, len(pos_list))]
    linear_velocity = [norm(dpos)*FPS for dpos in vec_dpos]
    #scalar_dpos = [norm(dpos) for dpos in vec_dpos]  # for calculating total distance travelled
    #print("total distance traveled ", sum(scalar_dpos))

    heading = [np.arctan2(dpos[1], dpos[0])*180/np.pi for dpos in vec_dpos]
    heading = clark_filter(heading)
    angular_velocity = [(heading[i] - heading[i-1])*FPS for i in range(1, len(heading))]

    return [time, x, y, linear_velocity, heading, angular_velocity]

def plot(time, x, y, linear_velocity, heading, angular_velocity):

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

    plt.figure(5)
    plt.plot(time, x)
    plt.xlabel("time (s)")
    plt.ylabel("x (m)")

    plt.figure(6)
    plt.plot(time, y)
    plt.xlabel("time (s)")
    plt.ylabel("y (m)")

    plt.show()

def save_data(savepath, t, x, y, lv, h, av):
    avg_velocity = sum(lv)/len(lv)
    avg_angular_velocity = sum(av)/len(av)

    data = {'t': t,
            'x': x,
            'y': y,
            'linear_velocity': lv,
            'heading': h,
            'angular velocity': av,
            'avg_velocity': avg_velocity,
            'avg_angular_velocity': avg_angular_velocity}

    df = pd.DataFrame(data, columns = ['t', 'x', 'y', 'linear_velocity', 'heading',
                                       'angular_velocity', 'avg_velocity', 'avg_angular_velocity'])
    df.to_csv(savepath, index=False)

filename = 'DJI_0586'
#filepath = '/Users/Declan/Desktop/' + filename + '.MP4'
filepath = '/Users/Declan/Desktop/cropped.mp4'

save_name = '/Users/Declan/Desktop/data' + filename + '.csv'

cap = cv2.VideoCapture(filepath)

if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, FPS, (960,540))

pos_list = []

while(True):
    ret, frame = cap.read()
    if (ret and frame is not None):
        image, cnts = process_frame(frame)
        ray_cnt = cnts[0]
        M = cv2.moments(ray_cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = 0
            cY = 0
        coords = np.array([cX, cY])
        s_coords = np.array([coords[0], IMAGE_HEIGHT-coords[1]])
        pos_list.append(s_coords*PIX_TO_M)
        cv2.circle(image, (cX, cY), 15, (320, 159, 22), -1)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        resized = cv2.resize(image, (960, 540))
        cv2.imshow('image', resized)

        if SAVE_VIDEO:
            out.write(resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q is for quit
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

t, x, y, lv, h, av = calculate_data(pos_list)
plot(t, x, y, lv, h, av)

if SAVE_DATA:
    save_data(save_name, t, x, y, lv, h, av)



