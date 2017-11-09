import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.linalg import norm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import math
from matplotlib import animation
import pdb
import scipy.io as sio
import plotly
plotly.tools.set_credentials_file(username='tonyzhang', api_key='6zoZzkCL9hqQSWDOj0xq')
import plotly.plotly as py
from plotly.graph_objs import *
import matplotlib.patches as patches
from numpy import diff, where, split

class Load_Pose():

    def __init__(self, directory):
        with open(directory+'/output/pose.json', 'r') as f:
            POSE = json.load(f)
        self.bscores = np.array(POSE['bscores'])
        self.bbox = np.array(POSE['bbox'])
        self.kp_confidence = np.array(POSE['scores'])  # keypoint confidence scores
        self.keypoints = np.array(POSE['keypoints'])  # locations of 7 points characterizing mouse


class VideoCapture():

    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total number of frames
        self.TRIALS = []
        self.LED_flash = []

    def capture_frame(self,frame_N):
        self.cap.set(1,frame_N)
        ret, frame = self.cap.read()
        return frame

    def compute_first_trial_frame(self): # frame = single RGB frame from the video
        threshold = 130  # calibrated based on video: Animal4_LearnWNandLight_20171028T202815
        print('Frames to process: ' + str(self.frames))
        for i in range(self.frames):
            if i % 100 == 0:
                print('Progress: '+str(i))
            frame = self.capture_frame(i)
            L_brightness = self.compute_luminance(frame[145:250, 535:610])  # left cue block
            # print(L_brightness)
            R_brightness = self.compute_luminance(frame[353:458, 535:610])  # right cue block
            # print(R_brightness)
            if L_brightness > threshold:
                self.TRIALS.append([i, 0]) # i = frame, 0 = left
                break
            elif R_brightness > threshold:
                self.TRIALS.append([i, 1]) # i = frame, 1 = right
                break

    def compute_visual_CUE_times(self): # frame = single RGB frame from the video
        threshold = 150 ## test value
        trial_max = 200
        print('Frames to process: ' + str(self.frames))
        repeat = False
        for i in range(self.frames):
            if i % 100 == 0:
                print('Progress: '+str(i))
            frame = self.capture_frame(i)
            L_brightness = self.compute_luminance(frame[145:250, 535:610])  # left cue block
            R_brightness = self.compute_luminance(frame[353:458, 535:610])  # right cue block
            if L_brightness > threshold:
                if repeat == False:
                    self.TRIALS.append([i, 0]) # i = frame, 0 = left
                    print('Frame: ' + str(i) + '\nTrial: Left')
                    if len(self.TRIALS) == trial_max: break
                    repeat = True
            elif R_brightness > threshold:
                if repeat == False:
                    self.TRIALS.append([i, 1]) # i = frame, 1 = right
                    print('Frame: ' + str(i) + '\nTrial: Right')
                    if len(self.TRIALS) == trial_max: break
                    repeat = True
            else: repeat = False
        self.TRIALS = np.array(self.TRIALS)

    def compute_LED_times(self):
        threshold = 150
        print('Frames to process: ' + str(self.frames))
        i = 0
        while i < self.frames:
            if i % 100 == 0:
                print('Progress: '+str(i))
            frame = self.capture_frame(i)
            LED_brightness = self.compute_luminance(frame[310:360, 705:750]) # position of LED. May need adjusting.
            if LED_brightness > threshold:
                self.LED_flash.append(i) # i = frame, 0 = left
                print('Frame: ' + str(i))
                i += 6
            i += 1
        self.LED_flash = np.array(self.LED_flash)

    def compute_LED_luminence(self):
        self.all_LED_lum = []
        print('Frames to process: ' + str(self.frames))
        for i in range(self.frames):
            if i % 100 == 0:
                print('Progress: '+str(i))
            frame = self.capture_frame(i)
            LED_brightness = self.compute_luminance(frame[310:360, 705:750]) # position of LED. May need adjusting.
            self.all_LED_lum.append(LED_brightness)
        self.all_LED_lum = np.array(self.all_LED_lum)
        np.save('all_LED_lum', self.all_LED_lum)

    def compute_luminance(self, block):
        avg_brightness = np.average(block)
        return avg_brightness
        # block is a RGB patch in shape (x-dim, y-dim, 3)

    def show_clip(self, start, end, interval = 1000/30, repeat = False, save = False):
        ims = []
        fig = plt.figure("Animation")
        ax = fig.add_subplot(111)
        ax.axis('off')
        print('Animating..')
        for i in range(start, end+1):
            img = Video.capture_frame(i)
            frame_i = ax.imshow(img)
            label = ax.annotate(i, (10,40), color = 'white', size = 15)
            ims.append([frame_i, label])
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        Anim = animation.ArtistAnimation(fig, ims, interval = interval, blit = True, repeat = repeat)
        if save: Anim.save(str(start) + '_' + str(end) + '.mp4')
        return Anim


class Smooth_Keypoints():

    def __init__(self, Pose, window):
        self.window = window
        self.keypoints = Pose.keypoints
        self.kp_confidence = Pose.kp_confidence
        self.shift = self.window // 2

    def Weighted_Mov_Avg(self):
        window = self.window
        keypoints = self.keypoints
        kp_confidence = self.kp_confidence
        frames = np.shape(keypoints)[0]
        output = np.zeros((frames - window + 1, 2, 7))
        for i in range(frames - window):
            seq_i = keypoints[i:i + window]
            movavg_i_x = np.average(seq_i[:, :, 0, :], axis=0, weights=kp_confidence[i:i + window])
            movavg_i_y = np.average(seq_i[:, :, 1, :], axis=0, weights=kp_confidence[i:i + window])
            output[i, 0], output[i, 1] = movavg_i_x, movavg_i_y
        return output

    def Remove_Anomaly_Frames(self): # hard code in frame removal given change in xy coordinates greater than threshold
        keypoints = self.keypoints
        frames = np.shape(keypoints)[0]
        self.all_euc_dist = np.zeros(frames-1)
        dist_threshold = 600
        for i in range(1,frames):
            # compare current frame one previous frame
            kp_i = self.keypoints[i-1:i+1]
            # compute euclidean distance two frames
            euc_dist = norm(kp_i[1,0] - kp_i[0,0])
            self.all_euc_dist[i-1] = euc_dist

            if euc_dist > dist_threshold:
                # pass
                keypoints[i] = self.keypoints[i-1]
        return keypoints

        self.keypoints = keypoints

        # return all_euc_dist


class Analysis():

    # compute low dimensional info (centroid x-y coordinates & orientation in angles) for cluster analysis

    def __init__(self, keypoints, smoothing): # pass smoothed keypoints in here
        self.keypoints = keypoints
        self.frames = np.shape(keypoints)[0]
        self.shift = smoothing.shift

    def compute_centroid(self):
        self.centroids = np.average(self.keypoints, axis = 2)
        return self.centroids

    def compute_orientation(self):
        self.compute_centroid()
        nose_positions = self.keypoints[:,:,0]
        # connect centroid to keypoints. Output list of angles (taking centroid as origin), one for each frame.
        centroid_origin = nose_positions - self.centroids
        x = centroid_origin[:,0]
        y = centroid_origin[:,1]
        self.orientations = np.arctan2(y, x) * 180 / np.pi

    def tSNE(self, behavior_window, label): # behavior window = size of the window for analysis (in number of frames)
        self.compute_orientation()
        centroids = self.centroids
        orientations = self.orientations
        end_frame = self.frames - behavior_window
        step = behavior_window//2
        data =self.compile_data_tSNE_positions(end_frame, step, behavior_window)
        self.embedded = TSNE(n_components=2, verbose = True).fit_transform(data) # fit tSNE (sklearn)
        self.plot_tSNE(label, behavior_window, end_frame, step)

    def tSNE_only_orientation_and_velocity(self, behavior_window, label, perplexity):
        self.compute_speed_from_centroids()
        self.compute_orientation()
        orientations = self.orientations
        end_frame = self.frames - behavior_window
        step = behavior_window//2
        data = self.compile_data_tSNE_orientations(end_frame, step, behavior_window, orientations)
        self.embedded = TSNE(n_components=2, verbose = True, perplexity = perplexity).fit_transform(data) # fit tSNE
        self.plot_tSNE(label, behavior_window, end_frame, step)

    def tSNE_test_intertrial(self, behavior_window, trial_history, speed, padding, label = True): # TEST INTERTRIAL TSNE. group with other functions
        ''' positions only '''
        self.compute_orientation()
        centroids = self.centroids
        orientations = self.orientations
        end_frame = self.frames - behavior_window
        step = behavior_window//2
        if speed:
            self.compute_speed_from_centroids()
            (data, behavior_window) = self.compile_intertrial_data_dimreduction_orientation_speed(trial_history, step, behavior_window, orientations)
        else:
            (data, behavior_window) = self.compile_intertrial_data_dimreduction_positions(trial_history, step, behavior_window,
                                                                                  orientations, padding)
        self.embedded = TSNE(n_components=2, verbose = True).fit_transform(data) # fit tSNE (sklearn)
        self.plot_tSNE_intertrial(label, behavior_window, trial_history, step)

    def PCA_intertrial(self, behavior_window, label, trial_history, speed, padding): # TEST INTERTRIAL TSNE. group with other functions
        ''' positions only '''
        self.compute_orientation()
        centroids = self.centroids
        orientations = self.orientations
        end_frame = self.frames - behavior_window
        step = behavior_window//2
        if speed:
            self.compute_speed_from_centroids()
            (data, behavior_window) = self.compile_intertrial_data_dimreduction_orientation_speed(trial_history, step, behavior_window, orientations)
        else:
            (data, behavior_window) = self.compile_intertrial_data_dimreduction_positions(trial_history, step, behavior_window,
                                                                                  orientations, padding)
        self.embedded = PCA(n_components=2).fit_transform(data) # fit tSNE (sklearn)
        self.plot_PCA_intertrial(label, behavior_window, trial_history, step)

    def compile_data_tSNE_positions(self, end_frame, step, behavior_window): # compile all data from all frames
        data = np.zeros((np.size(range(0, end_frame, step)), 3 * behavior_window))
        for count, frame in enumerate(range(0, end_frame, step)):
            centroids_flattened = np.ravel(self.centroids[frame:frame+behavior_window])
            orientations_flattened = self.orientations[frame:frame+behavior_window]
            data[count] = np.concatenate((centroids_flattened, orientations_flattened))
        return data

    def compile_data_tSNE_orientations(self, end_frame, step, behavior_window, orientations): # compile all data from all frames
        data = np.zeros((np.size(range(0,end_frame,step)), 2 * behavior_window))
        for count, frame in enumerate(range(0, end_frame, step)):
            speeds_flattened = self.speeds[frame:frame+behavior_window]
            orientations_flattened = orientations[frame:frame + behavior_window]
            data[count] = np.concatenate((speeds_flattened, orientations_flattened))
        return data

    # def compile_intertrial_data_dimreduction_positions(self, trial_history, step, behavior_window, orientations): # only compile relevant features after
    #     inter_trial_durations = trial_history[1:-1,0] - trial_history[0:-2,0]
    #     min_trial_duration = np.min(inter_trial_durations)
    #     behavior_window = min_trial_duration
    #     data = np.zeros((trial_history.shape[0], 3 * behavior_window))
    #     for i in range(data.shape[0]):
    #         frame = trial_history[i,0] - self.shift # THIS IS DONE B/C OF SMOOTHING MISMATCH
    #         centroids_flattened = np.ravel(self.centroids[frame:frame+behavior_window])
    #         orientations_flattened = orientations[frame:frame+behavior_window]
    #         data[i] = np.concatenate((centroids_flattened, orientations_flattened))
    #     return data, behavior_window

    def compile_intertrial_data_dimreduction_positions(self, trial_history, step, behavior_window, orientations, padding): # only compile relevant features after

        inter_trial_durations = trial_history[1:,0] - trial_history[0:-1,0]
        last_trial_duration = self.frames - trial_history[-1, 0]
        inter_trial_durations = np.append(inter_trial_durations, last_trial_duration)
        min_trial_duration = np.min(inter_trial_durations)
        if padding == False:
            behavior_window = min_trial_duration
        data = np.zeros((trial_history.shape[0], 3 * behavior_window))

        if padding:
            assert behavior_window > min_trial_duration
            for i in range(data.shape[0]):
                frame = trial_history[i,0] - self.shift # THIS IS DONE B/C OF SMOOTHING MISMATCH
                trial_duration = inter_trial_durations[i]
                if trial_duration < behavior_window:
                    centroids_flattened = np.ravel(self.centroids[frame:frame+trial_duration])
                    orientations_flattened = orientations[frame:frame+trial_duration]
                    data[i, :len(centroids_flattened)] = centroids_flattened
                    data[i, behavior_window*2:behavior_window*2+len(orientations_flattened)] = orientations_flattened
                else:
                    centroids_flattened = np.ravel(self.centroids[frame:frame+behavior_window])
                    orientations_flattened = orientations[frame:frame+behavior_window]
                    data[i] = np.concatenate((centroids_flattened, orientations_flattened))
        else: # no padding
            for i in range(data.shape[0]):
                frame = trial_history[i,0] - self.shift # THIS IS DONE B/C OF SMOOTHING MISMATCH
                centroids_flattened = np.ravel(self.centroids[frame:frame+behavior_window])
                orientations_flattened = orientations[frame:frame+behavior_window]
                data[i] = np.concatenate((centroids_flattened, orientations_flattened))
        return data, behavior_window

    def compile_intertrial_data_dimreduction_orientation_speed(self, trial_history, step, behavior_window, orientations): # only compile relevant features after
        inter_trial_durations = trial_history[1:,0] - trial_history[0:-1,0]
        min_trial_duration = np.min(inter_trial_durations)
        behavior_window = min_trial_duration
        data = np.zeros((trial_history.shape[0], 2 * behavior_window))
        for i in range(data.shape[0]):
            frame = trial_history[i,0] - self.shift # THIS IS DONE B/C OF SMOOTHING MISMATCH
            speeds_flattened = self.speeds[frame:frame + behavior_window]
            orientations_flattened = orientations[frame:frame+behavior_window]
            data[i] = np.concatenate((speeds_flattened, orientations_flattened))
        return data

    def plot_tSNE(self, label, behavior_window, end_frame, step):
        embedded = self.embedded
        print('Plotting..')
        plt.figure()
        plt.title('t-SNE (window = '+str(behavior_window)+')')
        plt.scatter(embedded[:,0], embedded[:,1], color = 'red', s = 5)
        if label:
            for count, frame in enumerate(range(0, end_frame, step)):
                plt.annotate(frame,(embedded[count,0], embedded[count,1]), fontsize = 6)

    def plot_tSNE_intertrial(self, label, behavior_window, trial_history, step):
        embedded = self.embedded
        print('Plotting..')
        plt.figure()
        plt.title('t-SNE Inter-trial (window = '+str(behavior_window)+')')
        trial_types = trial_history[:,1]
        frames = trial_history[:,0]
        plt.scatter(embedded[:,0], embedded[:,1], cmap = 'winter', c = trial_types)
        if label:
            for count, frame in enumerate(frames):
                plt.annotate(frame,(embedded[count,0], embedded[count,1]), fontsize = 6)
                ''' note: this annotation is for the frames in the VIDEO, NOT indices of the centroids / orientations! 
                Remember to subtract by shift '''

    def plot_PCA_intertrial(self, label, behavior_window, trial_history, step):
        embedded = self.embedded
        print('Plotting..')
        plt.figure()
        plt.title('PCA Inter-trial (window = '+str(behavior_window)+')')
        trial_types = trial_history[:,1]
        frames = trial_history[:,0]
        plt.scatter(embedded[:,0], embedded[:,1], cmap = 'winter', c = trial_types)
        if label:
            for count, frame in enumerate(frames):
                plt.annotate(frame,(embedded[count,0], embedded[count,1]), fontsize = 6)
                ''' note: this annotation is for the frames in the VIDEO, NOT indices of the centroids / orientations! 
                Remember to subtract by shift '''

    def compute_speed_from_centroids(self):
        centroids = self.centroids
        speeds = np.zeros(self.frames-1)

        for i in range(1,self.frames):
            euc_dist = norm(centroids[i] - centroids[i-1])
            speed_i = euc_dist * 30 # unit: euclidean distance / second
            speeds[i - 1] = speed_i

        self.speeds = speeds

    def compute_horizontal_length(self):
        pass


class LED_Sync(): # this class outputs the trials' start and end frames in the video
    # point of synchrony = first cue light as detected in function 'compute_first_trial_frame'
    def __init__(self, file):
        self.LED_timestamps = sio.loadmat(file)['LED_timeDelay']
        self.frame_rate = 30 # per second
        self.LED_frames = self.convert_to_frames(self.LED_timestamps)

    def convert_to_frames(self, timestamps): # convert matfile timestamps to video frames
        frames = timestamps * self.frame_rate
        frames = np.ceil(frames)
        return frames

    def comparison_plots(self, Video, shift):
        plt.figure(figsize = (12,1.5))
        plt.plot(Video.all_LED_lum)
        shifted_LED_frames = self.LED_frames + shift
        plt.scatter(shifted_LED_frames, 56*np.ones(self.LED_frames.size), color = 'red', s = 2)
        
    def comparison_plot(self, LED_lum, title, shift = 0):
        '''
        plots all LED luminance (not detection) against the actual timestamps from the mat files,
        then plot it all on pylot (login to see changes)
        '''
        data = []
        max_LED_lum = max(LED_lum)
        LED_frames_zeroed = self.LED_frames - self.LED_frames[0,0]
        for i in range(LED_frames_zeroed.shape[0]):
            trace = Scatter(x = LED_frames_zeroed[i] + shift,
                            y = [max_LED_lum+1, max_LED_lum+1],
                            line = dict(color = ('rgb(205, 12, 24)')))
            data.append(trace)
        trace = Scatter(y=LED_lum, line = dict(color = 'black'))
        data.append(trace)
        layout = Layout(
            title=title,
            xaxis=dict(title='Frames'),
            yaxis=dict(title='Average luminance'))
        py.iplot(Figure(data=data, layout=layout), filename=title)

    def extract_flashframes_from_luminance(self, LED_lum):
        threshold = 150 # this threshold MUST be high. Otherwise, cannot distinguish b/w 2 consecutive lights
        LED_flashes = []
        for i, lum_i in enumerate(LED_lum):
            if lum_i > threshold:
                LED_flashes.append(i)
        # Group these into trials
        LED_flashes_bytrial = split(LED_flashes, where(diff(LED_flashes)>1)[0]+1)
        return LED_flashes_bytrial

    def align_LED_video_with_matfile(self, LED_lum, trial1_in_video_that_correspond_to_matfile):
        LED_flashes_bytrial = self.extract_flashframes_from_luminance(LED_lum)
        first_trial, last_trial = LED_flashes_bytrial[0][0], LED_flashes_bytrial[-1][0]
        diff_frames = last_trial - first_trial
        diff_times = self.LED_timestamps[-1, 0] - self.LED_timestamps[trial1_in_video_that_correspond_to_matfile-1, 0]
        callibrated_frame_rate = diff_frames / diff_times
        self.frame_rate = callibrated_frame_rate  # update frame rate
        self.LED_frames = self.convert_to_frames(self.LED_timestamps) # recompute frames from timestamps
        self.shift = self.LED_frames[-1,0] - LED_flashes_bytrial[-1][0]


class Plot(): # all plot related functions

    def __init__(self, smoothing, Video):

        self.shift = smoothing.shift
        self.Video = Video

    def keypoints_with_confidence(self, keypoints, confidence):
        # this function needs SHIFT added
        for i in range(frames):
            x_i = keypoints[i, 0, 0, :]
            y_i = keypoints[i, 0, 1, :]
            con_score_i = confidence[i, 0]
            plt.ion()
            plt.axis((0, 640, 660, 0))

            plt.imshow(self.Video.capture_frame(i))  # plot video

            plt.scatter(x_i[0], y_i[0], cmap='winter', c=con_score_i[0], marker='$H$')  # head
            plt.scatter(x_i[1], y_i[1], cmap='winter', c=con_score_i[1], marker='$L$')  # left forelimb
            plt.scatter(x_i[2], y_i[2], cmap='winter', c=con_score_i[2], marker='$R$')  # right forelimb
            plt.scatter(x_i[3], y_i[3], cmap='winter', c=con_score_i[3], marker='$N$')  # neck
            plt.scatter(x_i[4], y_i[4], cmap='winter', c=con_score_i[4], marker='$L$')  # left hindlimb
            plt.scatter(x_i[5], y_i[5], cmap='winter', c=con_score_i[5], marker='$R$')  # right hindlimb
            plt.scatter(x_i[6], y_i[6], cmap='winter', c=con_score_i[6], marker='$T$')  # trail

            plt.pause(0.01)
            plt.clf()

    def keypoints_only(self, keypoints, start, duration):
        shift = self.shift
        for i in range(start,start+duration):
            x_i = keypoints[i, 0, :]
            y_i = keypoints[i, 1, :]
            plt.ion()
            plt.axis((0, 640, 660, 0))
            plt.title(i)
            plt.imshow(self.Video.capture_frame(i + shift))  # plot video frame

            plt.scatter(x_i[0], y_i[0], color='red', marker='$N$')  # nose
            plt.scatter(x_i[1], y_i[1], color='blue', marker='$L$')  # left forelimb
            plt.scatter(x_i[2], y_i[2], color='blue', marker='$R$')  # right forelimb
            plt.scatter(x_i[3], y_i[3], color='white', marker='$N$')  # neck
            plt.scatter(x_i[4], y_i[4], color='green', marker='$L$')  # left hindlimb
            plt.scatter(x_i[5], y_i[5], color='green', marker='$R$')  # right hindlimb
            plt.scatter(x_i[6], y_i[6], color='white', marker='$T$')  # tail

            plt.pause(0.001)
            plt.clf()

    def centroids_only(self, centroids):
        for i in range(frames):
            x_i = centroids[i, 0]
            y_i = centroids[i, 1]
            plt.ion()
            plt.axis((0, 640, 660, 0))
            plt.imshow(self.Video.capture_frame(i))  # plot video
            plt.scatter(x_i, y_i, color='red', marker='$*$')  # head

            plt.pause(0.01)
            plt.clf()

    def plot_all_centroids(self, centroids):
        plt.figure()
        plt.imshow(Video.capture_frame(frames))
        plt.axis((0, 640, 660, 0))
        plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'o', color = 'red', s = 0.15)

    def plot_only_nose(self, keypoints):
        plt.figure()
        plt.imshow(Video.capture_frame(frames))
        plt.axis((0, 640, 660, 0))
        plt.scatter(keypoints[:, 0, 0], keypoints[:, 1, 0], marker = 'o', color = 'red', s = 0.15)

    def plot_centroid_and_nose(self, centroids, keypoints, start, duration):
        shift = self.shift
        # only plot one figure mode
        if start.__class__ == int:
            plt.figure()
            for i in range(start,start+duration):
                x1 = centroids[i - shift, 0]
                y1 = centroids[i - shift, 1]
                x2 = keypoints[i - shift, 0, 0]
                y2 = keypoints[i - shift, 1, 0]
                plt.ion()
                plt.axis((0, 640, 660, 0))
                plt.title('Frame in video: '+str(i))
                plt.imshow(self.Video.capture_frame(i))  # plot video frame

                plt.plot(x1, y1, x2, y2, marker='o')

                plt.pause(0.001)
                plt.clf()
        else: # MULTI-VIDEO SIMULTANEOUS PLOTTING. Start = list
            # fig, axarr = plt.subplots(len(start))
            skip_frames = 3 # display 1 frame in place of every 3 frames
            box = math.sqrt(len(start))
            box = math.ceil(box)
            plt.figure(figsize = (11,11))
            gs1 = gridspec.GridSpec(box,box)
            gs1.update(wspace=0, hspace=0)
            for i in range(0, duration, skip_frames):
                plt.clf()
                for j in range(len(start)):
                    ax1 = plt.subplot(gs1[j])
                    plt.axis('off')
                    frame = start[j]+i
                    ax1.imshow(self.Video.capture_frame(frame))
                    x1 = centroids[start[j] + i - shift, 0]
                    y1 = centroids[start[j] + i - shift, 1]
                    x2 = keypoints[start[j] + i - shift, 0, 0]
                    y2 = keypoints[start[j] + i - shift, 1, 0]
                    plt.plot(x1, y1, x2, y2, marker='o')
                    plt.annotate(start[j], (4, 45), fontsize=12, color = 'white')
                    plt.annotate(frame, (4, 85), fontsize=9, color='white')
                plt.pause(0.0001)

    def animate_multi_view_beta(self, start, duration, centroids, keypoints, skip_frames = 1,
                           interval = 1000/30, repeat = False, save = False, only_keypoints = False,
                                kp_confidence = None, smoothed = True, bounding_box = None, plot_bbox = False):
        shift = self.shift
        box = math.ceil(math.sqrt(len(start)))
        fig = plt.figure(figsize = (10,10))
        gs1 = gridspec.GridSpec(box,box)
        gs1.update(wspace=0, hspace=0)
        ims = []
        print('Animating..')
        for i in range(0, duration, skip_frames):
            ims_i = []
            for j in range(len(start)):
                # CHOOSE SUBPLOT
                ax = plt.subplot(gs1[j])
                ax.axis('off')
                # PLOT FRAME
                img_i_j = self.Video.capture_frame(start[j]+i)
                frame_i_j = ax.imshow(img_i_j)
                # LABELS
                label_start_frame = ax.annotate(start[j], (4, 45), fontsize=12, color = 'white')
                label_curr_frame = ax.annotate(start[j]+i, (4, 85), fontsize=9, color='white')

                if only_keypoints:
                    # PLOT ALL 7 KEYPOINTS
                    if smoothed:
                        x_i = keypoints[start[j] + i - shift, 0, :]
                        y_i = keypoints[start[j] + i - shift, 1, :]
                        plot_kpoints = ax.scatter(x_i, y_i, c='cyan')
                    else:
                        x_i = keypoints[start[j] + i, 0, 0, :]
                        y_i = keypoints[start[j] + i, 0, 1, :]
                        con_score_i = kp_confidence[i, 0]
                        plot_kpoints = ax.scatter(x_i, y_i, cmap = plt.cm.coolwarm, c = con_score_i)
                    # COMBINE ALL ACTIONS
                    ims_i.extend((frame_i_j, plot_kpoints, label_start_frame, label_curr_frame))
                else:
                    # PLOT CENTROID / NOSE
                    x1, y1 = centroids[start[j] + i - shift, 0], centroids[start[j] + i - shift, 1]
                    x2, y2 = keypoints[start[j] + i - shift, 0, 0], keypoints[start[j] + i - shift, 1, 0]
                    centroid, = ax.plot(x1, y1, marker='o', c='cyan')
                    nose, = ax.plot(x2, y2, marker='o', c='lime')
                    # COMBINE ALL ACTIONS
                    ims_i.extend((frame_i_j, centroid, nose, label_start_frame, label_curr_frame))
                ####
                # if plot_bbox:
                #     x1, y1 = bounding_box[start[j] + i, 0], bounding_box[start[j] + i, 1]
                #     x2, y2 = bounding_box[start[j] + i, 2], bounding_box[start[j] + i, 3]
                #     w = x2 - x1
                #     h = y2 - y1
                #     rect = patches.Rectangle((x1, y1), w, h,linewidth=1,edgecolor='r',facecolor='none')
                #     bbox = ax.add_patch(rect)
                #     ims_i.append(bbox)
                # ####
            ims.append(ims_i)
        Anim = animation.ArtistAnimation(fig, ims, interval = interval, blit = True, repeat = repeat, repeat_delay=3000)
        if save: Anim.save(str(start) + '_' + str(duration) + '.mp4')
        return Anim

    def animate_multi_view(self, start, duration, centroids, keypoints, skip_frames = 1,
                           interval = 1000/30, repeat = False, save = False):
        shift = self.shift
        box = math.ceil(math.sqrt(len(start)))
        fig = plt.figure(figsize = (10,10))
        gs1 = gridspec.GridSpec(box,box)
        gs1.update(wspace=0, hspace=0)
        ims = []
        print('Animating..')
        for i in range(0, duration, skip_frames):
            ims_i = []
            for j in range(len(start)):
                # CHOOSE SUBPLOT
                ax = plt.subplot(gs1[j])
                ax.axis('off')
                # PLOT FRAME
                img_i_j = self.Video.capture_frame(start[j]+i)
                frame_i_j = ax.imshow(img_i_j)
                # PLOT CENTROID / NOSE
                x1 = centroids[start[j] + i - shift, 0]
                y1 = centroids[start[j] + i - shift, 1]
                x2 = keypoints[start[j] + i - shift, 0, 0]
                y2 = keypoints[start[j] + i - shift, 1, 0]
                centroid, = ax.plot(x1, y1, marker='o', c = 'blue')
                nose, = ax.plot(x2, y2, marker='o', c = 'green')
                # LABELS
                label_start_frame = ax.annotate(start[j], (4, 45), fontsize=12, color = 'white')
                label_curr_frame = ax.annotate(start[j]+i, (4, 85), fontsize=9, color='white')
                # COMBINE ALL ACTIONS
                ims_i.extend((frame_i_j, centroid, nose, label_start_frame, label_curr_frame))
            ims.append(ims_i)
        # Ac
        Anim = animation.ArtistAnimation(fig, ims, interval = interval, blit = True, repeat = repeat, repeat_delay=3000)
        if save: Anim.save(str(start) + '_' + str(duration) + '.mp4')
        return Anim

