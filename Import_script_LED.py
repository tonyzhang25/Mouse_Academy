from Import_pose_module import *

name = 'LED_Test_for_Tony_20171029/Light2AFC'

# Pose = Load_Pose(name)
Video = VideoCapture(name+'.mp4') # folder / filename
# Video.compute_visual_CUE_times()

smoothed = Smooth_Keypoints(Pose, window = 10)
# smoothed.Remove_Anomaly_Frames()
smoothed_keypoints = smoothed.Weighted_Mov_Avg()

Post_smooth_analysis = Analysis(smoothed_keypoints, smoothed)
centroids = Post_smooth_analysis.compute_centroid()
# Post_smooth_analysis.tSNE(30, label = False) # 30-50 seems to have the best clusters

# Post_smooth_analysis.tSNE_only_orientation_and_velocity(90, label = True, perplexity = 50) # 30-50 also
trial_history = np.load(name + '/TRIALS_start_times_' + name + '.npy')

# Post_smooth_analysis.tSNE_test_intertrial(60, True, trial_history, speed = False)
# Post_smooth_analysis.tSNE_test_intertrial(80, True, trial_history, speed = False, padding = True)
# Post_smooth_analysis.PCA_intertrial(60, True, trial_history, speed = False, padding = True)

## analyze luminance plots
LED_lum = np.load('all_LED_lum.npy')
Sync = LED_Sync('LED_Test_for_Tony_20171029/LED_matfiles/LED_timestamps_Animal4_LearnLight2AFC_20171027_142047.mat')
Sync.LED_timestamps = Sync.LED_timestamps[1:]
Sync.align_LED_video_with_matfile(LED_lum, trial1_in_video_that_correspond_to_matfile = 1)
Sync.comparison_plot(LED_lum, title = 'LED Sync Aligned (Animal4_LearnLight2AFC_20171027_142047)', shift = 98)

# Plot1.keypoints_only(smoothed_keypoints, 200, 300)
# Plot.keypoints_with_confidence(keypoints, kp_confidence)
# Plot.centroids_only(centroids)
# Plot.plot_all_centroids(centroids)
# Plot.plot_only_nose(smoothed_keypoints)


###### PLOT
Plot1 = Plot(smoothed, Video)

############################ BEHAVIOR CLUSTER VISUALIZATION (centroids & orientation) ##################################

''' RIGHT TRIALS (wrong) '''
Plot1.plot_centroid_and_nose(centroids,smoothed_keypoints, [19276,13244,25253,31366,14713,17023,13427,11971,11765,16433,21961,10764,16836], duration+10) # bottom left cluster

''' LEFT TRIALS (correct). REACTION TIME: slow to fast'''
Plot1.plot_centroid_and_nose(centroids,smoothed_keypoints, [12761,33609,1742,12318,31868,16144,20071,11463,33777,10647,3079,20175,30844,12531,13197], duration) # bottom right cluster

''' LEFT TRIALS (WRONG), then switch sides. Reaction time in switching side: fast to slow (as it sweeps across tTSNE)
switch when compared to cluster above '''
Plot1.plot_centroid_and_nose(centroids,smoothed_keypoints, [3776,19696,17921,6718,33022,10348,28496,4928,18393,3976,7238,1905,1020,5660,8587], duration) # middle green cluster


#########


duration = 80

Plot1.plot_centroid_and_nose(centroids,smoothed_keypoints, [22949,26564,22949,29625], duration) # middle green cluster


########################### speeds and centroids #####################
''' LEFT TRIALS. Go left (correct). Fast to slow in reaction time '''
Plot1.plot_centroid_and_nose(centroids,smoothed_keypoints,
    [], duration) # bottom left cluster, top to bottom

''' RIGHT TRIAL. Go right (correct). Fast to slow. '''
Plot1.plot_centroid_and_nose(centroids,smoothed_keypoints,
    [], duration) # green cluster in the middle

''' RIGHT TRIAL. Go left (wrong), then quickly correct'''
Plot1.plot_centroid_and_nose(centroids,smoothed_keypoints,
    [], duration) # Visualize some green trials in the top right region

Plot1.plot_centroid_and_nose(centroids,smoothed_keypoints,
    [], duration) # Visualize some green trials in the top right region
