from Import_pose_module import *

name = 'Animal1_LearnLight2AFC_20170914T231352'

Pose = Load_Pose(name)
Video = VideoCapture(name+'/'+name+'.mp4') # folder / filename
# Video.compute_visual_CUE_times()
# Video.compute_LED_luminence()

smoothed = Smooth_Keypoints(Pose, window = 10)
# smoothed.Remove_Anomaly_Frames()
smoothed_keypoints = smoothed.Weighted_Mov_Avg()

Post_smooth_analysis = Analysis(smoothed_keypoints, smoothed)
centroids = Post_smooth_analysis.compute_centroid()
# Post_smooth_analysis.tSNE(30, label = False) # 30-50 seems to have the best clusters

# Post_smooth_analysis.tSNE_only_orientation_and_velocity(90, label = True, perplexity = 50) # 30-50 also
trial_history = np.load(name + '/TRIALS_start_times_' + name + '.npy')
Post_smooth_analysis.tSNE_test_intertrial(60, trial_history, speed = False, padding = False, label = True)
# Post_smooth_analysis.PCA_intertrial(60, True, trial_history, speed = False, padding = True)

###### PLOT

Plot1 = Plot(smoothed, Video)

Plot1.animate_multi_view_beta([32710], 10, centroids, smoothed_keypoints, save = True)

# Plot1.keypoints_only(smoothed_keypoints, 200, 300)
# Plot.keypoints_with_confidence(keypoints, kp_confidence)
# Plot.centroids_only(centroids)
# Plot.plot_all_centroids(centroids)
# Plot.plot_only_nose(smoothed_keypoints)

############################ BEHAVIOR CLUSTER VISUALIZATION (centroids & orientation) ##################################

duration = 47

''' RIGHT TRIALS (correct) '''
Plot1.animate_multi_view([32710,17725,17321,14986,31624,19609,14903,18982,33304,11572,24787,31962,3246,12432,12652,16242],
                         duration, centroids, smoothed_keypoints, repeat = False, save = True, interval = 100) # bottom left cluster

''' RIGHT TRIALS (wrong) '''
Plot1.animate_multi_view([19276,13244,25253,31366,14713,17023,13427,11971,11765,16631,16433,20274,21961,10764,22160,16836],
                         duration, centroids, smoothed_keypoints, repeat = True, save = True, interval = 100) # bottom left cluster

''' LEFT TRIALS (correct). REACTION TIME: slow to fast'''
Plot1.animate_multi_view([12761,33609,1742,12318,31868,16144,11666,20071,11463,33777,10647,3079,20175,30844,12531,13197],
                         duration, centroids, smoothed_keypoints, repeat = True, save = True, interval = 100) # bottom right cluster

''' LEFT TRIALS (WRONG), then switch sides. Reaction time in switching side: fast to slow (as it sweeps across tTSNE)
switch when compared to cluster above '''
Plot1.animate_multi_view([3776,19696,17921,6718,33022,10348,28496,4928,18393,3976,7238,1905,2135,1020,5660,8587],
                         duration, centroids, smoothed_keypoints, repeat = False, save = True, interval = 100) # middle green cluster

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
