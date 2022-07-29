# ray-tracking-2022

This repo is for tracking eagle rays (or potentially other marine life) from overhead drone footage. The system uses blob detection to get measurements for the ray from video frames and sends these measurements to a particle filter to track the ray over time. As the tracking runs, the measurements and particles in the particle filter are conveniently displayed in a pop-up window and data from the tracking is saved when the video is over. 

There are two scripts, ray_tracker and particle_filter. ray_tracker is the file to run to analyze a video. To do that, first move the videos to analyze to the videos folder or another location. Then, update all of the video-specific parameters at the top of ray_tracker (like frames per second and the pixels to meters conversion) and all of the filepaths for the video and where to save the output video and csv. 

Before anything will run, all of the python libraries used in the project (numpy, matplotlib, opencv, and pandas) need to be installed. 

When ray_tracker is run, a window will pop up with the first frame of the video. Click on the ray and press "g" for the system to run. At any time, press "p" to pause, click on the ray again, and press "g" to go again. This could help if the particles move off the ray. Or, press "q" to quit. Otherwise, the system will run until the end of the video and then plot some charts. Based on what is set in the parameters at the top of the ray_tracker file, a video and csv from the tracking will be saved to the specified locations.
