package com.mscarlett.sfm;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.highgui.VideoCapture;

public class SFMLauncher {

	static{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
	
	private final SFMPipeline sfm;
	
	public SFMLauncher() {
		// Obtains structure from images
		sfm = new SFMPipeline();
	}
	
	public void run() {
		// Register the default camera
		VideoCapture cap = new VideoCapture(0);
		cap.open(0);
		
		// Wait for camera to be registered
		int timeOut = 1000;
		int sleepTime = 100;
		
		for (int i = 0; i < timeOut && !cap.isOpened(); i += sleepTime) {
			try {
			    Thread.sleep(sleepTime);
			} catch (InterruptedException e) {
				throw new RuntimeException(e);
			}
 		}

		// Check if video capturing is enabled
		if (!cap.isOpened()) {
			throw new RuntimeException("Video capturing not enabled");
		}

		// Matrix for storing image
		Mat image = new Mat();
		cap.read(image);
		// Frame for displaying image
		ImageFrame frame = new ImageFrame();
		frame.setVisible(true);
        
		// Main loop
		while (true) {
			// Render frame if the camera is still acquiring images
			if (cap.read(image)) {
				// Get accelerometer data
				// Get rotation vector data
				// Get homography between last image and current image
			    sfm.applyAsync(image);
				// Render image
				frame.render(image);
			} else {
				System.out.println("No captured frame -- camera disconnected");
				break;
			}
		}
		
		cap.release();
	}
	
	public static void main(String[] args) {
		new SFMLauncher().run();
	}
}
