package com.mscarlett.sfm;

import org.opencv.core.Mat;
import org.opencv.highgui.VideoCapture;

public class Main {

	static{
		nu.pattern.OpenCV.loadLibrary();
    }
	
	public static void main(String[] args) throws InterruptedException {
		// Register the default camera
		VideoCapture cap = new VideoCapture(0);
		cap.open(0);
		
		// Wait for camera to be registered
		Thread.sleep(1000);

		// Check if video capturing is enabled
		if (!cap.isOpened()) {
			System.err.println("Video capturing not enabled");
			System.exit(-1);
		}

		// Matrix for storing image
		Mat image = new Mat();
		cap.read(image);
		// Frame for displaying image
		ImageFrame frame = new ImageFrame();
		frame.setVisible(true);
        // Prev image
		Mat lastImage;
		
		FeatureMatching featureMatching = new FeatureMatching();
		
		// Main loop
		while (true) {
			lastImage = image;
			
			// Render frame if the camera is still acquiring images
			if (cap.read(image)) {
				// Get homography between last image and current image
			    featureMatching.match(lastImage, image);
				
				frame.render(image);
			} else {
				System.out.println("No captured frame -- camera disconnected");
				break;
			}
		}
		
		cap.release();
	}
}
