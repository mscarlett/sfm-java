package com.mscarlett.sfm;

import java.util.List;

import org.opencv.core.Mat;
import org.opencv.features2d.DMatch;

public class CameraMatrix {
	
	public final Mat K;
	
	public CameraMatrix() {
		K = new Mat();
	}

	/**
	 * Find the camera matrix K which satisfies x_i = K*X_i for every i, where
	 * x_i is a point on the camera image plane and X_i is the same point in 3D
	 * space.
	 */
	public void findCameraMatrix(Mat K, Mat Kinv, Mat distcoeff,
			Mat F, Mat P,
			Mat P1, List<DMatch> matches, List<CloudPoint> outCloud) {
		
	}
	
	/**
	 * Undistort the camera image using the camera calibration matrix
	 * @param src
	 * @param dst
	 */
	public void undistort(Mat src, Mat dst) {
		
	}
}
