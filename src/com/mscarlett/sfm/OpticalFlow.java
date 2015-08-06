package com.mscarlett.sfm;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.video.Video;

public class OpticalFlow {
	
	public final double pyr_scale;
	public final int levels;
	public final int winsize;
	public final int iterations;
	public final int poly_n;
	public final double poly_sigma;
	public final int flags;
	
	public OpticalFlow() {
		this(0.5, 3, 15, 3, 5, 1.2, 0);
	}
	
	public OpticalFlow(double pyr_scale, int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags) {
		this.pyr_scale = pyr_scale;
		this.levels = levels;
		this.winsize = winsize;
		this.iterations = iterations;
		this.poly_n = poly_n;
		this.poly_sigma = poly_sigma;
		this.flags = flags;
	}
	
	public void calcOpticalFlow(Mat img1, Mat img2, Mat uFlow) {
		Video.calcOpticalFlowFarneback(img1, img2, uFlow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	}
	
	
}

