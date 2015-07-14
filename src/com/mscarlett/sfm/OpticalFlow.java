package com.mscarlett.sfm;

import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.DMatch;
import org.opencv.imgproc.Imgproc;
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
		
		/*List<DMatch> good_matches_ = new LinkedList<DMatch>();
		MatOfKeyPoint imgpts1 = new MatOfKeyPoint();
		
		for (int x=0;x< uFlow.cols(); x++) {
			for (int y=0; y< uFlow.rows(); y++) {
				double movement = MathUtil.norm(uFlow.get(y,x));
				if (movement < 20 || movement > 100) {
					continue; //discard points that havn't moved
				}
				Point p = new Point(x,y);
				Point p1 = new Point(x+uFlow.get(y,x)[0],y+uFlow.get(y,x)[1]);
				
				if (x%10 == 0 && y%10 == 0) {
					DMatch match = new DMatch(imgpts1.rows()-1,imgpts1.rows()-1,1.0f);
					good_matches_.
					keypoints_1.push_back(KeyPoint(p,1));
					keypoints_2.push_back(KeyPoint(p1,1));
				}
				fullpts1.push_back(KeyPoint(p,1));
				fullpts2.push_back(KeyPoint(p1,1));
			}
		}*/
		
	}
}

