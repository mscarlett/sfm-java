package com.mscarlett.sfm;

import java.util.ArrayList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.KeyPoint;

public class FundamentalMat {

	private final List<Point> mp1Points;
    private final List<Point> mp2Points;
    private final MatOfPoint2f mp1Inliers;
    private final MatOfPoint2f mp2Inliers;
    
    private final double dist1;
    private final double dist2;
    private final double dist3;
    
    public FundamentalMat() {
    	this(1.0, 0.5, 3.0);
    }
    
    public FundamentalMat(double dist1, double dist2, double dist3) {
    	mp1Points = new ArrayList<Point>();
		mp2Points = new ArrayList<Point>();
		
		mp1Inliers = new MatOfPoint2f();
		mp2Inliers = new MatOfPoint2f();
		
		this.dist1 = dist1;
		this.dist2 = dist2;
		this.dist3 = dist3;
    }
	
	public Mat getF(MatOfKeyPoint keypoints1, MatOfKeyPoint keypoints2, MatOfDMatch matches, MatOfPoint2f mp1, MatOfPoint2f mp2, Mat mask) {
		DMatch[] matchesArray = matches.toArray();
	    
	    KeyPoint[] kpArray1 = keypoints1.toArray();
	    KeyPoint[] kpArray2 = keypoints2.toArray();
	    
	    for ( int i = 0; i < matchesArray.length; i++ ) {
	        // Get the keypoints from the good matches
	    	KeyPoint kp1 = kpArray1[matchesArray[i].queryIdx];
	        mp1Points.add(kp1.pt);
	        KeyPoint kp2 = kpArray2[matchesArray[i].trainIdx];
	        mp2Points.add(kp2.pt);
	    }
	    
	    mp1.fromList(mp1Points);
	    mp2.fromList(mp2Points);
	    
	    mp1Points.clear();
	    mp2Points.clear();
		
	    // Calculate fundamental matrix using RANSAC
		Mat F1 = Calib3d.findFundamentalMat(mp1, mp2, Calib3d.FM_RANSAC, dist1, 0.99, mask);
		
		byte[] return_buff = new byte[(int) (mask.total() * 
	    		mask.channels())];
	    mask.get(0, 0, return_buff);
	    
	    for (int i = 0; i < return_buff.length; i++) {
	    	// Get all inliers
	    	if (return_buff[i] == 1) {
	    		double[] pt1_buff = mp1.row(i).get(0, 0);
	    		double[] pt2_buff = mp2.row(i).get(0, 0);
	    		
	    		mp1Points.add(new Point(pt1_buff));
	    		mp2Points.add(new Point(pt2_buff));
	    	}
	    }
	    
	    mp1Inliers.fromList(mp1Points);
	    mp2Inliers.fromList(mp2Points);
	    
	    mp1Points.clear();
	    mp2Points.clear();
		
	    // Calculate fundamental matrix using linear equations
		Mat F2 = Calib3d.findFundamentalMat(mp1Inliers, mp2Inliers, Calib3d.FM_8POINT, dist2, 0.99);
		Mat F3 = Calib3d.findFundamentalMat(mp1Inliers, mp2Inliers, Calib3d.FM_8POINT, dist3, 0.99);
		
		// Return the fundamental matrix with the least error
		double err1 = avgError(F1, mask, mp1, mp2);
		double err2 = avgError(F2, mask, mp1, mp2);
		double err3 = avgError(F3, mask, mp1, mp2);
		
		if (err1 > err2) {
			if (err3 > err2) {
				return F2;
			} else {
				return F3;
			}
		} else if (err1 > err3) {
			return F3;
		} else {
			return F1;
		}
	}
	
	public static double avgError(Mat F, Mat mask, MatOfPoint2f mp1, MatOfPoint2f mp2) {
		// Store data from fundamental matrix
	    byte[] return_buff = new byte[(int) (mask.total() * 
	    		mask.channels())];
	    mask.get(0, 0, return_buff);
	    
	    // Inliers used for fundamental matrix calculation
	    int inliers = 0;
	    double err = 0;
	    
	    Mat result = new Mat();
	    
	    // Stores points temporarily
	    Mat pt1 = new Mat(3, 1, CvType.CV_64FC1);
	    Mat pt2 = new Mat(3, 1, CvType.CV_64FC1);
	    
	    for (int i = 0; i < mask.size().height; i++) {
	    	if (true) {
	    		inliers++;
	    			    		
	    		double[] pt1_buff = mp1.row(i).get(0, 0);
	    		double[] pt2_buff = mp2.row(i).get(0, 0);
	    		
	    		pt1.put(0, 0, pt1_buff);
	    		pt2.put(0, 0, pt2_buff);
	    		
	    		MathUtil.matMul(pt2, F, result);
	    		MathUtil.matMul(result, pt1.t(), result);
	    		
	    		err += Core.norm(result, Core.NORM_L1);
	    	}
	    }
	    
	    return err/inliers;
	}
}
