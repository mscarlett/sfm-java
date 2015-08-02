package com.mscarlett.sfm;

import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Point3;

public class ExtractRT {

	public ExtractRT() {
		
	}
	
	public boolean extractRT(Mat F, Mat K, Mat R, Mat t, Mat inliers1, Mat inliers2) {
		Mat E = new Mat();
		
        MathUtil.essentialMatrix(K, F, E);
        
        // det(E) must equal 0
        // see https://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
        if(Math.abs(Core.determinant(E)) > 1e-07) {
        	return false;
        }
	    
	    Mat R1 = new Mat();
	    Mat R2 = new Mat();
	    Mat t1 = new Mat();
	    Mat t2 = new Mat();
	    
	    // Find all solutions for the rotation and translation
	    MathUtil.extractRTfromEssential(E, R1, R2, t1, t2);
	    
	    // Find the correct solution by checking that points are in front of both cameras
	    if (inFrontOfBothCameras(inliers1, inliers2, R1, t1)) {
	    	R.setTo(R1);
	    	t.setTo(t1);
	    } else if (inFrontOfBothCameras(inliers1, inliers2, R1, t2)) {
	    	R.setTo(R1);
	    	t.setTo(t2);
	    } else if (inFrontOfBothCameras(inliers1, inliers2, R2, t1)) {
	    	R.setTo(R2);
	    	t.setTo(t1);
	    } else if (inFrontOfBothCameras(inliers1, inliers2, R2, t2)) {
	    	R.setTo(R2);
	    	t.setTo(t2);
		} else {
			return false;
		}
	    
	    return true;
	}
	
	// where does this function come from?
	// http://linux2biz.net/276869/camera-pose-estimation-from-essential-matrix
	public static boolean inFrontOfBothCameras(Mat inliers1, Mat inliers2, Mat R, Mat T) {
		int numPoints = inliers1.rows();
		
		if (numPoints != inliers2.rows()) {
        	throw new RuntimeException("Assertion failed: inliers1.rows() != inliers2.rows()");
        }
		
		// check if the point correspondences are in front of both images
	    for (int i = 0; i < numPoints; i++) {
	    	double[] first = inliers1.get(i, 0);
	    	double[] second = inliers2.get(i, 0);
	    	
	    	Mat m1 = new Mat();
	    	Mat m2 = new Mat();
	    	Mat first_z = new Mat();
	    	Mat first_3d_point = new Mat();
	    	Mat second_3d_point = new Mat();
	    	
	    	// first is 1x2 matrix
	    	// second is 1x2 matrix
	    	// R is 3x3 matrix
	    	// T is 3x1 matrix
	    	// first_z is ???
	    	//Core.multiply(second, R.row(2), m1);
	    	Core.subtract(R.row(0), m1, m1);
	    	m1.copyTo(m2);
	    	MathUtil.matMul(m1, T, m1);
	    	//MathUtil.matMul(m2, second, m2);
	    	Core.divide(m1, m2, first_z);
	        
	    	//Core.multiply(first, first_z, m1);
	    	//Core.multiply(second, first_z, m2);
	    	
	    	List<Mat> hz = new LinkedList<Mat>();
	    	
	    	hz.add(m1);
	    	hz.add(m2);
	    	hz.add(first_z);
	    	
	    	Core.hconcat(hz, first_3d_point);
	        MathUtil.matMul(R.t(), first_3d_point, m1);
	        MathUtil.matMul(R.t(), T, m2);
	        Core.subtract(m1, m2, second_3d_point);
	        
	        double[] p1 = first_3d_point.get(0, 0);
	        double[] p2 = second_3d_point.get(0, 0);
	        
	        if (p1[2] < 0 || p2[2] < 0) {
	            return false;
	        }
	    }
		
		return true;
	}
}
