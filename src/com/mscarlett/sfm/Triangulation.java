package com.mscarlett.sfm;

import java.util.LinkedList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Point3;

public class Triangulation {

	public Triangulation() {
		
	}
	
	public void getInitialTriangulation() {
		Mat P = Mat.eye(3, 4, CvType.CV_64F);
		Mat P1 = Mat.eye(3, 4, CvType.CV_64F);
		
		List<CloudPoint> tmp_pcloud;
	}
	
	public void triangulate(Mat projMatr1, Mat projMatr2, Mat projPoints1, Mat projPoints2, Mat points4D) {
		Calib3d.triangulatePoints(projMatr1,
                projMatr2,
                projPoints1,
                projPoints2,
                points4D);
	}
	
}
