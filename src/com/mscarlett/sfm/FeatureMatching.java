package com.mscarlett.sfm;

import java.util.ArrayList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;

public class FeatureMatching {
	
	private final FeatureDetector featureDetector;
	private final DescriptorExtractor extractor;
	private final DescriptorMatcher matcher;
	
	private final MatOfKeyPoint keypoints1;
	private final MatOfKeyPoint keypoints2;
	private final Mat descriptors1;
	private final Mat descriptors2;
	private final MatOfDMatch matches;
	private final Mat imgMatches;
	
	private final List<DMatch> goodMatches;
	private final List<Point> mp1Points;
    private final List<Point> mp2Points;
	
    private final MatOfPoint2f mp1;
    private final MatOfPoint2f mp2;
    
    private Mat H;
    
	public FeatureMatching() {
		this(FeatureDetector.SIFT, DescriptorExtractor.SIFT, DescriptorMatcher.FLANNBASED);
	}
	
	public FeatureMatching(int detectorType, int extractorType, int matcherType) {
	    featureDetector = FeatureDetector.create(detectorType);
	    extractor = DescriptorExtractor.create(extractorType);
	    matcher = DescriptorMatcher.create(matcherType);
	    
		keypoints1 = new MatOfKeyPoint();
		keypoints2 = new MatOfKeyPoint();
		descriptors1 = new Mat();
		descriptors2 = new Mat();
		matches = new MatOfDMatch();
		imgMatches = new Mat();
		
		goodMatches = new ArrayList<DMatch>();
		mp1Points = new ArrayList<Point>();
		mp2Points = new ArrayList<Point>();
		
	    mp1 = new MatOfPoint2f();
	    mp2 = new MatOfPoint2f();
	}
	
	// http://docs.opencv.org/doc/tutorials/features2d/feature_homography/feature_homography.html
	public void match(Mat img1, Mat img2) {
		featureDetector.detect(img1, keypoints1);
		featureDetector.detect(img2, keypoints2);
		extractor.compute(img1, keypoints1, descriptors1);
		extractor.compute(img2, keypoints2, descriptors2);
		matcher.match(descriptors1, descriptors2, matches);
		
		List<DMatch> matchesList = matches.toList();
		
	    double min_dist = 100;

	    for( int i = 0; i < descriptors1.rows(); i++ ) {
	    	double dist = matchesList.get(i).distance;
	        if( dist < min_dist ) {
	        	min_dist = dist;
	        }
	    }

	    for( int i = 0; i < descriptors1.rows(); i++ ) {
            if( matchesList.get(i).distance < 3*min_dist ) {
                goodMatches.add(matchesList.get(i));
            }
	    }
	    
	    matches.fromList(goodMatches);
	    
	    Features2d.drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);
	    
	    List<DMatch> imgMatches = matches.toList();
	    
	    double[] tmp = {0,0};
	    
	    for ( int i = 0; i < imgMatches.size(); i++ ) {
	        //-- Get the keypoints from the good matches
	    	// does it work?
	        keypoints1.get(imgMatches.get(i).queryIdx, 0, tmp);
	        mp1Points.add(new Point(tmp));
	        keypoints2.get(imgMatches.get(i).trainIdx, 0, tmp);
	        mp2Points.add(new Point(tmp));
	    }
	    
	    mp1.fromList(mp1Points);
	    mp2.fromList(mp2Points);
	    
		H = Calib3d.findHomography(mp1, mp2, Calib3d.RANSAC, 5);
	}
	
	public Mat H() {
		return H;
	}
}
