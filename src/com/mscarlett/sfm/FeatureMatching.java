package com.mscarlett.sfm;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
import org.opencv.features2d.KeyPoint;

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
		
		DMatch[] matchesArray = matches.toArray();
		
		// Min distance among matches
	    double min_dist = 100;

	    for( int i = 0; i < descriptors1.rows(); i++ ) {
	    	double dist = matchesArray[i].distance;
	        if( dist < min_dist ) {
	        	min_dist = dist;
	        }
	    }

	    for( int i = 0; i < descriptors1.rows(); i++ ) {
            if( matchesArray[i].distance < 3*min_dist ) {
                goodMatches.add(matchesArray[i]);
            }
	    }
	    
	    matches.fromList(goodMatches);
	    
	    Features2d.drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);
	    
	    List<DMatch> imgMatches = matches.toList();
	    
	    KeyPoint[] kpArray1 = keypoints1.toArray();
	    KeyPoint[] kpArray2 = keypoints2.toArray();
	    
	    for ( int i = 0; i < imgMatches.size(); i++ ) {
	        //-- Get the keypoints from the good matches
	    	// does it work?
	    	KeyPoint kp1 = kpArray1[imgMatches.get(i).queryIdx];
	        mp1Points.add(kp1.pt);
	        KeyPoint kp2 = kpArray2[imgMatches.get(i).trainIdx];
	        mp2Points.add(kp2.pt);
	    }
	    
	    mp1.fromList(mp1Points);
	    mp2.fromList(mp2Points);
	    
	    // Maximum allowed reprojection error to treat point pair as inlier
	    int ransacReprojThreshold = 5;
	    
		H = Calib3d.findHomography(mp1, mp2, Calib3d.RANSAC, ransacReprojThreshold);
	}
	
	ExecutorService service = Executors.newFixedThreadPool(1);
	
	public void matchAsync(Mat lastImage, Mat image, long time) {
		service.submit(new Runnable() {
			@Override
			public void run() {
				FeatureMatching.this.match(lastImage, image);
			}
		});
	}
	
	public Mat H() {
		return H;
	}
}
