package com.mscarlett.sfm;

import java.util.ArrayList;
import java.util.Arrays;
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
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;


public class FeatureMatching {
	
	private final FeatureDetector featureDetector;
	private final DescriptorExtractor extractor;
	private final DescriptorMatcher matcher;
	
	private final Mat descriptors1;
	private final Mat descriptors2;
	
	private final List<DMatch> goodMatches;
    
	private final double threshold;
	
	public FeatureMatching() {
		this(FeatureDetector.SIFT, DescriptorExtractor.SIFT, DescriptorMatcher.FLANNBASED, 3.0);
	}
	
	public FeatureMatching(int detectorType, int extractorType, int matcherType, double threshold) {
	    featureDetector = FeatureDetector.create(detectorType);
	    extractor = DescriptorExtractor.create(extractorType);
	    matcher = DescriptorMatcher.create(matcherType);
		descriptors1 = new Mat();
		descriptors2 = new Mat();
		goodMatches = new ArrayList<DMatch>();
		
		this.threshold = threshold;
	}
	
	public synchronized void match(Mat img1, Mat img2, MatOfKeyPoint keypoints1, MatOfKeyPoint keypoints2, MatOfDMatch matches) {
		featureDetector.detect(img1, keypoints1);
		featureDetector.detect(img2, keypoints2);
		extractor.compute(img1, keypoints1, descriptors1);
		extractor.compute(img2, keypoints2, descriptors2);
		matcher.match(descriptors1, descriptors2, matches);
		
		DMatch[] matchesArray = matches.toArray();
		
		int numDescriptors = descriptors1.rows();
		
		double min_dist = 100;
		
		for( int i = 0; i < descriptors1.rows(); i++ ) {
	    	double dist = matchesArray[i].distance;
	        if (dist < min_dist) {
	        	min_dist = dist;
	        }
	    }
	    
	    double max_dist = threshold*min_dist;
        
	    for( int i = 0; i < numDescriptors; i++ ) {
	    	// check that matches are sufficiently close
            if (matchesArray[i].distance <= max_dist) {
                goodMatches.add(matchesArray[i]);
            }
	    }
	    
	    matches.fromList(goodMatches);
	    
	    goodMatches.clear();
	}
}
