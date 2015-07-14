package com.mscarlett.sfm;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.imgproc.Imgproc;

public class SFMPipeline { //TODO should all images be matched pairwise?
	
	public static final boolean DEBUG = true; 
	
	private final FeatureMatching featureMatching;
	private final OpticalFlow opticalFlow;
	private final FundamentalMat fundamentalMat;
	private final Triangulation triangulation;
	private final Model3D model3d;
	
	private final Mat mask;
	private final Mat flow;
	
	private ImgData lastImage;
	
	private final ExecutorService executor;
	
	// Assumption: Images are related by a 3d rotation and translation
	// scaling is negligible for small distances
	public SFMPipeline() {
		featureMatching = new FeatureMatching();
		opticalFlow = new OpticalFlow();
		fundamentalMat = new FundamentalMat();
		triangulation = new Triangulation();
		model3d = new Model3D();
		
		mask = new Mat();
		flow = new Mat();
		
		lastImage = null;
		
		executor = Executors.newSingleThreadExecutor();
	}
	
	public synchronized void apply(Mat mat) {
		Mat grayscale = new Mat();
		
		// Convert image to grayscale
		Imgproc.cvtColor(mat, grayscale, Imgproc.COLOR_RGB2GRAY);
		
		Mat H;
		
		try {
			if (lastImage != null) {
				MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
				MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
				MatOfDMatch matches = new MatOfDMatch();
				MatOfPoint2f mp1 = new MatOfPoint2f();
				MatOfPoint2f mp2 = new MatOfPoint2f();
				Mat mask = new Mat();
				
				// Get keypoints and matches
				featureMatching.match(lastImage.grayscale, grayscale, keypoints1, keypoints2, matches);
				// Get fundamental matrix
				Mat F = fundamentalMat.getF(keypoints1, keypoints2, matches, mp1, mp2, mask);
				// Perform matrix multiplication to get the transformation
				//MathUtil.matMul(lastImage.transform, H, H);
				// Calculate optical flow
				opticalFlow.calcOpticalFlow(lastImage.grayscale, grayscale, flow);
			} else {
				H = Mat.eye(3, 3, CvType.CV_64F);
			}
			
			// Store image data
			//lastImage = new ImgData(mat, grayscale, H);
			
			// Update 3D model
			model3d.add(lastImage);
			
		} catch (CvException e) {
			e.printStackTrace();
		}
	}
	
	public void applyAsync(Mat mat) {
		executor.submit(new Runnable() {
			
			@Override
			public void run() {
				SFMPipeline.this.apply(mat);
			}
			
		});
	}
	
	public void getFundamentalMatrix() {
		
	}
	
	public void decompose() {
		
	}
	
	public void getModel() {
		
	}
}




