package com.mscarlett.sfm;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class SFM {
	
	private static final Mat ZEROS = Mat.zeros(3, 3, CvType.CV_64F);
	
	private static final double qualityLevel = 0.35;
	private static final double minDistance = 10;
	private static final int blockSize = 8;
	private static final boolean useHarrisDetector = false;
	private static final double k = 0.0;
	private static final int maxCorners = 100;
	
	private final FeatureMatching featureMatching;
	private final OpticalFlow opticalFlow;
	
	private final Mat mask;
	private final Mat flow;
	
	private ImgData lastImage;
	
	private final ExecutorService executor;
	
	// Assumption: Images are related by a 3d rotation and translation
	// scaling is negligible for small distances
	public SFM() {
		featureMatching = new FeatureMatching();
		opticalFlow = new OpticalFlow();
		
		mask = new Mat();
		flow = new Mat();
		
		lastImage = null;
		
		executor = Executors.newSingleThreadExecutor();
	}
	
	public synchronized void apply(Mat mat) {
		Mat grayscale = new Mat();
		/*MatOfPoint corners = new MatOfPoint();
		
		Imgproc.goodFeaturesToTrack(grayscale,
				corners,
				maxCorners,
				qualityLevel,
				minDistance,
				mask,
				blockSize,
				useHarrisDetector,
				k);
		
		Point[] points = corners.toArray();*/
		
		Imgproc.cvtColor(mat, grayscale, Imgproc.COLOR_RGB2GRAY);
		
		Mat H;
		
		try {
			if (lastImage != null) {
				// TODO check if good match
				H = featureMatching.match(lastImage.grayscale, grayscale);
				Core.gemm(lastImage.transform, H, 0, ZEROS, 0, H);
				opticalFlow.calcOpticalFlow(lastImage.grayscale, grayscale, flow);
			} else {
				H = Mat.eye(3, 3, CvType.CV_64F);
			}
			
			lastImage = new ImgData(mat, grayscale, H);
			
		} catch (CvException e) {
			e.printStackTrace();
		}
	}
	
	public void applyAsync(Mat mat) {
		executor.submit(new Runnable() {
			@Override
			public void run() {
				SFM.this.apply(mat);
			}
		});
	}
	
	private static class ImgData {
		
		public final Mat image;
		public final Mat grayscale;
		public final Mat transform;
		
		public ImgData(Mat image, Mat grayscale, Mat transform) {
			this.image = image;
			this.grayscale = grayscale;
			this.transform = transform;
		}
	}
}




