package com.mscarlett.sfm;

import java.util.Random;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class EpipolarLinesOpticalFlowDemo extends AbstractDemo {

	private final OpticalFlow opticalFlow;
	private final OpticalFlowPoints ofPoints;
	private final FundamentalMat fundamentalMat;
	private Mat prev;
	private Mat prevGrayscale;

	public EpipolarLinesOpticalFlowDemo(String path) {
		super(path);
		
		opticalFlow = new OpticalFlow();
		ofPoints = new OpticalFlowPoints();
		fundamentalMat = new FundamentalMat();
		
		prev = null;
		prevGrayscale = null;
	}
	
	int i = 1;
	
	public void handleImg(Mat mat) {
		Mat grayscale = new Mat();
		Imgproc.cvtColor(mat, grayscale, Imgproc.COLOR_RGB2GRAY);
		
		if (prev != null) {
			Mat uFlow = new Mat();
			MatOfPoint2f mp1 = new MatOfPoint2f();
			MatOfPoint2f mp2 = new MatOfPoint2f();
			Mat mask = new Mat();
			
			opticalFlow.calcOpticalFlow(prevGrayscale, grayscale, uFlow);
			ofPoints.getPoints(uFlow, grayscale.cols(), grayscale.rows(), mp1, mp2);
			Mat F = fundamentalMat.getF(mp1, mp2, mask);
			
			double error = FundamentalMat.avgError(F, mask, mp1, mp2);
			System.out.println("Image " + i++ + " avg error: " + error);
			
			GraphicsUtil.drawEpipolarLines(F, prev, mat, mp1, mp2); 
		}
		
		prev = mat;
		prevGrayscale = grayscale;
	}
	
	public static void main(String[] args) {
		new EpipolarLinesOpticalFlowDemo("demo/resources/kermit").run();
	}
}
