package com.mscarlett.sfm;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

public class OpticalFlowDemo extends AbstractDemo {

	private final OpticalFlow opticalFlow;
	private Mat prev;
	private Mat prevGrayscale;
	
	public OpticalFlowDemo(String path) {
		super(path);
		opticalFlow = new OpticalFlow();
		prev = null;
		prevGrayscale = null;
	}

	@Override
	public void handleImg(Mat mat) {
		Mat grayscale = new Mat();
		Imgproc.cvtColor(mat, grayscale, Imgproc.COLOR_RGB2GRAY);
		
		if (prev != null) {
			Mat flow = new Mat();
			Mat flowMap = prev;
			opticalFlow.calcOpticalFlow(prevGrayscale, grayscale, flow);
			drawOptFlowMap(flow, flowMap, 16, 1.5, new Scalar(0, 255, 0));
		    showResult(flowMap); 
		}
		
		prev = mat;
		prevGrayscale = grayscale;
	}
	
	private void drawOptFlowMap(Mat flow, Mat cflowmap, int step, double scale, Scalar color) 
	{
		for (int y = 0; y < cflowmap.rows(); y += step) {
	        for (int x = 0; x < cflowmap.cols(); x += step) {
	        	double[] point = flow.get(y, x);
	        	Point first = new Point(x, y);
	        	Point second = new Point(x+point[0], y+point[1]);
	        	Core.line(cflowmap, first, second, color, 1, 8, 0);
	        	Core.circle(cflowmap, first, 2, color, -1, 8, 0);
	        }
	    }
	}
	
	public static void main(String[] args) {
		new OpticalFlowDemo("demo/resources/kermit").run();
	}

}
