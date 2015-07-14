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

public class EpipolarLinesDemo extends AbstractDemo {

	private final FeatureMatching featureMatching;
	private final FundamentalMat fundamentalMat;
	private Mat prev;
	private Mat prevGrayscale;

	public EpipolarLinesDemo(String path) {
		super(path);
		
		featureMatching = new FeatureMatching();
		fundamentalMat = new FundamentalMat();
		
		prev = null;
		prevGrayscale = null;
	}
	
	int i = 1;
	
	public void handleImg(Mat mat) {
		Mat grayscale = new Mat();
		Imgproc.cvtColor(mat, grayscale, Imgproc.COLOR_RGB2GRAY);
		
		if (prev != null) {
			MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
			MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
			MatOfDMatch matches = new MatOfDMatch();
			MatOfPoint2f mp1 = new MatOfPoint2f();
			MatOfPoint2f mp2 = new MatOfPoint2f();
			Mat mask = new Mat();
			
			featureMatching.match(prevGrayscale, grayscale, keypoints1, keypoints2, matches);
			Mat F = fundamentalMat.getF(keypoints1, keypoints2, matches, mp1, mp2, mask);
			
			double error = FundamentalMat.avgError(F, mask, mp1, mp2);
			System.out.println("Image " + i++ + " avg error: " + error);
			
			drawEpipolarLines(F, prev, mat, mp1, mp2); 
		}
		
		prev = mat;
		prevGrayscale = grayscale;
	}
	
	public static void drawEpipolarLines(Mat F, Mat img1, Mat img2, Mat points1, Mat points2) {
        if (!img1.size().equals(img2.size())) {
        	throw new RuntimeException("Assertion failed: !img1.size().equals(img2.size())");
        }
        
        if (img1.type() != img2.type()) {
        	throw new RuntimeException("Assertion failed: img1.type() != img2.type()");
        }
        
        if (!points1.size().equals(points2.size())) {
        	throw new RuntimeException("Assertion failed: !points1.size().equals(points2.size())");
        }
		
		Mat lines1 = new Mat();
        Mat lines2 = new Mat();
        
		Calib3d.computeCorrespondEpilines(points1, 1, F, lines1);
		Calib3d.computeCorrespondEpilines(points2, 2, F, lines2);
		
		Rect rect1 = new Rect(0, 0, img1.cols(), img1.rows());
		Rect rect2 = new Rect(img1.cols(), 0, img1.cols(), img1.rows());
		
		Mat outImg = new Mat(img1.rows(), img1.cols()*2, img1.type());
		img1.copyTo(outImg.submat(rect1));
		img2.copyTo(outImg.submat(rect2));
		
		int epiLinesCount = lines1.rows();

        double a, b, c;

        int x0, y0, x1, y1;
        
        Point p1, p2;
        
        Scalar color;
        
        for (int line = 0; line < epiLinesCount; line++) {
            a = lines1.get(line, 0)[0];
            b = lines1.get(line, 0)[1];
            c = lines1.get(line, 0)[2];

            x0 = 0;
            y0 = (int) (-(c + a * x0) / b);
            x1 = img1.cols();
            y1 = (int) (-(c + a * x1) / b);

            p1 = new Point(x0, y0);
            p2 = new Point(x1, y1);
            color = randColor();
            Core.line(outImg.submat(rect2), p1, p2, color);
            Core.circle(outImg.submat(rect1), p1, 5, color);
		    Core.circle(outImg.submat(rect1), p2, 5, color);

            a = lines2.get(line, 0)[0];
            b = lines2.get(line, 0)[1];
            c = lines2.get(line, 0)[2];

            x0 = 0;
            y0 = (int) (-(c + a * x0) / b);
            x1 = img2.cols();
            y1 = (int) (-(c + a * x1) / b);

            p1 = new Point(x0, y0);
            p2 = new Point(x1, y1);
            color = randColor();
            Core.line(outImg.submat(rect1), p1, p2, color);
            Core.circle(outImg.submat(rect2), p1, 5, color);
		    Core.circle(outImg.submat(rect2), p2, 5, color);
        }
		
		showResult(outImg);
	}
	
	private static Scalar randColor() {
		Random rand = new Random();
		return new Scalar(rand.nextInt(256), rand.nextInt(256), rand.nextInt(256));
	}
	
	public static void main(String[] args) {
		new EpipolarLinesDemo("demo/resources/kermit").run();
	}
}
