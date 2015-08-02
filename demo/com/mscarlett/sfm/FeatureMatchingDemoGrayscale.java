package com.mscarlett.sfm;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Size;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class FeatureMatchingDemoGrayscale extends AbstractDemo {
	
	private final FeatureMatching featureMatching;
	private Mat prev;
	private Mat prevGrayscale;

	public FeatureMatchingDemoGrayscale(String path) {
		super(path);
		
		featureMatching = new FeatureMatching();
		prev = null;
		prevGrayscale = null;
	}
	
	int i = 0;
	
	public void handleImg(Mat mat) {
		Mat grayscale = new Mat();
		Imgproc.cvtColor(mat, grayscale, Imgproc.COLOR_RGB2GRAY);
		
		if (prev != null) {
			MatOfKeyPoint mp1Points = new MatOfKeyPoint();
			MatOfKeyPoint mp2Points = new MatOfKeyPoint();
			MatOfDMatch matches = new MatOfDMatch();
			
			featureMatching.match(prevGrayscale, grayscale, mp1Points, mp2Points, matches);
			
			Mat rgb = new Mat();
		    
			Features2d.drawMatches(prev, mp1Points, mat, mp2Points, matches, rgb);
		    //Highgui.imwrite("match_" +i+++ ".jpeg", rgb);
		    showResult(rgb); 
		}
		
		prev = mat;
		prevGrayscale = grayscale;
	}
	
	public static void main(String[] args) {
		new FeatureMatchingDemoGrayscale("demo/resources/kermit").run();
	}
}
