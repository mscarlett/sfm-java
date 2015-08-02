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

public class FeatureMatchingDemoColor extends AbstractDemo {
	
	private final FeatureMatching featureMatching;
	private Mat prev;
	private Mat prevGrayscale;

	public FeatureMatchingDemoColor(String path) {
		super(path);
		
		featureMatching = new FeatureMatching();
		prev = null;
		prevGrayscale = null;
	}
	
	int i = 0;
	
	public void handleImg(Mat mat) {
		if (prev != null) {
			MatOfKeyPoint mp1Points = new MatOfKeyPoint();
			MatOfKeyPoint mp2Points = new MatOfKeyPoint();
			MatOfDMatch matches = new MatOfDMatch();
			
			featureMatching.match(prev, mat, mp1Points, mp2Points, matches);
			
			Mat rgb = new Mat();
		    
			Features2d.drawMatches(prev, mp1Points, mat, mp2Points, matches, rgb);
		    //Highgui.imwrite("match_" +i+++ ".jpeg", rgb);
		    showResult(rgb); 
		}
		
		prev = mat;
	}
	
	public static void main(String[] args) {
		new FeatureMatchingDemoColor("demo/resources/kermit").run();
	}
}
