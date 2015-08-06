package com.mscarlett.sfm;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.Random;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class GraphicsUtil {
	
	public static void drawOptFlowMap(Mat flow, Mat cflowmap, int step, double scale, Scalar color) {
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

	public static void showResult(Mat img) {
	    Imgproc.resize(img, img, new Size(640*2, 480*2));
	    MatOfByte matOfByte = new MatOfByte();
	    Highgui.imencode(".jpg", img, matOfByte);
	    byte[] byteArray = matOfByte.toArray();
	    BufferedImage bufImage = null;
	    try {
	        InputStream in = new ByteArrayInputStream(byteArray);
	        bufImage = ImageIO.read(in);
	        JFrame frame = new JFrame();
	        frame.getContentPane().add(new JLabel(new ImageIcon(bufImage)));
	        frame.pack();
	        frame.setVisible(true);
	    } catch (Exception e) {
	        e.printStackTrace();
	    }
	}
	
	public static Scalar randColor() {
		Random rand = new Random();
		return new Scalar(rand.nextInt(256), rand.nextInt(256), rand.nextInt(256));
	}
}
