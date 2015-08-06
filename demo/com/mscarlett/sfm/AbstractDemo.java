package com.mscarlett.sfm;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public abstract class AbstractDemo {
	
	static{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
	
	protected final String path;
	
	public AbstractDemo(String path) {
		this.path = path;
	}
	
	public void run() {
		File[] files = new File(path).listFiles();
		
		for (File file: files) {
			Mat mat = loadFromFile(file);
			handleImg(mat);
		}
	}
	
	public Mat loadFromFile(File file) {
		Mat mat = Highgui.imread(file.getAbsolutePath(), Highgui.CV_LOAD_IMAGE_COLOR);
		if (mat.empty()) {
			throw new RuntimeException("Image " + file + " not loaded");
		}
		return mat;
	}
	
	public abstract void handleImg(Mat mat);

}
