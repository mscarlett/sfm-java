package com.mscarlett.sfm;

import org.opencv.core.Mat;

public class ImgData {
	
	public final Mat image;
	public final Mat grayscale;
	public final Mat transform;
	
	public ImgData(Mat image, Mat grayscale, Mat transform) {
		this.image = image;
		this.grayscale = grayscale;
		this.transform = transform;
	}
}