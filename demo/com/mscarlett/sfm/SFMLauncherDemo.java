package com.mscarlett.sfm;

import org.opencv.core.Mat;

public class SFMLauncherDemo extends AbstractDemo {
	
	private final SFMPipeline sfm;
	
	public SFMLauncherDemo(String path) {
		super(path);
		
		this.sfm = new SFMPipeline();
	}
	
	public void handleImg(Mat mat) {
		sfm.apply(mat);
	}
	
	public static void main(String[] args) {
		new SFMLauncherDemo("demo/resources/kermit").run();
	}
}
