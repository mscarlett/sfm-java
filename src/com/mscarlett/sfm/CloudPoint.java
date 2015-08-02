package com.mscarlett.sfm;

import java.util.List;

import org.opencv.core.Point3;

public class CloudPoint {

	public Point3 pt;
	public List<Integer> imgpt_for_img;
	public double reprojection_error;
}
