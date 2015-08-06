package com.mscarlett.sfm;

import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

public class OpticalFlowPoints {
	
	public final double threshold;
	public final int interval;
	
	private final List<Point> list1;
	private final List<Point> list2;
	
	public OpticalFlowPoints() {
	    this(0.1, 6);
	}
	
	public OpticalFlowPoints(double threshold, int interval) {
		this.threshold = threshold;
		this.interval = interval;
		
	    list1 = new LinkedList<Point>();
	    list2 = new LinkedList<Point>();
    }

	public void getPoints(Mat uFlow, int width, int height, MatOfPoint2f mp1, MatOfPoint2f mp2) {
		for (int y = 0; y < height; y+=interval) {
	        for (int x = 0; x < width; x+=interval) {
	            /* Flow is basically the delta between left and right points */
	            double[] flow = uFlow.get(y, x);

	            /*  There's no need to calculate for every single point,
	                if there's not much change, just ignore it
	             */
	            if(Math.abs(flow[0]) < threshold && Math.abs(flow[1]) < threshold)
	                continue;

	            list1.add(new Point(x, y));
	            list2.add(new Point(x + flow[0], y + flow[1]));
	        }
	    }
		
		mp1.fromList(list1);
		mp2.fromList(list2);
		
		list1.clear();
		list2.clear();
	}
}
