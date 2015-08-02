package com.mscarlett.sfm;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.KeyPoint;

public class Model3D {

	public void add(ImgData lastImage) {
		
	}
	
	public void update() {
		
	}
	
	/*
	private final List<CloudPoint> pcloud;
	private final List<List<KeyPoint>> imgPoints;
	private final List<List<KeyPoint>> fullPoints;
	private final List<List<KeyPoint>> imgPoints_good;
	
	private final Map<Pair, List<DMatch>> matches_matrix;
	
	private final List<ImgData> imgs;
	
	private final Map<Integer, Mat> Pmats;
	
	Mat K;
	Mat Kinv;
	
	Mat cam_matrix,distortion_coeff;
	Mat distcoeff_32f; 
	Mat K_32f;

	List<byte[]> pointCloudRGB;
	//List<cv::KeyPoint> correspImg1Pt; //TODO: remove
	
	//IFeatureMatcher feature_matcher;
	
	public Model3D() {
		pcloud = new LinkedList<CloudPoint>();
		imgPoints = new LinkedList<List<KeyPoint>>();
		fullPoints = new LinkedList<List<KeyPoint>>();
		imgPoints_good = new LinkedList<List<KeyPoint>>();
		
		matches_matrix = new HashMap<Pair, List<DMatch>>();
		
		imgs_orig = new LinkedList<ImgData>();
		imgs = new LinkedList<Mat>();
		Pmats = new HashMap<Integer, Mat>();
		
		pointCloudRGB = new LinkedList<byte[]>();
	}
	
	public void initialize() {
		//imgPoints.clear();
		//fullPoints.clear();
		//imgPoints_good.clear();
		//matches_matrix.clear();
		//imgs_orig.clear();
		//imgs.clear();
		//Pmats.clear();
	}
		
	public void add(ImgData img) {
		//imgPoints.add(new LinkedList<CloudPoint>());
		//fullPoints.add(new LinkedList<Keypoint>());
		//imgPoints_good.add(new LinkedList<Keypoint>());
		//imgs_orig.add(img);
		//imgs.add(new LinkedList());
		//Pmats.add(new LinkedList());
	}

	void find2D3DCorrespondences(int working_view, List<Point> ppcloud, List<Point> imgPoints) 
		{
			ppcloud.clear();
            imgPoints.clear();

			List<Integer> pcloud_status =  new LinkedList<Integer>();
			
			for (;;)
			{
				//check for matches_from_old_to_working between i'th frame and <old_view>'th frame (and thus the current cloud)
				DMatch[] matches_from_old_to_working = matches_matrix.get(new Pair(old_view,working_view));

				for (int match_from_old_view=0; match_from_old_view < matches_from_old_to_working.size(); match_from_old_view++) {
					// the index of the matching point in <old_view>
					int idx_in_old_view = matches_from_old_to_working[match_from_old_view].queryIdx;

					//scan the existing cloud (pcloud) to see if this point from <old_view> exists
					for (int pcldp=0; pcldp<pcloud.size(); pcldp++) {
						// see if corresponding point was found in this point
						if (idx_in_old_view == pcloud.get(pcldp).imgpt_for_img[old_view] && pcloud_status[pcldp] == 0) //prevent duplicates
						{
							//3d point in cloud
							ppcloud.add(pcloud.get(pcldp).pt);
							//2d point in image i
							imgPoints.push_back(imgPoints.get(working_view)[matches_from_old_to_working[match_from_old_view].trainIdx].pt);

							pcloud_status.set(pcldp, 1);
							break;
						}
					}
				}
			}
			System.out.println("found " + ppcloud.size() + " 3d-2d point correspondences");
		}
	
	public void recoverDepthFromImages() {
		// Match features
		OnlyMatchFeatures();
		
		// Depth recovery start
		
		PruneMatchesBasedOnF();
		GetBaseLineTriangulation();
		AdjustCurrentBundle();
		update(); //notify listeners
		
		cv::Matx34d P1 = Pmats[m_second_view];
		cv::Mat_<double> t = (cv::Mat_<double>(1,3) << P1(0,3), P1(1,3), P1(2,3));
		cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P1(0,0), P1(0,1), P1(0,2), 
													   P1(1,0), P1(1,1), P1(1,2), 
													   P1(2,0), P1(2,1), P1(2,2));
		cv::Mat_<double> rvec(1,3); Rodrigues(R, rvec);
		
		done_views.clear(); good_views.clear();

		done_views.insert(m_first_view);
		done_views.insert(m_second_view);
		good_views.insert(m_first_view);
		good_views.insert(m_second_view);

		//loop images to incrementally recover more cameras 
		//for (unsigned int i=0; i < imgs.size(); i++) 
		while (done_views.size() != imgs.size())
		{
			//find image with highest 2d-3d correspondance [Snavely07 4.2]
			unsigned int max_2d3d_view = -1, max_2d3d_count = 0;
			vector<cv::Point3f> max_3d; vector<cv::Point2f> max_2d;
			for (unsigned int _i=0; _i < imgs.size(); _i++) {
				if(done_views.find(_i) != done_views.end()) continue; //already done with this view

				vector<cv::Point3f> tmp3d; vector<cv::Point2f> tmp2d;
				cout << imgs_names[_i] << ": ";
				Find2D3DCorrespondences(_i,tmp3d,tmp2d);
				if(tmp3d.size() > max_2d3d_count) {
					max_2d3d_count = tmp3d.size();
					max_2d3d_view = _i;
					max_3d = tmp3d; max_2d = tmp2d;
				}
			}
			int i = max_2d3d_view; //highest 2d3d matching view

			std::cout << "-------------------------- " << imgs_names[i] << " --------------------------\n";
			done_views.insert(i); // don't repeat it for now

			bool pose_estimated = FindPoseEstimation(i,rvec,t,R,max_3d,max_2d);
			if(!pose_estimated)
				continue;

			//store estimated pose	
			Pmats[i] = cv::Matx34d	(R(0,0),R(0,1),R(0,2),t(0),
									 R(1,0),R(1,1),R(1,2),t(1),
									 R(2,0),R(2,1),R(2,2),t(2));
			
			// start triangulating with previous GOOD views
			for (set<int>::iterator done_view = good_views.begin(); done_view != good_views.end(); ++done_view) 
			{
				int view = *done_view;
				if( view == i ) continue; //skip current...

				cout << " -> " << imgs_names[view] << endl;
				
				vector<CloudPoint> new_triangulated;
				vector<int> add_to_cloud;
				bool good_triangulation = TriangulatePointsBetweenViews(i,view,new_triangulated,add_to_cloud);
				if(!good_triangulation) continue;

				std::cout << "before triangulation: " << pcloud.size();
				for (int j=0; j<add_to_cloud.size(); j++) {
					if(add_to_cloud[j] == 1)
						pcloud.push_back(new_triangulated[j]);
				}
				std::cout << " after " << pcloud.size() << std::endl;
				//break;
			}
			good_views.insert(i);
			
			AdjustCurrentBundle();
			update();
		}

		cout << "======================================================================\n";
		cout << "========================= Depth Recovery DONE ========================\n";
		cout << "======================================================================\n";
	}
	
	public List<Point> getPointCloud() {
		return pcloud;
	}*/

}