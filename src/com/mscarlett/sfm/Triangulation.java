package com.mscarlett.sfm;

import org.opencv.core.Point3;

public class Triangulation {

	public Triangulation() {
		
	}
	/*
	bool SimpleAdHocTracker::triangulateAndCheckReproj(const Mat& P, const Mat& P1) {
	    //undistort
	    Mat normalizedTrackedPts,normalizedBootstrapPts;
	    undistortPoints(Points<float>(trackedFeatures), normalizedTrackedPts, camMat, Mat());
	    undistortPoints(Points<float>(bootstrap_kp), normalizedBootstrapPts, camMat, Mat());
	 
	    //triangulate
	    Mat pt_3d_h(4,trackedFeatures.size(),CV_32FC1);
	    cv::triangulatePoints(P,P1,normalizedBootstrapPts,normalizedTrackedPts,pt_3d_h);
	    Mat pt_3d; convertPointsFromHomogeneous(Mat(pt_3d_h.t()).reshape(4, 1),pt_3d);
	    //    cout << pt_3d.size() << endl;
	    //    cout << pt_3d.rowRange(0,10) << endl;
	 
	    vector<uchar> status(pt_3d.rows,0);
	    for (int i=0; i<pt_3d.rows; i++) {
	        status[i] = (pt_3d.at<Point3f>(i).z > 0) ? 1 : 0;
	    }
	    int count = countNonZero(status);
	 
	    double percentage = ((double)count / (double)pt_3d.rows);
	    cout << count << "/" << pt_3d.rows << " = " << percentage*100.0 << "% are in front of camera";
	    if(percentage < 0.75)
	        return false; //less than 75% of the points are in front of the camera
	 
	 
	    //calculate reprojection
	    cv::Mat_<double> R = P(cv::Rect(0,0,3,3));
	    Vec3d rvec(0,0,0); //Rodrigues(R ,rvec);
	    Vec3d tvec(0,0,0); // = P.col(3);
	    vector<Point2f> reprojected_pt_set1;
	    projectPoints(pt_3d,rvec,tvec,camMat,Mat(),reprojected_pt_set1);
//	    cout << Mat(reprojected_pt_set1).rowRange(0,10) << endl;
	    vector<Point2f> bootstrapPts_v = Points<float>(bootstrap_kp);
	    Mat bootstrapPts = Mat(bootstrapPts_v);
//	    cout << bootstrapPts.rowRange(0,10) << endl;
	 
	    double reprojErr = cv::norm(Mat(reprojected_pt_set1),bootstrapPts,NORM_L2)/(double)bootstrapPts_v.size();
	    cout << "reprojection Error " << reprojErr;
	    if(reprojErr < 5) {
	        vector<uchar> status(bootstrapPts_v.size(),0);
	        for (int i = 0;  i < bootstrapPts_v.size(); ++ i) {
	            status[i] = (norm(bootstrapPts_v[i]-reprojected_pt_set1[i]) < 20.0);
	        }
	 
	        trackedFeatures3D.clear();
	        trackedFeatures3D.resize(pt_3d.rows);
	        pt_3d.copyTo(Mat(trackedFeatures3D));
	 
	        keepVectorsByStatus(trackedFeatures,trackedFeatures3D,status);
	        cout << "keeping " << trackedFeatures.size() << " nicely reprojected points";
	        bootstrapping = false;
	        return true;
	    }
	    return false;
	}
	 
	bool SimpleAdHocTracker::cameraPoseAndTriangulationFromFundamental(Mat_<double>& P, Mat_<double>& P1) {
	    //find fundamental matrix
	    double minVal,maxVal;
	    vector<Point2f> trackedFeaturesPts = Points<float>(trackedFeatures);
	    vector<Point2f> bootstrapPts = Points<float>(bootstrap_kp);
	    cv::minMaxIdx(trackedFeaturesPts,&minVal,&maxVal);
	    vector<uchar> status;
	    Mat F = findFundamentalMat(trackedFeaturesPts, bootstrapPts, FM_RANSAC, 0.006 * maxVal, 0.99, status);
	    int inliers_num = countNonZero(status);
	    cout << "Fundamental keeping " << inliers_num << " / " << status.size();
	    keepVectorsByStatus(trackedFeatures,bootstrap_kp,status);
	 
	    if(inliers_num > min_inliers) {
	        //Essential matrix: compute then extract cameras [R|t]
	        Mat_<double> E = camMat.t() * F * camMat; //according to HZ (9.12)
	 
	        //according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
	        if(fabsf(determinant(E)) > 1e-07) {
	            cout << "det(E) != 0 : " << determinant(E);
	            return false;
	        }
	 
	        Mat_<double> R1(3,3);
	        Mat_<double> R2(3,3);
	        Mat_<double> t1(1,3);
	        Mat_<double> t2(1,3);
	        if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;
	 
	        if(determinant(R1)+1.0 < 1e-09) {
	            //according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
	            cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign";
	            E = -E;
	            if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;
	        }
	        if(fabsf(determinant(R1))-1.0 > 1e-07) {
	            cerr << "det(R) != +-1.0, this is not a rotation matrix";
	            return false;
	        }
	 
	        Mat P = Mat::eye(3,4,CV_64FC1);
	 
	        //TODO: there are 4 different combinations for P1...
	        Mat_<double> P1 = (Mat_<double>(3,4) <<
	                           R1(0,0),   R1(0,1),    R1(0,2),    t1(0),
	                           R1(1,0),   R1(1,1),    R1(1,2),    t1(1),
	                           R1(2,0),   R1(2,1),    R1(2,2),    t1(2));
	        cout << "P1\n" << Mat(P1) << endl;
	 
	        bool triangulationSucceeded = true;
	        if(!triangulateAndCheckReproj(P,P1)) {
	            P1 = (Mat_<double>(3,4) <<
	                  R1(0,0),   R1(0,1),    R1(0,2),    t2(0),
	                  R1(1,0),   R1(1,1),    R1(1,2),    t2(1),
	                  R1(2,0),   R1(2,1),    R1(2,2),    t2(2));
	            cout << "P1\n" << Mat(P1) << endl;
	 
	            if(!triangulateAndCheckReproj(P,P1)) {
	                Mat_<double> P1 = (Mat_<double>(3,4) <<
	                                   R2(0,0),   R2(0,1),    R2(0,2),    t2(0),
	                                   R2(1,0),   R2(1,1),    R2(1,2),    t2(1),
	                                   R2(2,0),   R2(2,1),    R2(2,2),    t2(2));
	                cout << "P1\n" << Mat(P1) << endl;
	 
	                if(!triangulateAndCheckReproj(P,P1)) {
	                    Mat_<double> P1 = (Mat_<double>(3,4) <<
	                                       R2(0,0),   R2(0,1),    R2(0,2),    t1(0),
	                                       R2(1,0),   R2(1,1),    R2(1,2),    t1(1),
	                                       R2(2,0),   R2(2,1),    R2(2,2),    t1(2));
	                    cout << "P1\n" << Mat(P1) << endl;
	 
	                    if(!triangulateAndCheckReproj(P,P1)) {
	                        cerr << "can't find the right P matrix\n";
	                        triangulationSucceeded = false;
	                    }
	                }
	 
	            }
	 
	        }
	        return triangulationSucceeded;
	    }
	    return false;
	}*/
}
