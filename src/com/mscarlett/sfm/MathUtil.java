package com.mscarlett.sfm;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class MathUtil {
	
	public static final Mat ZEROS = Mat.zeros(3, 3, CvType.CV_64FC1);
	public static final Mat EYE = Mat.eye(3, 3, CvType.CV_64FC1);
	public static final Mat W = new Mat(3, 3, CvType.CV_64FC1);
	public static final Mat Winv = new Mat(3, 3, CvType.CV_64FC1);
	
	public static final Scalar NEG_ONE = new Scalar(-1);
	
	static {
	    W.put(0, 0,
	    		0, -1, 0,
	    	    1, 0, 0,
	    	    0, 0, 1);
	    Winv.put(0,  0,
                0, 1, 0,
	    	    -1, 0, 0,
	    	    0, 0, 1);
	}

	public static double norm(double[] d) {
		double sum = 0;
		for (int i = 0; i < d.length; i++) {
			sum += d[i]*d[i];
		}
		return Math.sqrt(sum);
	}
	
	public static void matMul(Mat A, Mat B, Mat C) {
		Core.gemm(B, A, 1, ZEROS, 0, C);
	}
	
	public static void matMul(double[] A, double[] B, double[] C, int A_rows, int A_cols, int B_cols) {
		for (int i = 0; i < A_rows; i++) {
			for (int j = 0; j < B_cols; j++) {
				C[i*B_cols+j] = 0;
				
				for (int k = 0; k < A_cols; k++) {
				    C[i*B_cols+j] += A[i*A_cols+k]*B[k*B_cols+j];
				}
			}
		}
	}
	
	public static void essentialMatrix(Mat A, Mat F, Mat E) {
		MathUtil.matMul(A.t(), F, E);
		MathUtil.matMul(E, A, E);
	}
	
	public static void extractRTfromEssential(Mat E, Mat R1, Mat R2, Mat t1, Mat t2) {
		Mat w = new Mat();
		Mat u = new Mat();
		Mat vt = new Mat();
		
		Core.SVDecomp(E, w, u, vt);
		
		matMul(u, W, R1);
		matMul(R1, vt, R1);
		
		matMul(u, W.t(), R2);
		matMul(R2, vt, R2);
		
		t1.setTo(u.col(2));
		Core.multiply(t1, NEG_ONE, t2);
	}
	
	/*
	public static boolean FindCameraMatrices(Mat K, 
			Mat Kinv, 
			Mat distcoeff,
			List<KeyPoint> imgpts1,
			List<KeyPoint> imgpts2,
			List<KeyPoint> imgpts1_good,
			List<KeyPoint> imgpts2_good,
			Mat P,
			Mat P1,
			List<DMatch> matches,
			List<CloudPoint> outCloud) 
{
//Find camera matrices
{
cout << "Find camera matrices...";
double t = getTickCount();

Mat F = GetFundamentalMat(imgpts1,imgpts2,imgpts1_good,imgpts2_good,matches);
if(matches.size() < 100) { // || ((double)imgpts1_good.size() / (double)imgpts1.size()) < 0.25
//cerr << "not enough inliers after F matrix" << endl;
return false;
}

//Essential matrix: compute then extract cameras [R|t]
Mat_<double> E = K.t() * F * K; //according to HZ (9.12)

//according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
if(fabsf(determinant(E)) > 1e-07) {
cout << "det(E) != 0 : " << determinant(E) << "\n";
P1 = 0;
return false;
}

Mat_<double> R1(3,3);
Mat_<double> R2(3,3);
Mat_<double> t1(1,3);
Mat_<double> t2(1,3);

//decompose E to P' , HZ (9.19)
{

if(determinant(R1)+1.0 < 1e-09) {
	//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
	cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
	E = -E;
	DecomposeEtoRandT(E,R1,R2,t1,t2);
}
if (!CheckCoherentRotation(R1)) {
	cout << "resulting rotation is not coherent\n";
	P1 = 0;
	return false;
}

P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
			 R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
			 R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
cout << "Testing P1 " << endl << Mat(P1) << endl;

vector<CloudPoint> pcloud,pcloud1; vector<KeyPoint> corresp;
double reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
double reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
vector<uchar> tmp_status;
//check if pointa are triangulated --in front-- of cameras for all 4 ambiguations
if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
	P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
				 R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
				 R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
	cout << "Testing P1 "<< endl << Mat(P1) << endl;

	pcloud.clear(); pcloud1.clear(); corresp.clear();
	reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
	reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
	
	if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
		if (!CheckCoherentRotation(R2)) {
			cout << "resulting rotation is not coherent\n";
			P1 = 0;
			return false;
		}
		
		P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
					 R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
					 R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
		cout << "Testing P1 "<< endl << Mat(P1) << endl;

		pcloud.clear(); pcloud1.clear(); corresp.clear();
		reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
		reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
		
		if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
			P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
						 R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
						 R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
			cout << "Testing P1 "<< endl << Mat(P1) << endl;

			pcloud.clear(); pcloud1.clear(); corresp.clear();
			reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
			reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
			
			if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
				cout << "Shit." << endl; 
				return false;
			}
		}				
	}			
}
for (unsigned int i=0; i<pcloud.size(); i++) {
	outCloud.push_back(pcloud[i]);
}
}		

t = ((double)getTickCount() - t)/getTickFrequency();
cout << "Done. (" << t <<"s)"<< endl;
}
return true;
}*/
}
