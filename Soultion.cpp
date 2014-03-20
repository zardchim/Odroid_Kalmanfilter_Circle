//OpenCV library
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/video/tracking.hpp>
//Standard C++ library
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <time.h> // to calculate time needed
#include <limits.h> // to get INT_MAX, to protect against overflow


using namespace cv;
using namespace std;

#define DEBUG			//Output data in dos window
#define kalmanfilter	//Apply kalman filter in program 
#define imgDEBUG		//Output image to debug

float sequence_counter=0;
float succeed_rate=0;
Point average,average_3=(0,0);	//average=average of predicted pts and real pts		average_3=average of real pts
double SQ_TRI_Len_3=1;			//square of length in triangle with all real pts

CvPoint2D32f rotation(CvPoint2D32f a,CvPoint2D32f b,bool clockwise){		//function to determind thrid point
	CvPoint2D32f ans;
	float SQRT3=sqrt((float)3);
	CvPoint2D32f a_offset={a.x-b.x,a.y-b.y};
	if (clockwise==1){
		ans.x=a_offset.x/2+SQRT3*a_offset.y/2;
		ans.y=a_offset.y/2-SQRT3*a_offset.x/2;
	}
	else{
		ans.x=a_offset.x/2-SQRT3*a_offset.y/2;
		ans.y=a_offset.y/2+SQRT3*a_offset.x/2;
	}
	ans.x+=b.x;
	ans.y+=b.y;
	return ans;
}

double LenSQ (CvPoint2D32f a,CvPoint2D32f b){	//square of length in two pts
	return ((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}

CvPoint2D32f translation(CvPoint2D32f a,CvPoint2D32f b,double SQdis,bool right){	//predict pt base on 1 pt
	CvPoint2D32f ans;
	double SQlength=LenSQ(a,b);
	if (right==1){
		ans.x=a.x+sqrt(SQdis/SQlength)*(b.x-a.x);
		ans.y=a.y+sqrt(SQdis/SQlength)*(b.y-a.y);
	}
	else{
		ans.x=a.x-sqrt(SQdis/SQlength)*(b.x-a.x);
		ans.y=a.y-sqrt(SQdis/SQlength)*(b.y-a.y);
	}
	return ans;
}


int main(int argc, char *argv[])
{
 
    VideoCapture cap(1); 
	cap.set(CV_CAP_PROP_FRAME_WIDTH ,640);	//video properties (640*480)
	cap.set(CV_CAP_PROP_FRAME_HEIGHT ,480);
    if(!cap.isOpened()) //
    {
        cout << "Couldn't open Video  " ; 
        return -1; 
    }
    //SimpleBlobDetector_Parameter
        SimpleBlobDetector::Params params;
        params.thresholdStep = 50;
        params.minThreshold = 0;
        params.maxThreshold = 200;
        params.filterByArea=true;  
		params.minArea = 300; 
        params.maxArea = 60000;
		params.filterByColor=true;
		params.blobColor=0;
		params.filterByCircularity=true;
		params.minCircularity=(float)0.85;
		params.maxCircularity=1.25;
		params.filterByConvexity=true;
		params.maxConvexity=10;
	//Variabe
	#ifdef DEBUG
	//Time Setting
	time_t start , end , effend ;
	double sec;
	time(&start);
	#endif

	#ifdef kalmanfilter 
	//karman filter 
	KalmanFilter KF(4, 2, 0);
	Mat_<float> state(4, 1);
	Mat_<float> processNoise(4, 1, CV_32F);
	Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));

	KF.statePre.at<float>(0) = 0;
	KF.statePre.at<float>(1) = 0;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;

	KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1); // Including velocity
	KF.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,  0,0,0,0.3);

	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(.1));
	
	vector<Point> mousev,kalmanv;
	mousev.clear();
	kalmanv.clear();
	#endif
	for(;;)  // video capture for loop
    {
        Mat frame,labelImg; //thresholdStep
        cap >> frame; 
        if(frame.empty()) break;  
        
		Mat out;
        SimpleBlobDetector blobDetector( params );//Find keyPoints
        blobDetector.create("SimpleBlob""FAST");
		vector<cv::KeyPoint> keyPoints;
        blobDetector.detect( frame, keyPoints );	
		
		#ifdef imgDEBUG
			drawKeypoints( frame, keyPoints, frame, CV_RGB(255,0,0), DrawMatchesFlags::DEFAULT);
		#endif
		
		double average_x=0,average_y=0;
        //find average
		for(int i=0; i<keyPoints.size(); i++){
            average_x += keyPoints[i].pt.x;
            average_y += keyPoints[i].pt.y;
        }
		if(keyPoints.size()==1){//predict pt base on 1 pt
			CvPoint2D32f R_tri=translation(keyPoints[0].pt,average_3,SQ_TRI_Len_3,1);
			CvPoint2D32f L_tri=translation(keyPoints[0].pt,average_3,SQ_TRI_Len_3,0);
			if (LenSQ(R_tri,average_3)<LenSQ(L_tri,average_3)){
				average_x+=R_tri.x;
				average_y+=R_tri.y;
				#ifdef imgDEBUG
					circle(frame,R_tri,5,Scalar(128,128,0),3);
					circle(frame,average_3,5,Scalar(128,128,128),3);
				#endif
			}
			else{
				average_x+=L_tri.x;
				average_y+=L_tri.y;
				#ifdef imgDEBUG
					circle(frame,L_tri,5,Scalar(128,128,0),3);
					circle(frame,average_3,5,Scalar(128,128,128),3);
				#endif
			}
		}
		
		if(keyPoints.size()==2){//predict third pt base on 2 pts
			CvPoint2D32f cw_tri=rotation(keyPoints[0].pt,keyPoints[1].pt,1);
			CvPoint2D32f ccw_tri=rotation(keyPoints[0].pt,keyPoints[1].pt,0);
			if (LenSQ(cw_tri,average_3)<LenSQ(ccw_tri,average_3)){
				average_x+=cw_tri.x;
				average_y+=cw_tri.y;
				#ifdef imgDEBUG
					circle(frame,cw_tri,5,Scalar(128,128,0),3);
				#endif
			}
			else{
				average_x+=ccw_tri.x;
				average_y+=ccw_tri.y;
				#ifdef imgDEBUG
					circle(frame,ccw_tri,5,Scalar(128,128,0),3);
				#endif
			}
		}	
		//divide and storage proper value	
		if (keyPoints.size()==2 || keyPoints.size() ==3){ 
				average_x/=3;//keyPoints.size();
				average_y/=3;//keyPoints.size();
				average.x=average_x;
				average.y=average_y;
				SQ_TRI_Len_3=LenSQ(keyPoints[0].pt,keyPoints[1].pt)*3/4;
				average_3=average;
				succeed_rate++;
		}
		if (keyPoints.size()==1){
				average_x/=2;//keyPoints.size();
				average_y/=2;//keyPoints.size();
				average.x=average_x;
				average.y=average_y;
				average_3=average;
				succeed_rate++;
		}
		#ifdef DEBUG
				time(&effend);
				sec=difftime(effend,start);
				cout<<"Succeed: ";
				cout<<(succeed_rate/sequence_counter*100);
				cout<<"  No. PT: ";
				cout<<keyPoints.size();
				cout<<"  X is: ";
				cout<<average.x;
				cout<<"  Y is: ";
				cout<<average.y;
				cout<<"  Eff FPS: ";
				cout<<succeed_rate/sec;
				cout<<endl;
				sequence_counter++;
				//Time show
				time(&end);
				sec = difftime(end, start);
				cout<<"Process time: "<<sequence_counter/sec<<endl;
		#endif

		#ifdef kalmanfilter 
				//Karman filter 
				if (keyPoints.size() == 3 || keyPoints.size() == 2||keyPoints.size() ==1) mousev.push_back(average);
				// First predict, to update the internal statePre variable
				Mat prediction = KF.predict();
				Point predictPt((int)prediction.at<float>(0),(int)prediction.at<float>(1));
             
				// Get average point
				measurement(0) = average.x;
				measurement(1) = average.y; 
				Point measPt(measurement(0),measurement(1));
 
				// The "correct" phase that is going to use the predicted value and our measurement
				if (keyPoints.size() == 3 || keyPoints.size() == 2||keyPoints.size() ==1) {
					Mat estimated = KF.correct(measurement);
					Point statePt(estimated.at<float>(0),estimated.at<float>(1));
					kalmanv.push_back(statePt);
				}
		#endif

		#ifdef imgDEBUG
			#ifdef kalmanfilter
				circle(frame,predictPt,5,Scalar(255,0,0),3);
			#endif
			circle(frame,average,5,Scalar(0,0,255),3);
			imshow("main",frame);  
		#endif

		//More stuff should be handle the serial control
		//Valueshould be using statePt
		//


		if (cvWaitKey(1)==27) break;  // ESC will break
		}
    system("pause");

}