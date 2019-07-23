#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>

using std::cout;
using std::cerr;
using std::vector;
using std::string;

using cv::Mat;
using cv::Point2f;
using cv::KeyPoint;
using cv::Scalar;
using cv::Ptr;

using cv::FastFeatureDetector;
using cv::SimpleBlobDetector;

using cv::DMatch;
using cv::BFMatcher;
using cv::DrawMatchesFlags;
using cv::Feature2D;
using cv::ORB;
using cv::BRISK;
using cv::AKAZE;
using cv::KAZE;

using cv::xfeatures2d::BriefDescriptorExtractor;
using cv::xfeatures2d::SURF;
using cv::xfeatures2d::SIFT;
using cv::xfeatures2d::DAISY;
using cv::xfeatures2d::FREAK;

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;
const double subsamplingRatio = 0.45;

void detect_and_compute(string type, Mat& img, vector<KeyPoint>& kpts, Mat& desc) {
    if (type.find("fast") == 0) {
        type = type.substr(4);
        Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);
        detector->detect(img, kpts);
    }
    if (type.find("blob") == 0) {
        type = type.substr(4);
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();
        detector->detect(img, kpts);
    }
    if (type == "surf") {
        Ptr<Feature2D> surf = SURF::create(800.0);
        surf->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "sift") {
        Ptr<Feature2D> sift = SIFT::create();
        sift->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "orb") {
        Ptr<ORB> orb = ORB::create();
        orb->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "brisk") {
        Ptr<BRISK> brisk = BRISK::create();
        brisk->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "kaze") {
        Ptr<KAZE> kaze = KAZE::create();
        kaze->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "akaze") {
        Ptr<AKAZE> akaze = AKAZE::create();
        akaze->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "freak") {
        Ptr<FREAK> freak = FREAK::create();
        freak->compute(img, kpts, desc);
    }
    if (type == "daisy") {
        Ptr<DAISY> daisy = DAISY::create();
        daisy->compute(img, kpts, desc);
    }
    if (type == "brief") {
        Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(64);
        brief->compute(img, kpts, desc);
    }
}


void match(string type, Mat& desc1, Mat& desc2, vector<DMatch>& matches) {
    matches.clear();
    if (type == "bf") {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        desc_matcher.match(desc1, desc2, matches, Mat());
    }
    if (type == "knn") {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        vector< vector<DMatch> > vmatches;
        desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
        for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
            if (!vmatches[i].size()) {
                continue;
            }
            matches.push_back(vmatches[i][0]);
        }
    }
    std::sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }
}


int main(int argc, char** argv) {
    // Program expects at least four arguments:
    //   - descriptors type ("surf", "sift", "orb", "brisk", "kaze", "akaze", "freak", "daisy", "brief").
    //          
    //   For "brief", "freak" and "daisy" you also need a prefix that is either "blob" or "fast" (e.g. "fastbrief", "blobdaisy")
    if (argc != 5) {
        cerr << "\nError: wrong (you had: " << argc << ") number of arguments (should be 5).\n"; 
        cerr    << "Examples:\n"
                << argv[0] << " surf knn ../box.png ../box_in_scene.png\n"
                << argv[0] << " fastfreak bf ../box.png ../box_in_scene.png\n"
                << "\nNOTE: Not all of these methods are free, check licensing conditions!\n\n"
                << std::endl;
        exit(1);
    }

    string desc_type(argv[1]);
    string match_type(argv[2]);

    string img_file1(argv[3]);
    string img_file2(argv[4]);

    Mat img1 = cv::imread(img_file1, CV_LOAD_IMAGE_COLOR);
    Mat img2 = cv::imread(img_file2, CV_LOAD_IMAGE_COLOR);

    if (img1.channels() != 1) {
        cvtColor(img1, img1, cv::COLOR_RGB2GRAY);
    }

    if (img2.channels() != 1) {
        cvtColor(img2, img2, cv::COLOR_RGB2GRAY);
    }

    // Read input video
    cv::VideoCapture cap(img_file2);


    vector<KeyPoint> kpts1;
    vector<KeyPoint> kpts2;

    Mat desc1;
    Mat desc2;


    detect_and_compute(desc_type, img1, kpts1, desc1);

    Mat last_T;
    

    for (;;)
    {

        // Start timer
        double timer = cv::getTickCount();

        // Define variable for storing frames
        Mat curr, curr_gray;

        // Read next frame 
        bool success = cap.read(curr);
        if(!success) break; 

        // Convert frame to grayscale
        cv::cvtColor(curr, curr_gray, cv::COLOR_BGR2GRAY);

        detect_and_compute(desc_type, curr_gray, kpts2, desc2);


        vector<DMatch> matches;
        match(match_type, desc1, desc2, matches);

        vector<char> match_mask(matches.size(), 1);

        vector<Point2f> obj;
        vector<Point2f> scene;
        for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
            obj.push_back(kpts1[matches[i].queryIdx].pt);
            scene.push_back(kpts2[matches[i].trainIdx].pt);
        }

        Mat T = cv::findHomography(obj, scene, cv::RANSAC, 4, match_mask);
 
        // We'll just use the last known good transform.
        if(T.data == NULL) last_T.copyTo(T);
        T.copyTo(last_T);

        // Extract traslation
        double dx = T.at<double>(0,2);
        double dy = T.at<double>(1,2);
    
        // Extract rotation angle
        double da = atan2(T.at<double>(1,0), T.at<double>(0,0));


        vector<Point2f> obj_corners(4);
        vector<Point2f> scene_corners(4);

        obj_corners[0] = cvPoint(0,0);
        obj_corners[1] = cvPoint( img1.cols, 0 );
        obj_corners[2] = cvPoint( img1.cols, img1.rows ); // img.shape[0] shows the row of the image, img.shape[1] shows the column of the image
        obj_corners[3] = cvPoint( 0, img1.rows );
    
        cv::perspectiveTransform( obj_corners, scene_corners, T);
    
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( res, scene_corners[0] + Point2f( img1.cols, 0), scene_corners[1] + Point2f( img1.cols, 0), Scalar(0, 255, 0), 4 );
        line( res, scene_corners[1] + Point2f( img1.cols, 0), scene_corners[2] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
        line( res, scene_corners[2] + Point2f( img1.cols, 0), scene_corners[3] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
        line( res, scene_corners[3] + Point2f( img1.cols, 0), scene_corners[0] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );


        Mat res;
        cv::drawMatches(img1, kpts1, img2, kpts2, matches, res, Scalar::all(-1), Scalar::all(-1), match_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        int fps = cv::getTickFrequency() / (cv::getTickCount()-timer);
        
        // Display Decsriptor Type on the frame
        putText(res, "Decsriptor Type : " + desc_type, cvPoint(10,20), FONT_HERSHEY_SIMPLEX, 0.75, cvScalar(0,0,0), 2);


        // Display FPS on the frame
        string dsp = to_string(fps)
        putText(res, "FPS : " + fps, cvPoint(10,50), FONT_HERSHEY_SIMPLEX, 0.75, cvScalar(0,0,0), 2);


        // Display Good Matches Size on the frame
        string GoodMatches = to_string(matches.size())
        putText(res, "Good Matches : " + GoodMatches, cvPoint(10,50), FONT_HERSHEY_SIMPLEX, 0.75, cvScalar(0,0,0), 2);


        // Display Transfprmations on the frame
        string dX = to_string(cv::round(dx))
        string dY = to_string(cv::round(dy))
        string dA = to_string(cv::round(da))
        putText(res, "Transfprmations = [dx] : " + dX + "  " + "[dy] : " + dY, cvPoint(10,50), FONT_HERSHEY_SIMPLEX, 0.75, cvScalar(0,0,0), 2);


        cv::imshow("result", res);
        cv::waitKey(0);
    }

    return 0;
}