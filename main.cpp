#pragma once
#include<Eigen/Eigen>
#include<vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/common/common_headers.h>
#include <Eigen/Geometry>
#include <boost/make_shared.hpp>
#include <XnCppWrapper.h>
#include <boost/thread/thread.hpp>

using namespace cv;
using namespace std;
using namespace pcl;
using namespace Eigen;                             

/* Global Variables */

int img_width = 800;            // camera resolution
int img_height = 600;           // camera resolution
float laser_angle = M_PI/4;     // Angle between camera and laser line, equal 45 [deg]
float rot_angle = M_PI/50;      // turntable rotation step angle


PointCloud<PointXYZ> output_cloud, current_cloud;
PointCloud<PointXYZ>::Ptr ptr_output_cloud(new PointCloud<PointXYZ>);
PointCloud<PointXYZ>::Ptr transformed_cloud (new PointCloud<PointXYZ>);

/* Functions */

string type2str(int type)    // for Mat_debug
{    
  string r;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }
  r += "C";
  r += (chans+'0');
  return r;
}

void Mat_debug(Mat src, string s) 
{
    namedWindow( s, CV_WINDOW_AUTOSIZE );
    imshow(s,src);
    waitKey();
    string src_debug =  type2str(src.type());
    cout<<"Matrix: "<<s;
    printf(" %s %dx%d \n", src_debug.c_str(), src.cols, src.rows );
}

PointCloud<PointXYZ>::Ptr MatToPointXYZ(Mat image) 
{
    PointCloud<PointXYZ>::Ptr current_cloud(new PointCloud<PointXYZ>);
    for(int i=0;i<image.rows;i++)
        {
            for(int j=0;j<image.cols;j++)
                {
                    if (image.at<unsigned char>(i,j) == 255)                // if pixel is illuminated by laser
                    {
                        PointXYZ point;

                        point.x = j-440;                                     // simple trial and error of searching for the axis of rotation with regards to the camera sensor
                        point.y = i-300;
                        point.z = j-440;        //*tan(laser_angle);         // Triangulation
                        current_cloud -> points.push_back(point);
                    }
                }
        }
    current_cloud->width = (int)current_cloud->points.size();
    current_cloud->height = 1;
    return current_cloud;
}

void noise_filter(Mat &src) 
{
    int level = 50;
    threshold(src,src,level, 255, 3);
}

void color_filter(Mat &src) 
{
    Mat bgr[3];
    split(src,bgr);
    src = bgr [2];                      //Red channel
}

void edge_detection(Mat &src)
{
    Mat row, filtered;
    threshold(src,filtered,1, 0, 0);                                    // zeroing
    double min=0, max=0, slope = 0, slope_threshold = 0.34;
    int max_x;
    bool still_ground = true;
    Point min_loc, max_loc;
    for (int i = img_height-1; i >= 0; i--)
    {
        row = src.row(i);
        minMaxLoc(row,&min, &max, &min_loc, &max_loc);
        max_x=max_loc.x;
        slope = (double)(600-i)/max_x;
        if (i<550&&slope>slope_threshold) still_ground = false;
        if ( max_loc.x != 0 ) filtered.at<uchar>(i,max_loc.x) = 255;    //creating edge
    }
    src = filtered;

}

void debug_processing (Mat &src) 
{

    Mat_debug(src,"Original image");
    color_filter(src);                   // Red separation
    Mat_debug(src,"Red Filtered");
    noise_filter(src);                   // Noise removal
    Mat_debug(src,"Noise Filtered");
    edge_detection(src);
    Mat_debug(src,"Edge Detected");
}

void processing (Mat &src)
{
    color_filter(src);                   // Laser color separation
    noise_filter(src);                   // Noise removal
    edge_detection(src);
}

int main(int argc, char** argv)
{
    Eigen::Matrix4f trans_matrix = Eigen::Matrix4f::Identity();     // Transformation matrix
    std::cout <<"ID matrix\n" << trans_matrix.matrix() << std::endl;
    trans_matrix (0,0) = cos (rot_angle);                           // Rotation coefficients
    trans_matrix (0,2) = sin(rot_angle);
    trans_matrix (2,0) = -sin (rot_angle);
    trans_matrix (2,2) = cos (rot_angle);
    std::cout << "Rotation matrix\n"<< trans_matrix.matrix() << std::endl;
    VideoCapture cap("LASER/sd_spline-%d.jpg");                     // folder with laser scans 
    for (int i=0 ; i<(int)2*M_PI/rot_angle ; i++)
    {
        Mat src;
        cap.read(src);
        if (i==0) debug_processing(src);
        else processing(src);
        output_cloud += *MatToPointXYZ(src);
        transformPointCloud (output_cloud, output_cloud, trans_matrix);
    }
    ptr_output_cloud = output_cloud.makeShared();
    visualization::CloudViewer cloud_viewer("3D Point Cloud");
    cloud_viewer.showCloud(ptr_output_cloud);
    while (!cloud_viewer.wasStopped ()){}
    io::savePCDFileASCII ("point_cloud.pcd", output_cloud);
    io::savePCDFileASCII ("point_cloud.xyz", output_cloud);             // or .pcd
    std::cout << "Saved " << output_cloud.points.size () << " data points to point_cloud.xyz and point_cloud.pcd" << std::endl;
    waitKey();
    return 0;
}

