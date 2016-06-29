////////////////////////////////////////////////
//
//  Drive a car in TORCS with two-lane scenario.
//  Used with TORCS driver agent "chenyi" (for two-lane), and modified tracks with two-lane pavement textures.
//  Must configure other traffic cars (chenyi_AIx) to drive in two-lane setting before run the code.
//
//  Input keys 
//  Esc: exit
//  P: pause (initially paused)
//  C: autonomous driving on/off (top-down view visualiation turns red when manaully controling the host car)
//  arrows: manually override steering, acceleration, and brake
//
////////////////////////////////////////////////

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <unistd.h>  
#include <stdlib.h>  
#include <stdio.h>  
#include <sys/shm.h>
#include <cuda_runtime.h>
#include <cstring>

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#define image_width 640
#define image_height 480
#define resize_width 280
#define resize_height 210
#define semantic_width 320
#define semantic_height 660

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

struct shared_use_st  
{  
    int written;  //a label, if 1: available to read, if 0: available to write
    uint8_t data[image_width*image_height*3];  // image data field 
    int control;
    int pause;
    double fast;

    double dist_L;
    double dist_R;

    double toMarking_L;
    double toMarking_M;
    double toMarking_R;

    double dist_LL;
    double dist_MM;
    double dist_RR;

    double toMarking_LL;
    double toMarking_ML;
    double toMarking_MR;
    double toMarking_RR;

    double toMiddle;
    double angle;
    double speed;

    double steerCmd;
    double accelCmd;
    double brakeCmd;
};

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);

    ////////////////////// set up memory sharing
    void *shm = NULL;
    struct shared_use_st *shared;  
    int shmid;

    shmid = shmget((key_t)4567, sizeof(struct shared_use_st), 0666|IPC_CREAT);  
    if(shmid == -1)  
    {  
        fprintf(stderr, "shmget failed\n");  
        exit(EXIT_FAILURE);  
    }  
 
    shm = shmat(shmid, 0, 0);  
    if(shm == (void*)-1)  
    {  
        fprintf(stderr, "shmat failed\n");  
        exit(EXIT_FAILURE);  
    }  
    printf("\n********** Memory sharing started, attached at %X **********\n", shm); 

    shared = (struct shared_use_st*)shm;  
    shared->written = 0;
    shared->control = 0;
    shared->pause = 0;
    shared->fast = 0.0;

    shared->dist_L = 0.0;
    shared->dist_R = 0.0;

    shared->toMarking_L = 0.0;
    shared->toMarking_M = 0.0;
    shared->toMarking_R = 0.0;

    shared->dist_LL = 0.0;
    shared->dist_MM = 0.0;
    shared->dist_RR = 0.0;

    shared->toMarking_LL = 0.0;
    shared->toMarking_ML = 0.0;
    shared->toMarking_MR = 0.0;
    shared->toMarking_RR = 0.0;

    shared->toMiddle = 0.0;
    shared->angle = 0.0;
    shared->speed = 0.0;

    shared->steerCmd = 0.0;
    shared->accelCmd = 0.0;
    shared->brakeCmd = 0.0;  
    ////////////////////// END set up memory sharing

    ////////////////////// set up opencv
    IplImage* screenRGB=cvCreateImage(cvSize(image_width,image_height),IPL_DEPTH_8U,3);
    IplImage* resizeRGB=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);
    IplImage* semanticRGB=cvCreateImage(cvSize(semantic_width,semantic_height),IPL_DEPTH_8U,3);
    IplImage* error_bar=cvCreateImage(cvSize(640,180),IPL_DEPTH_8U,3);
    IplImage* legend=cvLoadImage("../torcs/Legend6.png");
    IplImage* background=cvLoadImage("../torcs/semantic_background_2lane.png");
    cvNamedWindow("Semantic Visualization",1);
    cvNamedWindow("Image from leveldb",1);
    cvNamedWindow("Error Bar",1);
    int key;

    CvFont font;    
    cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1, 1, 1, 2, 8);  
    char vi_buf[4];
    ////////////////////// END set up opencv

    ////////////////////// set up leveldb  
    leveldb::Options options;
    options.error_if_exists = false;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    options.max_open_files = 100;

    leveldb::DB* db;
    LOG(INFO) << "Opening leveldb: Current_State_1F";
    leveldb::Status status = leveldb::DB::Open(options, "Current_State_1F", &db);
    CHECK(status.ok()) << "Failed to open leveldb: Current_State_1F";

    Datum datum;
    string value;
    
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];

    leveldb::WriteBatch* batch = new leveldb::WriteBatch();
    ////////////////////// END set up leveldb
    
    ////////////////////// set up Caffe
    if (argc < 3) {
      LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations " << "[CPU/GPU]";
      return 0;
    }

    cudaSetDevice(0);
    Caffe::set_phase(Caffe::TEST);

    if (argc == 4 && strcmp(argv[3], "GPU") == 0) {
      LOG(ERROR) << "Using GPU";
      Caffe::set_mode(Caffe::GPU);
    } else {
      LOG(ERROR) << "Using CPU";
      Caffe::set_mode(Caffe::CPU);
    }

    NetParameter test_net_param;
    ReadProtoFromTextFile(argv[1], &test_net_param);
    Net<float> caffe_test_net(test_net_param, db);
    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(argv[2], &trained_net_param);
    caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

    vector<Blob<float>*> dummy_blob_input_vec;
    ////////////////////// END set up Caffe

    ////////////////////// cnn output parameters
    float true_angle;
    int true_fast;

    float true_dist_L;
    float true_dist_R;

    float true_toMarking_L;
    float true_toMarking_M;
    float true_toMarking_R;

    float true_dist_LL;
    float true_dist_MM;
    float true_dist_RR;

    float true_toMarking_LL;
    float true_toMarking_ML;
    float true_toMarking_MR;
    float true_toMarking_RR;

    float angle;
    int fast;

    float dist_L;
    float dist_R;

    float toMarking_L;
    float toMarking_M;
    float toMarking_R;

    float dist_LL;
    float dist_MM;
    float dist_RR;

    float toMarking_LL;
    float toMarking_ML;
    float toMarking_MR;
    float toMarking_RR;
    ////////////////////// END cnn output parameters

    ////////////////////// control parameters
    float road_width=8.0;
    float center_line;
    float coe_steer=1.0;
    int lane_change=0;
    float pre_ML;
    float pre_MR;
    float desired_speed;
    float steering_record[5]={0,0,0,0,0};
    int steering_head=0;
    float slow_down=100;
    float dist_LL_record=30;
    float dist_RR_record=30;

    int left_clear=0;
    int right_clear=0;
    int left_timer=0;
    int right_timer=0;
    int timer_set=60;

    float pre_dist_L=60;
    float pre_dist_R=60;
    float steer_trend;
    ////////////////////// END control parameters

    ////////////////////// visualization parameters
    int marking_head=1;
    int marking_st;
    int marking_end;
    int pace;
    int car_pos;

    float p1_x,p1_y,p2_x,p2_y,p3_x,p3_y,p4_x,p4_y;
    CvPoint* pt = new CvPoint[4]; 
    int visualize_angle=1;

    float err_angle;
    float err_toMarking_ML;
    float err_toMarking_MR;
    float err_toMarking_M;

    int manual=0;
    int counter=0;
    ////////////////////// END visualization parameters

    while (1) {

        if (shared->written == 1) {  // the new image data is ready to be read

            for (int h = 0; h < image_height; h++) {
               for (int w = 0; w < image_width; w++) {
                  screenRGB->imageData[(h*image_width+w)*3+2]=shared->data[((image_height-h-1)*image_width+w)*3+0];
                  screenRGB->imageData[(h*image_width+w)*3+1]=shared->data[((image_height-h-1)*image_width+w)*3+1];
                  screenRGB->imageData[(h*image_width+w)*3+0]=shared->data[((image_height-h-1)*image_width+w)*3+2];
               }
            }

            cvResize(screenRGB,resizeRGB);

            /////////////////////////////// get the groundtruth value at the same time when we extract the image
            true_angle = shared->angle;           
            true_fast = int(shared->fast);

            true_dist_L = shared->dist_L;
            true_dist_R = shared->dist_R;

            true_toMarking_L = shared->toMarking_L;
            true_toMarking_M = shared->toMarking_M;
            true_toMarking_R = shared->toMarking_R;

            true_dist_LL = shared->dist_LL;
            true_dist_MM = shared->dist_MM;
            true_dist_RR = shared->dist_RR;

            true_toMarking_LL = shared->toMarking_LL;
            true_toMarking_ML = shared->toMarking_ML;
            true_toMarking_MR = shared->toMarking_MR;
            true_toMarking_RR = shared->toMarking_RR;
            /////////////////////////////// END get the groundtruth value at the same time when we extract the image

            ///////////////////////////// set caffe input
            datum.set_channels(3);
            datum.set_height(resize_height);
            datum.set_width(resize_width);
            datum.set_label(0); 
            datum.clear_data();
            datum.clear_float_data();
            string* datum_string = datum.mutable_data();

            for (int c = 0; c < 3; ++c) {
              for (int h = 0; h < resize_height; ++h) {
                for (int w = 0; w < resize_width; ++w) {
                  datum_string->push_back(static_cast<char>(resizeRGB->imageData[(h*resize_width+w)*3+c]));
                }
              }
            }

            datum.SerializeToString(&value);

            for (int i=1; i<=1; i++) {
               snprintf(key_cstr, kMaxKeyLength, "%08d", i);
               batch->Put(string(key_cstr), value);
            }
            db->Write(leveldb::WriteOptions(), batch);
            delete batch;
            batch = new leveldb::WriteBatch();
            ///////////////////////////// END set caffe input

            /////////////////////////////////////////////////////////// run deep learning CNN for one step to process the image
            caffe_test_net.ForwardData();
            const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_blob_input_vec);  

            const float* result_data = result[0]->cpu_data();

            ////// get output from cnn
            angle=(result_data[0]-0.5)*1.1;

            toMarking_L=(result_data[1]-1.34445)*5.6249;
            toMarking_M=(result_data[2]-0.39091)*6.8752;
            toMarking_R=(result_data[3]+0.34445)*5.6249;

            dist_L=(result_data[4]-0.12)*95;
            dist_R=(result_data[5]-0.12)*95;

            toMarking_LL=(result_data[6]-1.48181)*6.8752;
            toMarking_ML=(result_data[7]-0.98)*6.25;
            toMarking_MR=(result_data[8]-0.02)*6.25;
            toMarking_RR=(result_data[9]+0.48181)*6.8752;

            dist_LL=(result_data[10]-0.12)*95;
            dist_MM=(result_data[11]-0.12)*95;
            dist_RR=(result_data[12]-0.12)*95;

            if (result_data[13]>0.5) fast=1;
            else fast=0;
            /////////////////////////////////////////////////////////// END run deep learning CNN for one step to process the image 

            //////////////////////////////////////////////// a controller processes the cnn output and get the optimal steering, acceleration/brake
            if (shared->control==1) {

               slow_down=100; 

               if (pre_dist_L<20 && dist_LL<20) {   // left lane is occupied or not
                   left_clear=0;
                   left_timer=0;
               } else left_timer++;                
        
               if (pre_dist_R<20 && dist_RR<20) {   // right lane is occupied or not
                   right_clear=0;
                   right_timer=0;
               } else right_timer++;

               pre_dist_L=dist_LL;
               pre_dist_R=dist_RR;

               if (left_timer>timer_set) {   // left lane is clear
                  left_timer=timer_set;
                  left_clear=1;
               }

               if (right_timer>timer_set) {   // right lane is clear
                  right_timer=timer_set;
                  right_clear=1;
               }
      

               if (lane_change==0 && dist_MM<15) {   // if current lane is occupied

                  steer_trend=steering_record[0]+steering_record[1]+steering_record[2]+steering_record[3]+steering_record[4];   // am I turning or not

                  if (toMarking_LL>-8 && left_clear==1 && steer_trend>=0) {   // move to left lane
                     lane_change=-2;
                     coe_steer=6;
                     right_clear=0;
                     right_timer=0;
                     left_clear=0;
                     left_timer=0;
                     timer_set=30;
                  }

                  else if (toMarking_RR<8 && right_clear==1 && steer_trend<=0) {   // move to right lane
                     lane_change=2;
                     coe_steer=6;
                     left_clear=0;
                     left_timer=0;
                     right_clear=0;
                     right_timer=0;
                     timer_set=30;
                  }

                  else {
                     float v_max=20;
                     float c=2.772;
                     float d=-0.693;
                     slow_down=v_max*(1-exp(-c/v_max*dist_MM-d));  // optimal vilcity car-following model
                     if (slow_down<0) slow_down=0;
                  }
               }
 
               ///////////////////////////////////////////////// prefer to stay in the right lane
               else if (lane_change==0 && dist_MM>=15) {

                  steer_trend=steering_record[0]+steering_record[1]+steering_record[2]+steering_record[3]+steering_record[4];  // am I turning or not

                  if (toMarking_LL<-8 && right_clear==1 && steer_trend<=0 && steer_trend>-0.2) {  // in left lane, move to right lane
                     lane_change=2;
                     coe_steer=6;
                     right_clear=0;
                     right_timer=20;
                  }
               }
               ///////////////////////////////////////////////// END prefer to stay in the right lane

               ///////////////////////////////////////////////// implement lane changing or car-following
               if (lane_change==0) {
                  if (-toMarking_ML+toMarking_MR<5.5) {
                     coe_steer=1.5;
                     center_line=(toMarking_ML+toMarking_MR)/2;
                     pre_ML=toMarking_ML;
                     pre_MR=toMarking_MR;
                     if (toMarking_M<1)
                        coe_steer=0.4;
                  } else {
                     if (-pre_ML>pre_MR)
                        center_line=(toMarking_L+toMarking_M)/2;
                     else
                        center_line=(toMarking_R+toMarking_M)/2;
                     coe_steer=0.3;
                  }
               }

               else if (lane_change==-2) {
                  if (-toMarking_ML+toMarking_MR<5.5) {
                     center_line=(toMarking_LL+toMarking_ML)/2;
                     if (toMarking_L>-5 && toMarking_M<1.5)
                        center_line=(center_line+(toMarking_L+toMarking_M)/2)/2;
                  } else {
                     center_line=(toMarking_L+toMarking_M)/2;
                     coe_steer=20;
                     lane_change=-1;
                  }
               }

               else if (lane_change==-1) {
                  if (toMarking_L>-5 && toMarking_M<1.5) {
                     center_line=(toMarking_L+toMarking_M)/2;
                     if (-toMarking_ML+toMarking_MR<5.5)
                        center_line=(center_line+(toMarking_ML+toMarking_MR)/2)/2;
                  } else {
                     center_line=(toMarking_ML+toMarking_MR)/2;
                     lane_change=0;
                  }
               }

               else if (lane_change==2) {
                  if (-toMarking_ML+toMarking_MR<5.5) {
                     center_line=(toMarking_RR+toMarking_MR)/2;
                     if (toMarking_R<5 && toMarking_M<1.5)
                        center_line=(center_line+(toMarking_R+toMarking_M)/2)/2;
                  } else {
                     center_line=(toMarking_R+toMarking_M)/2;
                     coe_steer=20;
                     lane_change=1;
                  }
               }

               else if (lane_change==1) {
                  if (toMarking_R<5 && toMarking_M<1.5) {
                     center_line=(toMarking_R+toMarking_M)/2;
                     if (-toMarking_ML+toMarking_MR<5.5)
                        center_line=(center_line+(toMarking_ML+toMarking_MR)/2)/2;
                  } else {
                     center_line=(toMarking_ML+toMarking_MR)/2;
                     lane_change=0;
                  }
               }
               ///////////////////////////////////////////////// END implement lane changing or car-following

               shared->steerCmd = (angle - center_line/road_width) / 0.541052/coe_steer;   // steering control, "shared->steerCmd" [-1,1] is the value sent back to TORCS
 
               if (lane_change==0 && coe_steer>1 && shared->steerCmd>0.1)   // reshape the steering control curve
                  shared->steerCmd=shared->steerCmd*(2.5*shared->steerCmd+0.75);

               steering_record[steering_head]=shared->steerCmd;   // update previous steering record
               steering_head++;
               if (steering_head==5) steering_head=0;


               if (fast==1) desired_speed=20;
               else desired_speed=20-fabs(steering_record[0]+steering_record[1]+steering_record[2]+steering_record[3]+steering_record[4])*4.5;
               if (desired_speed<10) desired_speed=10;

               if (slow_down<desired_speed) desired_speed=slow_down;

               ///////////////////////////// speed control           
               if (desired_speed>=shared->speed) {
                   shared->accelCmd = 0.2*(desired_speed-shared->speed+1);
                   if (shared->accelCmd>1) shared->accelCmd=1.0;
                   shared->brakeCmd = 0.0;
               } else {
                   shared->brakeCmd = 0.1*(shared->speed-desired_speed);
                   if (shared->brakeCmd>1) shared->brakeCmd=1.0;
                   shared->accelCmd = 0.0;
               }
               ///////////////////////////// END speed control

            }

            printf("M_LL:%.2lf, M_ML:%.2lf, M_MR:%.2lf, M_RR:%.2lf, d_LL:%.2lf, d_MM:%.2lf, d_RR:%.2lf\n", toMarking_LL, toMarking_ML, toMarking_MR, toMarking_RR, dist_LL, dist_MM, dist_RR);
            printf("M_L:%.2lf, M_M:%.2lf, M_R:%.2lf, d_L:%.2lf, d_R:%.2lf, angle:%.3lf, fast:%d\n", toMarking_L, toMarking_M, toMarking_R, dist_L, dist_R, angle, fast);
            printf("coe_steer:%.1lf, lane_change:%d, steer:%.2lf, d_speed:%d, speed:%d, l_clear:%d, r_clear:%d, timer_set:%d\n\n", coe_steer, lane_change, shared->steerCmd, int(desired_speed*3.6), int(shared->speed*3.6), left_clear, right_clear, timer_set);
            fflush(stdout);
            //////////////////////////////////////////////// END a controller processes the cnn output and get the optimal steering, acceleration/brake

            //////////////////////////////////////////////// show legend and error bar
            err_angle=(angle-true_angle)*343.8; // full scale +-12 degree
            if (err_angle>72) err_angle=72;
            if (err_angle<-72) err_angle=-72;

            if (true_toMarking_ML>-5 && -toMarking_ML+toMarking_MR<5.5) {
               err_toMarking_ML=(toMarking_ML-true_toMarking_ML)*72; // full scale +-1 meter
               if (err_toMarking_ML>72) err_toMarking_ML=72;
               if (err_toMarking_ML<-72) err_toMarking_ML=-72;
               err_toMarking_MR=(toMarking_MR-true_toMarking_MR)*72; // full scale +-1 meter
               if (err_toMarking_MR>72) err_toMarking_MR=72;
               if (err_toMarking_MR<-72) err_toMarking_MR=-72;
            } else {
               err_toMarking_ML=0;
               err_toMarking_MR=0;
            }

            if (true_toMarking_M<3 && toMarking_M<2) {
               err_toMarking_M=(toMarking_M-true_toMarking_M)*72; // full scale +-1 meter
               if (err_toMarking_M>72) err_toMarking_M=72;
               if (err_toMarking_M<-72) err_toMarking_M=-72;
            } else {
               err_toMarking_M=0;
            }

            int bar_st=26;
            cvCopy(legend,error_bar);
            cvRectangle(error_bar,cvPoint(319,bar_st+0-10),cvPoint(319+err_angle,bar_st+0+10),cvScalar(127,127,127),-1);
            cvRectangle(error_bar,cvPoint(319,bar_st+42-10),cvPoint(319+err_toMarking_ML,bar_st+42+10),cvScalar(190,146,122),-1);
            cvRectangle(error_bar,cvPoint(319,bar_st+84-10),cvPoint(319+err_toMarking_MR,bar_st+84+10),cvScalar(0,121,0),-1);
            cvRectangle(error_bar,cvPoint(319,bar_st+126-10),cvPoint(319+err_toMarking_M,bar_st+126+10),cvScalar(128,0,0),-1);

            cvLine(error_bar,cvPoint(319,bar_st-15),cvPoint(319,bar_st+144),cvScalar(0,0,0),1);

            cvShowImage("Error Bar",error_bar);
            //////////////////////////////////////////////// END show legend and error bar

            ///////////////////////////// semantic visualization
            cvCopy(background,semanticRGB);

            pace=int(shared->speed*1.2);
            if (pace>50) pace=50;

            marking_head=marking_head+pace;
            if (marking_head>0) marking_head=marking_head-110;
            else if (marking_head<-110) marking_head=marking_head+110;

            marking_st=marking_head;
            marking_end=marking_head+55;

            while (marking_st<=660) {
                cvLine(semanticRGB,cvPoint(150,marking_st),cvPoint(150,marking_end),cvScalar(255,255,255),2);
                marking_st=marking_st+110;
                marking_end=marking_end+110;
            }

            sprintf(vi_buf,"%d",int(shared->speed*3.6));
            cvPutText(semanticRGB,vi_buf,cvPoint(245,85),&font,cvScalar(255,255,255));

            //////////////// visualize true_angle
            if (visualize_angle==1) {
               true_angle=-true_angle;
               p1_x=-14*cos(true_angle)+28*sin(true_angle);
               p1_y=14*sin(true_angle)+28*cos(true_angle);
               p2_x=14*cos(true_angle)+28*sin(true_angle);
               p2_y=-14*sin(true_angle)+28*cos(true_angle);
               p3_x=14*cos(true_angle)-28*sin(true_angle);
               p3_y=-14*sin(true_angle)-28*cos(true_angle);
               p4_x=-14*cos(true_angle)-28*sin(true_angle);
               p4_y=14*sin(true_angle)-28*cos(true_angle);
            }
            //////////////// END visualize true_angle

            /////////////////// draw groundtruth data
            if (true_toMarking_LL>-9) {     // right lane
  
               if (true_toMarking_M<2 && true_toMarking_R>6.5)
                   car_pos=int((174-(true_toMarking_ML+true_toMarking_MR)*6+198-true_toMarking_M*12)/2);
               else if (true_toMarking_M<2 && true_toMarking_R<6.5)
                   car_pos=int((174-(true_toMarking_ML+true_toMarking_MR)*6+150-true_toMarking_M*12)/2);
               else
                   car_pos=int(174-(true_toMarking_ML+true_toMarking_MR)*6);

               if (visualize_angle==1) {
                  pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                  pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                  pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                  pt[3] = cvPoint(p4_x+car_pos,p4_y+600);
                  cvFillConvexPoly(semanticRGB,pt,4,cvScalar(0,0,255));
               } else
                  cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,0,255),-1);

               if (true_dist_LL<60)
                  cvRectangle(semanticRGB,cvPoint(126-14,600-true_dist_LL*12-28),cvPoint(126+14,600-true_dist_LL*12+28),cvScalar(0,255,255),-1);
               if (true_dist_MM<60)
                  cvRectangle(semanticRGB,cvPoint(174-14,600-true_dist_MM*12-28),cvPoint(174+14,600-true_dist_MM*12+28),cvScalar(0,255,255),-1);
            }

            else if (true_toMarking_RR<9) {   // left lane

               if (true_toMarking_M<2 && true_toMarking_L<-6.5)
                   car_pos=int((126-(true_toMarking_ML+true_toMarking_MR)*6+102-true_toMarking_M*12)/2);
               else if (true_toMarking_M<2 && true_toMarking_L>-6.5)
                   car_pos=int((126-(true_toMarking_ML+true_toMarking_MR)*6+150-true_toMarking_M*12)/2);
               else
                   car_pos=int(126-(true_toMarking_ML+true_toMarking_MR)*6);

               if (visualize_angle==1) {
                  pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                  pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                  pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                  pt[3] = cvPoint(p4_x+car_pos,p4_y+600);
                  cvFillConvexPoly(semanticRGB,pt,4,cvScalar(0,0,255));
               } else
                  cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,0,255),-1);

               if (true_dist_MM<60)
                  cvRectangle(semanticRGB,cvPoint(126-14,600-true_dist_MM*12-28),cvPoint(126+14,600-true_dist_MM*12+28),cvScalar(0,255,255),-1);
               if (true_dist_RR<60)
                  cvRectangle(semanticRGB,cvPoint(174-14,600-true_dist_RR*12-28),cvPoint(174+14,600-true_dist_RR*12+28),cvScalar(0,255,255),-1);
            }

            else if (true_toMarking_M<3) {
                if (true_toMarking_L<-6.5) {   // left
                   car_pos=int(102-true_toMarking_M*12);
                   if (true_dist_R<60)
                      cvRectangle(semanticRGB,cvPoint(126-14,600-true_dist_R*12-28),cvPoint(126+14,600-true_dist_R*12+28),cvScalar(0,255,255),-1);
                } else if (true_toMarking_R>6.5) {  // right
                   car_pos=int(198-true_toMarking_M*12);
                   if (true_dist_L<60)
                      cvRectangle(semanticRGB,cvPoint(174-14,600-true_dist_L*12-28),cvPoint(174+14,600-true_dist_L*12+28),cvScalar(0,255,255),-1);
                } else {
                   car_pos=int(150-true_toMarking_M*12);
                   if (true_dist_L<60)
                      cvRectangle(semanticRGB,cvPoint(126-14,600-true_dist_L*12-28),cvPoint(126+14,600-true_dist_L*12+28),cvScalar(0,255,255),-1);
                   if (true_dist_R<60)
                      cvRectangle(semanticRGB,cvPoint(174-14,600-true_dist_R*12-28),cvPoint(174+14,600-true_dist_R*12+28),cvScalar(0,255,255),-1);
                }

                if (visualize_angle==1) {
                   pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                   pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                   pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                   pt[3] = cvPoint(p4_x+car_pos,p4_y+600);
                   cvFillConvexPoly(semanticRGB,pt,4,cvScalar(0,0,255));
                } else
                   cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,0,255),-1);
            }
            /////////////////// END draw groundtruth data

            //////////////// visualize angle
            if (visualize_angle==1) {
               angle=-angle;
               p1_x=-14*cos(angle)+28*sin(angle);
               p1_y=14*sin(angle)+28*cos(angle);
               p2_x=14*cos(angle)+28*sin(angle);
               p2_y=-14*sin(angle)+28*cos(angle);
               p3_x=14*cos(angle)-28*sin(angle);
               p3_y=-14*sin(angle)-28*cos(angle);
               p4_x=-14*cos(angle)-28*sin(angle);
               p4_y=14*sin(angle)-28*cos(angle);
            }
            //////////////// END visualize angle

            /////////////////// draw sensing data
            if (toMarking_LL>-8 && toMarking_RR>8 && -toMarking_ML+toMarking_MR<5.5) {     // right lane
 
               if (toMarking_M<1.5 && toMarking_R>6)
                   car_pos=int((174-(toMarking_ML+toMarking_MR)*6+198-toMarking_M*12)/2);
               else if (toMarking_M<1.5 && toMarking_R<=6)
                   car_pos=int((174-(toMarking_ML+toMarking_MR)*6+150-toMarking_M*12)/2);
               else
                   car_pos=int(174-(toMarking_ML+toMarking_MR)*6);

               if (visualize_angle==1) {
                  pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                  pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                  pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                  pt[3] = cvPoint(p4_x+car_pos,p4_y+600);  
                  int npts=4;
                  cvPolyLine(semanticRGB,&pt,&npts,1,1,cvScalar(0,255,0),2,CV_AA);  
               } else
                  cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,255,0),2);

               if (dist_LL<50)
                  cvRectangle(semanticRGB,cvPoint(126-14,600-dist_LL*12-28),cvPoint(126+14,600-dist_LL*12+28),cvScalar(237,99,157),2);
               if (dist_MM<50)
                  cvRectangle(semanticRGB,cvPoint(174-14,600-dist_MM*12-28),cvPoint(174+14,600-dist_MM*12+28),cvScalar(237,99,157),2);
            }

            else if (toMarking_RR<8 && toMarking_LL<-8 && -toMarking_ML+toMarking_MR<5.5) {   // left lane

               if (toMarking_M<1.5 && toMarking_L<-6)
                   car_pos=int((126-(toMarking_ML+toMarking_MR)*6+102-toMarking_M*12)/2);
               else if (toMarking_M<1.5 && toMarking_L>=-6)
                   car_pos=int((126-(toMarking_ML+toMarking_MR)*6+150-toMarking_M*12)/2);
               else
                   car_pos=int(126-(toMarking_ML+toMarking_MR)*6);

               if (visualize_angle==1) {
                  pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                  pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                  pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                  pt[3] = cvPoint(p4_x+car_pos,p4_y+600);  
                  int npts=4;
                  cvPolyLine(semanticRGB,&pt,&npts,1,1,cvScalar(0,255,0),2,CV_AA);  
               } else
                  cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,255,0),2);

               if (dist_MM<50)
                  cvRectangle(semanticRGB,cvPoint(126-14,600-dist_MM*12-28),cvPoint(126+14,600-dist_MM*12+28),cvScalar(237,99,157),2);
               if (dist_RR<50)
                  cvRectangle(semanticRGB,cvPoint(174-14,600-dist_RR*12-28),cvPoint(174+14,600-dist_RR*12+28),cvScalar(237,99,157),2);
            }

            else if (toMarking_M<2.5) {
                if (toMarking_L<-6) {   // left
                   car_pos=int(102-toMarking_M*12);
                   if (dist_R<50)
                      cvRectangle(semanticRGB,cvPoint(126-14,600-dist_R*12-28),cvPoint(126+14,600-dist_R*12+28),cvScalar(237,99,157),2);
                } else if (toMarking_R>6) {  // right
                   car_pos=int(198-toMarking_M*12);
                   if (dist_L<50)
                      cvRectangle(semanticRGB,cvPoint(174-14,600-dist_L*12-28),cvPoint(174+14,600-dist_L*12+28),cvScalar(237,99,157),2);
                } else if (toMarking_R<6 && toMarking_L>-6) {
                   car_pos=int(150-toMarking_M*12);
                   if (dist_L<50)
                      cvRectangle(semanticRGB,cvPoint(126-14,600-dist_L*12-28),cvPoint(126+14,600-dist_L*12+28),cvScalar(237,99,157),2);
                   if (dist_R<50)
                      cvRectangle(semanticRGB,cvPoint(174-14,600-dist_R*12-28),cvPoint(174+14,600-dist_R*12+28),cvScalar(237,99,157),2);
                }

                if (visualize_angle==1) {
                   pt[0] = cvPoint(p1_x+car_pos,p1_y+600);  
                   pt[1] = cvPoint(p2_x+car_pos,p2_y+600);
                   pt[2] = cvPoint(p3_x+car_pos,p3_y+600); 
                   pt[3] = cvPoint(p4_x+car_pos,p4_y+600);  
                   int npts=4;
                   cvPolyLine(semanticRGB,&pt,&npts,1,1,cvScalar(0,255,0),2,CV_AA);  
                } else
                   cvRectangle(semanticRGB,cvPoint(car_pos-14,600-28),cvPoint(car_pos+14,600+28),cvScalar(0,255,0),2);
            }
            /////////////////// END draw sensing data

            if ((shared->control==0) || (manual==1)) {
                for (int h = 0; h < semantic_height; h++) {
                   for (int w = 0; w < semantic_width; w++) {
                      semanticRGB->imageData[(h*semantic_width+w)*3+1]=0;
                      semanticRGB->imageData[(h*semantic_width+w)*3+0]=0;
                   }
                }
            }

            cvShowImage("Semantic Visualization",semanticRGB);
            ///////////////////////////// END semantic visualization

            shared->written=0;
        }  // end if (shared->written == 1)

        key=cvWaitKey( 1 );

        //////////////////////// Linux key
        if (key==1048603 || key==27) {  // Esc: exit
          shared->pause = 0;
          break;
        }
        else if (key==1048688 || key==112) shared->pause = 1-shared->pause;  // P: pause
        else if (key==1048675 || key==99) shared->control = 1-shared->control;  // C: autonomous driving on/off
        //////////////////////// END Linux key

        ////// override drive by keyboard
        else if (key==1113938 || key==65362) {
            shared->accelCmd = 1.0;
            shared->brakeCmd = 0;
            counter++;
        }
        else if (key==1113940 || key==65364) {
            shared->brakeCmd = 0.8;
            shared->accelCmd = 0;
            counter++;
        }
        else if (key==1113937 || key==65361) {
            shared->steerCmd = 0.5;
            manual=1;
            counter=0;
        }
        else if (key==1113939 || key==65363) {
            shared->steerCmd = -0.5;
            manual=1;
            counter=0;
        }
        else
            counter++;

        if (counter==20)
           manual=0;
        ////// END override drive by keyboard

    }  // end while (1) 


    ////////////////////// clean up opencv
    cvDestroyWindow("Image from leveldb");
    cvDestroyWindow("Semantic Visualization");
    cvDestroyWindow("Error Bar");
    cvReleaseImage( &screenRGB );
    cvReleaseImage( &resizeRGB );
    cvReleaseImage( &error_bar );
    cvReleaseImage( &semanticRGB );
    cvReleaseImage( &background );
    cvReleaseImage( &legend );
    ////////////////////// END clean up opencv

    ////////////////////// clean up leveldb
    delete batch;
    delete db;
    ////////////////////// END clean up leveldb

    ////////////////////// clean up memory sharing
    if(shmdt(shm) == -1)  
    {  
        fprintf(stderr, "shmdt failed\n");  
        exit(EXIT_FAILURE);  
    }  
 
    if(shmctl(shmid, IPC_RMID, 0) == -1)  
    {  
        fprintf(stderr, "shmctl(IPC_RMID) failed\n");  
        exit(EXIT_FAILURE);  
    }
    printf("\n********** Memory sharing stopped. Good Bye! **********\n");    
    exit(EXIT_SUCCESS); 
    ////////////////////// END clean up memory sharing 
}
