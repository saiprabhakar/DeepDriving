////////////////////////////////////////////////
//
//  Manaully drive a car in TORCS to collect training data for caffe, for all the lane configurations.
//  Used with TORCS driver agent "human" and corresponding modified tracks. The lane configuration can be 
//  one-lane, two-lane, or three-lane, but the "human" agent, the pavement texture, and the traffic cars (chenyi_AIx)
//  must be set consistently. The saved database is named "TORCS_Training_1F".
//
//  Input keys 
//  Esc: exit
//  P: pause (initially paused)
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
#include <math.h>

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#define image_width 640
#define image_height 480
#define resize_width 280
#define resize_height 210

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

struct shared_use_st  
{  
    int written;
    uint8_t data[image_width*image_height*3];  
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
    IplImage* leveldbRGB=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);
    cvNamedWindow("Image from leveldb",1);
    cvNamedWindow("Image from TORCS",1);
    int key;
    ////////////////////// END set up opencv

    ////////////////////// set up leveldb  
    leveldb::Options options;
    options.error_if_exists = false;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    options.max_open_files = 100;

    leveldb::DB* db;
    LOG(INFO) << "Opening leveldb: TORCS_Training_1F";
    leveldb::Status status = leveldb::DB::Open(options, "TORCS_Training_1F", &db);
    CHECK(status.ok()) << "Failed to open leveldb: TORCS_Training_1F";

    Datum datum;
    string value;
    
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];

    leveldb::WriteBatch* batch = new leveldb::WriteBatch();
    ////////////////////// END set up leveldb

    int frame = 0;
    int frame_offset = 0;  // used to add more images to existing database 

    while (1) {

       if (shared->written == 1) {  // the new image data is ready to be read

         frame++;

         for (int h = 0; h < image_height; h++) {
            for (int w = 0; w < image_width; w++) {
               screenRGB->imageData[(h*image_width+w)*3+2]=shared->data[((image_height-h-1)*image_width+w)*3+0];
               screenRGB->imageData[(h*image_width+w)*3+1]=shared->data[((image_height-h-1)*image_width+w)*3+1];
               screenRGB->imageData[(h*image_width+w)*3+0]=shared->data[((image_height-h-1)*image_width+w)*3+2];
            }
         }
        
         cvResize(screenRGB,resizeRGB);
 
         cvShowImage("Image from TORCS", screenRGB);

         ///////////////////////////// write leveldb     
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

         datum.add_float_data(shared->angle);
         datum.add_float_data(shared->toMarking_L);
         datum.add_float_data(shared->toMarking_M);
         datum.add_float_data(shared->toMarking_R);
         datum.add_float_data(shared->dist_L);
         datum.add_float_data(shared->dist_R);
         datum.add_float_data(shared->toMarking_LL);
         datum.add_float_data(shared->toMarking_ML);
         datum.add_float_data(shared->toMarking_MR);
         datum.add_float_data(shared->toMarking_RR);
         datum.add_float_data(shared->dist_LL);
         datum.add_float_data(shared->dist_MM);
         datum.add_float_data(shared->dist_RR);
         datum.add_float_data(shared->fast);

         printf("%f,%f,%f,%f,%f,%f,%f\n", shared->toMarking_LL, shared->toMarking_ML, shared->toMarking_MR, shared->toMarking_RR, shared->dist_LL, shared->dist_MM, shared->dist_RR);
         printf("%f,%f,%f,%f,%f,%f,%f\n\n", shared->toMarking_L, shared->toMarking_M, shared->toMarking_R, shared->dist_L, shared->dist_R, shared->angle, shared->fast);
         fflush(stdout);

         // sequential
         snprintf(key_cstr, kMaxKeyLength, "%08d", frame_offset+frame);
                         
         // get the value
         datum.SerializeToString(&value);
         batch->Put(string(key_cstr), value);
         if (frame % 100 == 0) {
           db->Write(leveldb::WriteOptions(), batch);
           LOG(ERROR) << "Processed " << frame << " files.";
           delete batch;
           batch = new leveldb::WriteBatch();
         }
         ///////////////////////////// END write leveldb

         ///////////////////////////// read leveldb and show
         if (frame>100) {                      
            snprintf(key_cstr, kMaxKeyLength, "%08d", frame_offset+frame-100);
            db->Get(leveldb::ReadOptions(), string(key_cstr), &value);
            datum.ParseFromString(value);
            const string& data = datum.data();
       
            for (int h = 0; h < resize_height; ++h) {
                for (int w = 0; w < resize_width; ++w) {
                    leveldbRGB->imageData[(h*resize_width+w)*3+0]=(uint8_t)data[h*resize_width+w];
                    leveldbRGB->imageData[(h*resize_width+w)*3+1]=(uint8_t)data[resize_height*resize_width+h*resize_width+w];
                    leveldbRGB->imageData[(h*resize_width+w)*3+2]=(uint8_t)data[resize_height*resize_width*2+h*resize_width+w];
                }
            }
            cvShowImage("Image from leveldb", leveldbRGB);

         } // end if (frame>100)
         ///////////////////////////// END read leveldb and show

         shared->written=0;
        }  // if (shared->written == 1)

        key=cvWaitKey( 5 );

        //////////////////////// Linux key
        if (key==1048603 || key==27) {  // Esc: exit
          shared->pause = 0;
          break;  
        }
        else if (key==1048688 || key==112) shared->pause = 1-shared->pause;  // P: pause
        //////////////////////// END Linux key

    }  // end while (1) 

    db->Write(leveldb::WriteOptions(), batch);

    ////////////////////// clean up opencv
    cvDestroyWindow("Image from TORCS");
    cvDestroyWindow("Image from leveldb");
    cvReleaseImage( &screenRGB );
    cvReleaseImage( &resizeRGB );
    cvReleaseImage( &leveldbRGB );
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
