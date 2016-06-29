#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

//////////////// by chenyi
#include <pthread.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#define resize_width 280
#define resize_height 210
#define random(x) (rand()%x)
//////////////// by chenyi

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

///// RUN test single image
//////////////////////////////////////////////////////////////////////////////////// by chenyi
IplImage* leveldbRGB=cvCreateImage(cvSize(resize_width,resize_height),IPL_DEPTH_8U,3);

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();

  // datum scales
  const int size = this->datum_size_;
  const Dtype* mean = this->mean_;

  string value;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob

    snprintf(key_cstr, kMaxKeyLength, "%08d", 1);
    db_->Get(leveldb::ReadOptions(), string(key_cstr), &value);
    datum.ParseFromString(value);
    const string& data = datum.data();

    for (int j = 0; j < size; ++j) {
        Dtype datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        top_data[item_id * size + j] = (datum_element - mean[j]);
    }

    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }


    for (int h = 0; h < resize_height; ++h) {
       for (int w = 0; w < resize_width; ++w) {
          leveldbRGB->imageData[(h*resize_width+w)*3+0]=(uint8_t)data[h*resize_width+w];
          leveldbRGB->imageData[(h*resize_width+w)*3+1]=(uint8_t)data[resize_height*resize_width+h*resize_width+w];
          leveldbRGB->imageData[(h*resize_width+w)*3+2]=(uint8_t)data[resize_height*resize_width*2+h*resize_width+w];
        }
    }
    cvShowImage("Image from leveldb", leveldbRGB);

  }
}


extern leveldb::DB* db_tmp;

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  // Initialize DB
  db_.reset(db_tmp);

  // Read a data point, and use it to initialize the top blob.
  string value;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  snprintf(key_cstr, kMaxKeyLength, "%08d", 2);
  db_->Get(leveldb::ReadOptions(), string(key_cstr), &value);
  Datum datum;
  datum.ParseFromString(value);

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        1, 1, 1);
  }
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}
//////////////////////////////////////////////////////////////////////////////////// by chenyi
///// RUN single image

INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
