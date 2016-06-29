#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  bool MULTIGPU = Caffe::gpu_mode() == Caffe::MASTER_SLAVE;
  // Initialize CUDA streams and cuDNN.
  stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
  }

  // Set the indexing parameters.
  weight_offset_ = (this->num_output_ / this->group_)
      * (this->channels_ / this->group_) * this->kernel_h_ * this->kernel_w_;
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      this->kernel_h_, this->kernel_w_);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensor4dDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensor4dDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }
  if(MULTIGPU){
    this->slave_weight_.reset(new Blob<Dtype>(
        this->num_output_, this->channels_ / this->group_,
        this->kernel_h_, this->kernel_w_));
    this->slave_bias_.reset(new Blob<Dtype>(1,1,1,this->num_output_));

    this->slave_bottom_.resize(bottom.size());
    this->slave_top_.resize((*top).size());
    for(int i = 0; i < bottom.size(); i++) {
      (this->slave_bottom_)[i].reset(new Blob<Dtype>());
    }
    for(int i = 0; i < (*top).size(); i++){
      (this->slave_top_)[i].reset(new Blob<Dtype>());
    }
    
    CUDA_CHECK(cudaStreamCreate(&data_stream_));
    Caffe::switch_to_slave_device();
    CUDA_CHECK(cudaStreamCreate(&slave_data_stream_));
    slave_stream_  = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
    slave_handle_  = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
    for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
      CUDA_CHECK(cudaStreamCreate(&slave_stream_[g]));
      CUDNN_CHECK(cudnnCreate(&slave_handle_[g]));
      CUDNN_CHECK(cudnnSetStream(slave_handle_[g], slave_stream_[g]));
    }
    Caffe::switch_to_master_device();
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  bottom_offset_ = (this->channels_ / this->group_)
      * this->height_ * this->width_;
  top_offset_ = (this->num_output_ / this->group_)
      * this->height_out_ * this->width_out_;
  bool MULTIGPU = Caffe::gpu_mode() == Caffe::MASTER_SLAVE;
  int split_num = this->num_;
  if(MULTIGPU) {
    split_num /= 2;
    CHECK_EQ(split_num * 2, this->num_)
       << "batch size needs to be even";
    this->slave_bottom_.resize(bottom.size());
    this->slave_top_.resize((*top).size());
    for(int i = 0; i < bottom.size(); i++) {
      (this->slave_bottom_)[i]->Reshape(split_num, this->channels_, this->height_, this->width_);
    }
    for(int i = 0; i < (*top).size(); i++){
      (this->slave_top_)[i]->Reshape(split_num, this->num_output_, this->height_out_, this->width_out_);
    }
  }
  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        split_num,
        this->channels_ / this->group_,
        this->height_, this->width_,
        this->channels_ * this->height_ * this->width_,
        this->height_ * this->width_,
        this->width_, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        split_num,
        this->num_output_ / this->group_,
        this->height_out_, this->width_out_,
        this->num_output_ * this->height_out_ * this->width_out_,
        this->height_out_ * this->width_out_,
        this->width_out_, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, this->pad_h_, this->pad_w_,
        this->stride_h_, this->stride_w_);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensor4dDescriptor(bottom_descs_[i]);
    cudnnDestroyTensor4dDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensor4dDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  delete [] stream_;
  delete [] handle_;
  // for slave
  bool MULTIGPU = Caffe::gpu_mode() == Caffe::MASTER_SLAVE;
  if (MULTIGPU) {
    Caffe::switch_to_slave_device();
  /*
  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensor4dDescriptor(slave_bottom_descs_[i]);
    cudnnDestroyTensor4dDescriptor(slave_top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(slave_conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensor4dDescriptor(slave_bias_desc_);
  }
  cudnnDestroyFilterDescriptor(slave_filter_desc_);
  */
    for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
      cudaStreamDestroy(slave_stream_[g]);
      cudnnDestroy(slave_handle_[g]);
    }

    delete [] slave_stream_;
    delete [] slave_handle_;
    Caffe::switch_to_master_device();
  }
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
