#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

#define CUDNN_STREAMS_PER_GROUP 3

__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  bool MULTIGPU = Caffe::gpu_mode() == Caffe::MASTER_SLAVE;
  Dtype* slave_weight_data = NULL;
  Dtype* slave_bias_data = NULL;
  Dtype* slave_bottom_data = NULL;
  Dtype* slave_top_data = NULL;

  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bias_data = NULL;
  if (this->bias_term_) {
    bias_data = this->blobs_[1]->gpu_data();
  }
  if (MULTIGPU) {
    CUDA_CHECK(cudaDeviceSynchronize());
    Caffe::switch_to_slave_device();
    CUDA_CHECK(cudaDeviceSynchronize());
    slave_weight_data = this->slave_weight_->mutable_gpu_data();
    Caffe::switch_to_master_device();
    CUDA_CHECK(cudaMemcpyPeerAsync(slave_weight_data, 
        Caffe::slave_device_id(), weight, Caffe::master_device_id(),
        this->blobs_[0]->count() * sizeof(Dtype), data_stream_));
    if (this->bias_term_) {
      Caffe::switch_to_slave_device();
      slave_bias_data = this->slave_bias_->mutable_gpu_data();
      Caffe::switch_to_master_device();
      CUDA_CHECK(cudaMemcpyPeerAsync(slave_bias_data, 
          Caffe::slave_device_id(), bias_data, Caffe::master_device_id(),
          this->blobs_[1]->count() * sizeof(Dtype), data_stream_)); 
    }
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
    if (MULTIGPU) {
      Caffe::switch_to_slave_device();
      slave_bottom_data = (this->slave_bottom_)[i]->mutable_gpu_data();
      slave_top_data = (this->slave_top_)[i]->mutable_gpu_data();
      Caffe::switch_to_master_device();
      CUDA_CHECK(cudaMemcpyPeerAsync(slave_bottom_data, 
        Caffe::slave_device_id(), bottom_data + bottom[i]->count() / 2,
        Caffe::master_device_id(), 
        bottom[i]->count() * sizeof(Dtype) / 2, data_stream_));
    }
    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
          bottom_descs_[i], bottom_data + bottom_offset_ * g,
          filter_desc_, weight + weight_offset_ * g,
          conv_descs_[i],
          top_descs_[i], top_data + top_offset_ * g,
          CUDNN_RESULT_NO_ACCUMULATE));
      // slave
      if (MULTIGPU) {
        CUDA_CHECK(cudaStreamSynchronize(data_stream_));
        Caffe::switch_to_slave_device();
        CUDNN_CHECK(cudnnConvolutionForward(slave_handle_[g],
            bottom_descs_[i], slave_bottom_data + bottom_offset_ * g,
            filter_desc_, slave_weight_data + weight_offset_ * g,
            conv_descs_[i],
            top_descs_[i], slave_top_data + top_offset_ * g,
            CUDNN_RESULT_NO_ACCUMULATE));
        Caffe::switch_to_master_device();
      }
      // Bias.
      if (this->bias_term_) {
        Dtype alpha = 1.;
        CUDNN_CHECK(cudnnAddTensor4d(handle_[g], CUDNN_ADD_SAME_C, &alpha,
            bias_desc_, bias_data + bias_offset_ * g,
            top_descs_[i], top_data + top_offset_ * g));
        if(MULTIGPU) {
          Caffe::switch_to_slave_device();
          CUDNN_CHECK(cudnnAddTensor4d(slave_handle_[g], CUDNN_ADD_SAME_C, &alpha,
              bias_desc_, slave_bias_data + bias_offset_ * g,
              top_descs_[i], slave_top_data + top_offset_ * g));
          Caffe::switch_to_master_device();
        }
      }
    }
    // multi-gpu copy back
    if(MULTIGPU) {
      Caffe::switch_to_slave_device();
      CUDA_CHECK(cudaStreamSynchronize(slave_stream_[this->group_ - 1]));
      CUDA_CHECK(cudaMemcpyPeerAsync(top_data + (*top)[i]->count() / 2, 
          Caffe::master_device_id(), slave_top_data, Caffe::slave_device_id(),
          (*top)[i]->count() * sizeof(Dtype) / 2, slave_data_stream_));
      Caffe::switch_to_master_device();
      CUDA_CHECK(cudaStreamSynchronize(slave_data_stream_));
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  bool MULTIGPU = Caffe::gpu_mode() == Caffe::MASTER_SLAVE;
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  const Dtype* bias = NULL;
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias = this->blobs_[1]->gpu_data();
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  Dtype* slave_weight_data = NULL;
  Dtype* slave_weight_diff = NULL;
  Dtype* slave_bias_data = NULL;
  Dtype* slave_bias_diff = NULL;
  if(MULTIGPU) {
    CUDA_CHECK(cudaDeviceSynchronize());
    if (this->param_propagate_down_[0]) {
      Caffe::switch_to_slave_device();
      slave_weight_data = this->slave_weight_->mutable_gpu_data();
      slave_weight_diff = this->slave_weight_->mutable_gpu_diff();
      caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), slave_weight_diff);
      Caffe::switch_to_master_device();
      // copy weight to the slave
      CUDA_CHECK(cudaMemcpyPeerAsync(slave_weight_data, 
          Caffe::slave_device_id(), weight, Caffe::master_device_id(),
          this->blobs_[0]->count() * sizeof(Dtype), data_stream_));
    }
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Caffe::switch_to_slave_device();
      slave_bias_data = this->slave_bias_->mutable_gpu_data(); 
      slave_bias_diff = this->slave_bias_->mutable_gpu_diff();
      caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), slave_bias_diff);
      Caffe::switch_to_master_device();
      // copy bias to the slave
      CUDA_CHECK(cudaMemcpyPeerAsync(slave_bias_data, 
          Caffe::slave_device_id(), bias, Caffe::master_device_id(),
          this->blobs_[1]->count() * sizeof(Dtype), data_stream_));
    }
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = NULL;
    if(this->param_propagate_down_[0]) {
      bottom_data = (*bottom)[i]->gpu_data(); 
    }
    Dtype* slave_top_diff = NULL;
    Dtype* slave_bottom_data = NULL;
    if (MULTIGPU) {
      CUDA_CHECK(cudaDeviceSynchronize());
      Caffe::switch_to_slave_device();
      slave_top_diff = (this->slave_top_)[i]->mutable_gpu_diff();
      Caffe::switch_to_master_device();
      // copy top diff and bottom act
      CUDA_CHECK(cudaMemcpyPeerAsync(slave_top_diff,
          Caffe::slave_device_id(), top_diff + top[i]->count() / 2, Caffe::master_device_id(),
          top[i]->count() * sizeof(Dtype) / 2, data_stream_));
          slave_bottom_data = (this->slave_bottom_)[i]->mutable_gpu_data();
      if(this->param_propagate_down_[0]) {
        slave_bottom_data = (this->slave_bottom_)[i]->mutable_gpu_data();
        CUDA_CHECK(cudaMemcpyPeerAsync(slave_bottom_data, 
            Caffe::slave_device_id(), bottom_data + (*bottom)[i]->count() / 2, Caffe::master_device_id(),
            (*bottom)[i]->count() * sizeof(Dtype) / 2, data_stream_));
      }
    }
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
            top_descs_[i],  top_diff + top_offset_ * g,
            bias_desc_, bias_diff + bias_offset_ * g,
            CUDNN_RESULT_ACCUMULATE));
        if (MULTIGPU) {
          CUDA_CHECK(cudaStreamSynchronize(data_stream_));
          Caffe::switch_to_slave_device();
          CUDNN_CHECK(cudnnConvolutionBackwardBias(slave_handle_[0*this->group_ + g],
            top_descs_[i],  slave_top_diff + top_offset_ * g,
            bias_desc_, slave_bias_diff + bias_offset_ * g,
            CUDNN_RESULT_ACCUMULATE));
          Caffe::switch_to_master_device();
        }
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle_[1*this->group_ + g],
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            top_descs_[i],    top_diff + top_offset_ * g,
            conv_descs_[i],
            filter_desc_, weight_diff + weight_offset_ * g,
            CUDNN_RESULT_ACCUMULATE));
        if (MULTIGPU) {
          CUDA_CHECK(cudaStreamSynchronize(data_stream_));
          Caffe::switch_to_slave_device();
          CUDNN_CHECK(cudnnConvolutionBackwardFilter(slave_handle_[1*this->group_ + g],
            bottom_descs_[i], slave_bottom_data + bottom_offset_ * g,
            top_descs_[i],    slave_top_diff + top_offset_ * g,
            conv_descs_[i],
            filter_desc_, slave_weight_diff + weight_offset_ * g,
            CUDNN_RESULT_ACCUMULATE));
          Caffe::switch_to_master_device();
        }
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
        Dtype* slave_bottom_diff = NULL;
        if (MULTIGPU) {
          Caffe::switch_to_slave_device();
          slave_bottom_diff = (this->slave_bottom_)[i]->mutable_gpu_diff();
          Caffe::switch_to_master_device();
        }
        CUDNN_CHECK(cudnnConvolutionBackwardData(handle_[2*this->group_ + g],
            filter_desc_, weight + weight_offset_ * g,
            top_descs_[i],    top_diff + top_offset_ * g,
            conv_descs_[i],
            bottom_descs_[i], bottom_diff + bottom_offset_ * g,
            CUDNN_RESULT_NO_ACCUMULATE));
        if (MULTIGPU) {
          CUDA_CHECK(cudaStreamSynchronize(data_stream_));
          Caffe::switch_to_slave_device();
          CUDNN_CHECK(cudnnConvolutionBackwardData(slave_handle_[2*this->group_ + g],
            filter_desc_, slave_weight_data + weight_offset_ * g,
            top_descs_[i],    slave_top_diff + top_offset_ * g,
            conv_descs_[i],
            bottom_descs_[i], slave_bottom_diff + bottom_offset_ * g,
            CUDNN_RESULT_NO_ACCUMULATE));
          Caffe::switch_to_master_device();
        }
      }
    }
    // copy back the bottom diff
    if (propagate_down[i] && MULTIGPU) {
      CUDA_CHECK(cudaStreamSynchronize(slave_stream_[this->group_ * 3 - 1]));
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
      Caffe::switch_to_slave_device();
      const Dtype* slave_bottom_diff = (this->slave_bottom_)[i]->gpu_diff();
      CUDA_CHECK(cudaMemcpyPeerAsync(bottom_diff + (*bottom)[i]->count() / 2, 
          Caffe::master_device_id(), slave_bottom_diff, Caffe::slave_device_id(),
          (*bottom)[i]->count() * sizeof(Dtype) / 2, slave_data_stream_)); 
      Caffe::switch_to_master_device();
    }

    // Add the gradient together
    if (MULTIGPU) {
      CUDA_CHECK(cudaStreamSynchronize(slave_stream_[this->group_ * 2 - 1]));
      CUDA_CHECK(cudaStreamSynchronize(slave_stream_[this->group_ * 1 - 1]));
      CUDA_CHECK(cudaStreamSynchronize(stream_[this->group_ * 2 - 1]));
      CUDA_CHECK(cudaStreamSynchronize(stream_[this->group_ * 1 - 1]));
      caffe_gpu_axpy<Dtype>(this->blobs_[0]->count(), (Dtype)1.0, slave_weight_diff, weight_diff);
      caffe_gpu_axpy<Dtype>(this->blobs_[1]->count(), (Dtype)1.0, slave_bias_diff, bias_diff);
      CUDA_CHECK(cudaStreamSynchronize(slave_data_stream_));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
