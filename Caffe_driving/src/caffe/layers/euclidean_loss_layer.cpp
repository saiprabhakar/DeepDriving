#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

//////// angle, toMarking, dist, fast, etc., (14 output)
///////////////////////////////// by chenyi
template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  int count = bottom[0]->count();
  int num = bottom[0]->num();   // batch size
  int dim = count/num;   // equals to number of outputs in last layer
  Dtype y_array[count];
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  for (int i = 0; i < num; ++i) {

    y_array[i * dim] = label[i*14+0]/1.1+0.5;     // angle range ~ [-0.5, 0.5]
    if (y_array[i * dim]>1.0) y_array[i * dim]=1.0;
    if (y_array[i * dim]<0.0) y_array[i * dim]=0.0;

    y_array[i * dim + 1] = label[i*14+1]*0.17778+1.34445;   // toMarking_L range ~ [-7, -2.5]

    y_array[i * dim + 2] = label[i*14+2]*0.14545+0.39091;   // toMarking_M range ~ [-2, 3.5]

    y_array[i * dim + 3] = label[i*14+3]*0.17778-0.34445;   // toMarking_R range ~ [2.5, 7]

    y_array[i * dim + 4] = label[i*14+4]/95.0+0.12;   // dist_L range ~ [0, 75]

    y_array[i * dim + 5] = label[i*14+5]/95.0+0.12;   // dist_R range ~ [0, 75]

    y_array[i * dim + 6] = label[i*14+6]*0.14545+1.48181;   // toMarking_LL range ~ [-9.5, -4]

    y_array[i * dim + 7] = label[i*14+7]*0.16+0.98;   // toMarking_ML range ~ [-5.5, -0.5]

    y_array[i * dim + 8] = label[i*14+8]*0.16+0.02;   // toMarking_MR range ~ [0.5, 5.5]

    y_array[i * dim + 9] = label[i*14+9]*0.14545-0.48181;   // toMarking_RR range ~ [4, 9.5]

    y_array[i * dim + 10] = label[i*14+10]/95.0+0.12;   // dist_LL range ~ [0, 75]

    y_array[i * dim + 11] = label[i*14+11]/95.0+0.12;   // dist_MM range ~ [0, 75]

    y_array[i * dim + 12] = label[i*14+12]/95.0+0.12;   // dist_RR range ~ [0, 75]

    y_array[i * dim + 13] = label[i*14+13]*0.6+0.2;   // fast range ~ {0, 1}

  }

  caffe_sub(count, bottom_data, y_array, diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / num / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;


  //for (int i = 0; i < num; ++i) {
      int i=25;
      for (int j = 0; j < dim; ++j) {
          printf("num: %d, dim: %d, out: %f, y_array: %f, diff: %f \n", i, j, bottom_data[i*dim+j], y_array[i*dim+j], diff_.cpu_data()[i*dim+j]); 
          fflush(stdout);
      }
  //}

}
///////////////////////////////// by chenyi

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / (*bottom)[i]->num();
      caffe_cpu_axpby(
          (*bottom)[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          (*bottom)[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);

}  // namespace caffe
