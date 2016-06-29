#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

//////// angle, toMarking, dist, fast, etc., (14 output)
///////////////////////////////// by chenyi
template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  int count = bottom[0]->count();
  int num = bottom[0]->num();   // batch size
  int dim = count/num;   // equals to number of outputs in last layer

  Dtype y_array[count];
  Dtype label[num*14];
  Dtype bottom_data[count]; 
  Dtype diff[count]; 

  Dtype* y_array_cuda;
  cudaMalloc((void**)&y_array_cuda,sizeof(Dtype)*count);

  const Dtype* bottom_data_cuda = bottom[0]->gpu_data();
  const Dtype* label_cuda = bottom[1]->gpu_data();

  cudaMemcpy(bottom_data,bottom_data_cuda,sizeof(Dtype)*count,cudaMemcpyDeviceToHost);
  cudaMemcpy(label,label_cuda,sizeof(Dtype)*num*14,cudaMemcpyDeviceToHost);


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

  cudaMemcpy(y_array_cuda,y_array,sizeof(Dtype)*count,cudaMemcpyHostToDevice);

  caffe_gpu_sub(count, bottom_data_cuda, y_array_cuda, diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / num / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;

  cudaMemcpy(diff,diff_.gpu_data(),sizeof(Dtype)*count,cudaMemcpyDeviceToHost);
  cudaFree(y_array_cuda);

  //for (int i = 0; i < num; ++i) {
      int i=25;
      for (int j = 0; j < dim; ++j) {
          printf("num: %d, dim: %d, out: %f, y_array: %f, diff: %f \n", i, j, bottom_data[i*dim+j], y_array[i*dim+j], diff[i*dim+j]); 
          fflush(stdout);
      }
  //}

}
///////////////////////////////// by chenyi

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / (*bottom)[i]->num();
      caffe_gpu_axpby(
          (*bottom)[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          (*bottom)[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_CLASS(EuclideanLossLayer);

}  // namespace caffe
