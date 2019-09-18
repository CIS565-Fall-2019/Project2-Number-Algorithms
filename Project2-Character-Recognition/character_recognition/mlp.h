#pragma once

#include "common.h"
#include <cublas_v2.h>

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here
    
    class MLP3 {
    public:
      // constructor
      MLP3(int n_input, int n_hidden, int n_output);

      // destructor
      ~MLP3();

      void init_weights(float *wkj, float *wji);
      void init_weights();
      void predict(float *x_input, float *y_pred);
      void train(float *x_train, float *y_train, int n_data, int n_epoch);

    private:
      void forward();
      void calculate_loss();
      void back_propagation();
      void update_weights();

      int input_size_;
      int hidden_size_;
      int output_size_;

      // weight & gradients
      float *wkj_, *wji_;
      float *gwkj_, *gwji_;

      // layers
      int n_data_;
      float* dev_input_;
      float* dev_hidden_;
      float* dev_hidden_sigmoid_;
      float* dev_output_;
      float* dev_output_sigmoid_;
      float* dev_target_;

      float total_error_;

      cublasHandle_t cublas_handle_;
    };
}
