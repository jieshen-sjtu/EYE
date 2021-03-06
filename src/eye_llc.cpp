/*
 * eye_llc.cpp
 *
 *  Created on: Feb 11, 2014
 *      Author: jieshen
 */

#include "EYE/eye_llc.hpp"

#include <mkl.h>

#include <vl/kdtree.h>
#include <cstring>
#include <iostream>
using std::cerr;
using std::endl;

namespace EYE
{
  LLC::LLC()
      : kdforest_model_(NULL),
        has_setup_(false)
  {
    init_with_default_parameter();
    dim_ = 0;
    num_base_ = 0;
  }

  LLC::LLC(const shared_ptr<float>& base, const uint32_t dim,
           const uint32_t num_base)
      : kdforest_model_(NULL),
        has_setup_(false)
  {
    init_with_default_parameter();
    set_base(base, dim, num_base);
  }

  LLC::~LLC()
  {
    Clear();
  }

  void LLC::set_base(const shared_ptr<float>& base, const uint32_t dim,
                     const uint32_t num_base)
  {
    base_ = base;
    dim_ = dim;
    num_base_ = num_base;

    if (base_.get() == NULL || dim_ == 0 || num_base_ == 0)
    {
      cerr << "ERROR in set_base" << endl;
      exit(-1);
    }

    has_setup_ = false;
  }

  void LLC::Clear()
  {
    init_with_default_parameter();
    clear_data();
    has_setup_ = false;
  }

  void LLC::SetUp()
  {
    if (base_.get() == NULL || dim_ == 0 || num_base_ == 0)
    {
      cerr << "ERROR: must set the base before." << endl;
      exit(-1);
    }

    if (kdforest_model_ != NULL)
      vl_kdforest_delete(kdforest_model_);

    kdforest_model_ = vl_kdforest_new(VL_TYPE_FLOAT, dim_, num_tree_,
                                      dist_method_);

    vl_kdforest_set_thresholding_method(kdforest_model_, thrd_method_);
    vl_kdforest_set_max_num_comparisons(kdforest_model_, max_comp_);
    vl_kdforest_build(kdforest_model_, num_base_, base_.get());

    has_setup_ = true;
  }

  void LLC::Encode_with_max_pooling(const float* const data, const uint32_t dim,
                                    const uint32_t num_frame,
                                    float* const code) const
  {
    if (data == NULL || dim != dim_ || num_frame <= 0)
    {
      cerr << "ERROR in input data" << endl;
      exit(-1);
    }

    if (!has_setup_)
    {
      cerr << "ERROR: Must call SetUp() before." << endl;
      exit(-1);
    }

    vl_uint32* index = (vl_uint32*) vl_malloc(
        sizeof(vl_uint32) * num_knn_ * num_frame);
    memset(index, 0, num_knn_ * num_frame);
    float* dist(NULL);

    vl_kdforest_query_with_array(kdforest_model_, index, num_knn_, num_frame,
                                 dist, data);

    // start to encode
    const uint32_t len_code = num_base_;
    memset(code, 0, sizeof(float) * len_code);

    const uint32_t len_z = dim_ * num_knn_;
    const uint32_t len_C = num_knn_ * num_knn_;
    const uint32_t len_b = num_knn_;
    float* z = (float*) malloc(sizeof(float) * len_z);
    float* C = (float*) malloc(sizeof(float) * len_C);
    float* b = (float*) malloc(sizeof(float) * len_b);
    memset(z, 0, sizeof(float) * len_z);
    memset(C, 0, sizeof(float) * len_C);
    memset(b, 0, sizeof(float) * len_b);

    double sum(0);
    const float* base = base_.get();

    for (uint32_t i = 0; i < num_frame; i++)
    {

      uint32_t tmp_ind;

      // z = B_i - 1 * x_i'
      for (uint32_t n = 0; n < num_knn_; n++)
      {
        tmp_ind = (uint32_t) index[i * num_knn_ + n];
        memcpy(z + n * dim_, base + tmp_ind * dim_, sizeof(float) * dim_);

        cblas_saxpy(dim_, -1.0f, data + i * dim_, 1, z + n * dim_, 1);
      }

      // C = z * z', i.e. covariance matrix
      for (uint32_t m = 0; m < num_knn_; ++m)
        for (uint32_t n = m; n < num_knn_; ++n)
        {
          float sum = cblas_sdot(dim_, z + m * dim_, 1, z + n * dim_, 1);
          C[m * num_knn_ + n] = sum;
          C[n * num_knn_ + m] = sum;
        }

      sum = 0;
      for (uint32_t m = 0; m < num_knn_; m++)
        sum += C[m * num_knn_ + m];
      sum = sum * beta_;
      for (uint32_t m = 0; m < num_knn_; m++)
        C[m * num_knn_ + m] += sum;

      for (uint32_t m = 0; m < num_knn_; m++)
        b[m] = 1;

      // solve
      {
        char upper_triangle = 'U';
        int INFO;
        int int_one = 1;
        const int num_knn = (int) num_knn_;
        sposv(&upper_triangle, &num_knn, &int_one, C, &num_knn, b, &num_knn,
              &INFO);
      }

      sum = 0;

      for (uint32_t m = 0; m < num_knn_; m++)
        sum += b[m];
      cblas_sscal(num_knn_, 1.0 / sum, b, 1);

      for (uint32_t m = 0; m < num_knn_; m++)
      {
        tmp_ind = (uint32_t) index[i * num_knn_ + m];

        if (code[tmp_ind] < b[m])
          code[tmp_ind] = b[m];
      }
    }

    free(index);
    free(z);
    free(C);
    free(b);
  }

  void LLC::Encode_with_max_pooling(const float* const data, const uint32_t dim,
                                    const uint32_t num_frame,
                                    shared_ptr<float>* const codes) const
  {
    // start to encode
    const uint32_t len_code = num_base_;
    float* code = new float[len_code];
    Encode_with_max_pooling(data, dim, num_frame, code);

    codes->reset(code);
  }

  void LLC::Encode(const float* const data, const uint32_t dim,
                   const uint32_t num_frame,
                   float* const code) const
  {
    if (data == NULL || dim != dim_ || num_frame <= 0)
    {
      cerr << "ERROR in input data" << endl;
      exit(-1);
    }

    if (!has_setup_)
    {
      cerr << "ERROR: Must call SetUp() before." << endl;
      exit(-1);
    }

    vl_uint32* index = (vl_uint32*) vl_malloc(
        sizeof(vl_uint32) * num_knn_ * num_frame);
    memset(index, 0, num_knn_ * num_frame);
    float* dist(NULL);

    vl_kdforest_query_with_array(kdforest_model_, index, num_knn_, num_frame,
                                 dist, data);

    // start to encode
    const uint32_t len_code = num_base_ * num_frame;
    memset(code, 0, sizeof(float) * len_code);

    const uint32_t len_z = dim_ * num_knn_;
    const uint32_t len_C = num_knn_ * num_knn_;
    const uint32_t len_b = num_knn_;
    float* z = (float*) malloc(sizeof(float) * len_z);
    float* C = (float*) malloc(sizeof(float) * len_C);
    float* b = (float*) malloc(sizeof(float) * len_b);
    memset(z, 0, sizeof(float) * len_z);
    memset(C, 0, sizeof(float) * len_C);
    memset(b, 0, sizeof(float) * len_b);

    double sum(0);
    const float* base = base_.get();

    for (uint32_t i = 0; i < num_frame; i++)
    {

      uint32_t tmp_ind;

      // z = B_i - 1 * x_i'
      for (uint32_t n = 0; n < num_knn_; n++)
      {
        tmp_ind = (uint32_t) index[i * num_knn_ + n];
        memcpy(z + n * dim_, base + tmp_ind * dim_, sizeof(float) * dim_);

        cblas_saxpy(dim_, -1.0f, data + i * dim_, 1, z + n * dim_, 1);
      }

      // C = z * z', i.e. covariance matrix
      for (uint32_t m = 0; m < num_knn_; ++m)
        for (uint32_t n = m; n < num_knn_; ++n)
        {
          float sum = cblas_sdot(dim_, z + m * dim_, 1, z + n * dim_, 1);
          C[m * num_knn_ + n] = sum;
          C[n * num_knn_ + m] = sum;
        }

      sum = 0;
      for (uint32_t m = 0; m < num_knn_; m++)
        sum += C[m * num_knn_ + m];
      sum = sum * beta_;
      for (uint32_t m = 0; m < num_knn_; m++)
        C[m * num_knn_ + m] += sum;

      for (uint32_t m = 0; m < num_knn_; m++)
        b[m] = 1;

      // solve
      {
        char upper_triangle = 'U';
        int INFO;
        int int_one = 1;
        const int num_knn = (int) num_knn_;
        sposv(&upper_triangle, &num_knn, &int_one, C, &num_knn, b, &num_knn,
              &INFO);
      }

      sum = 0;

      for (uint32_t m = 0; m < num_knn_; m++)
        sum += b[m];
      cblas_sscal(num_knn_, 1.0 / sum, b, 1);

      for (uint32_t m = 0; m < num_knn_; m++)
      {
        tmp_ind = (uint32_t) index[i * num_knn_ + m];

        code[i * num_base_ + tmp_ind] = b[m];
      }
    }

    /*
     float* max_code = (float*) malloc(sizeof(float) * num_base_);
     memset(max_code, 0, sizeof(float) * num_base_);
     for (uint32_t m = 0; m < num_base_; ++m)
     for (uint32_t n = 0; n < num_frame; ++n)
     if (max_code[m] < code[n * num_base_ + m])
     max_code[m] = code[n * num_base_ + m];

     codes.reset(max_code);
     free(code);
     */

    free(index);
    free(z);
    free(C);
    free(b);
  }

  void LLC::Encode(const float* const data, const uint32_t dim,
                   const uint32_t num_frame,
                   shared_ptr<float>* const codes) const
  {
    // start to encode
    const uint32_t len_code = num_base_ * num_frame;
    float* code = new float[len_code];

    Encode(data, dim, num_frame, code);
    codes->reset(code);
  }

  void LLC::init_with_default_parameter()
  {
    thrd_method_ = DEFAULT_THRD_METHOD;
    dist_method_ = DEFAULT_DIST_METHOD;
    num_tree_ = DEFAULT_NUM_TREE;
    num_knn_ = DEFAULT_NUM_KNN;
    max_comp_ = DEFAULT_MAX_COMP;
    beta_ = DEFAULT_BETA;
  }

  void LLC::clear_data()
  {
    if (kdforest_model_ != NULL)
    {
      vl_kdforest_delete(kdforest_model_);
      kdforest_model_ = NULL;
    }

    base_.reset();
    dim_ = 0;
    num_base_ = 0;
  }

}

