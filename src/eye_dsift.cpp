/*
 * EYE.cpp
 *
 *  Created on: Feb 10, 2014
 *      Author: jieshen
 */

#include "EYE/eye_dsift.hpp"

#include <vl/imopv.h>

#include <mkl.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::cerr;
using std::endl;

namespace EYE
{

  DSift::DSift()
      : dsift_model_(NULL), width_(0), height_(0), has_setup_(false)
  {
    init_with_default_parameter();
  }

  DSift::~DSift()
  {
    Clear();
  }

  void DSift::Clear()
  {
    init_with_default_parameter();
    clear_data();
    has_setup_ = false;
  }

  void DSift::init_with_default_parameter()
  {
    fast_ = DEFAULT_FAST;

    sizes_.resize(4, 0);
    for (uint32_t i = 0; i < 4; ++i)
      sizes_[i] = 2 * (i + 2);

    step_ = DEFAULT_STEP;
    float_desc_ = DEFAULT_FLT_DESC;
    magnif_ = DEFAULT_MAGNIF;
    win_size_ = DEFAULT_WIN_SIZE;
    contr_thrd_ = DEFAULT_CONTR_THRD;
  }

  void DSift::clear_data()
  {
    if (dsift_model_ != NULL)
    {
      vl_dsift_delete(dsift_model_);
      dsift_model_ = NULL;
      width_ = 0;
      height_ = 0;
    }
  }

  void DSift::SetUp(const uint32_t width, const uint32_t height)
  {
    if (dsift_model_ != NULL)
      vl_dsift_delete(dsift_model_);

    width_ = width;
    height_ = height;
    dsift_model_ = vl_dsift_new(width_, height_);

    vl_dsift_set_steps(dsift_model_, step_, step_);
    vl_dsift_set_window_size(dsift_model_, win_size_);
    vl_dsift_set_flat_window(dsift_model_, fast_);

    has_setup_ = true;
  }

  void DSift::Extract(const float* gray_img, const uint32_t width,
                      const uint32_t height, vector<VlDsiftKeypoint>* frames,
                      vector<float>* descrs, uint32_t* dim)
  {
    if (frames != NULL)
      frames->clear();

    if (descrs == NULL || dim == NULL)
    {
      cerr << "NULL pointer for descriptors and dim" << endl;
      exit(-1);
    }
    descrs->clear();
    *dim = 0;

    const uint32_t max_sz = *(std::max_element(sizes_.begin(), sizes_.end()));
    const uint32_t len_img = width * height;

    if (!has_setup_)
      SetUp(width, height);
    else
    {
      if(width != width_ || height != height_)
      {
        cerr << "ERROR: Image size not matching!" << endl;
        exit(-1);
      }
    }

    for (size_t i = 0; i < sizes_.size(); ++i)
    {
      const uint32_t sz = sizes_[i];

      const int off = std::floor(1 + 1.5 * (max_sz - sz));
      vl_dsift_set_bounds(dsift_model_, std::max(0, off - 1),
                          std::max(0, off - 1), width - 1, height - 1);

      VlDsiftDescriptorGeometry geom;
      geom.numBinX = DEFAULT_NUM_BIN_X;
      geom.numBinY = DEFAULT_NUM_BIN_Y;
      geom.numBinT = DEFAULT_NUM_BIN_T;
      geom.binSizeX = sz;
      geom.binSizeY = sz;
      vl_dsift_set_geometry(dsift_model_, &geom);

      /*
       {
       int stepX;
       int stepY;
       int minX;
       int minY;
       int maxX;
       int maxY;
       vl_bool useFlatWindow;

       int numFrames = vl_dsift_get_keypoint_num(dsift_model_);
       int descrSize = vl_dsift_get_descriptor_size(dsift_model_);
       VlDsiftDescriptorGeometry g = *vl_dsift_get_geometry(dsift_model_);

       vl_dsift_get_steps(dsift_model_, &stepY, &stepX);
       vl_dsift_get_bounds(dsift_model_, &minY, &minX, &maxY, &maxX);
       useFlatWindow = vl_dsift_get_flat_window(dsift_model_);

       printf(
       "vl_dsift: bounds:            [minX,minY,maxX,maxY] = [%d, %d, %d, %d]\n",
       minX + 1, minY + 1, maxX + 1, maxY + 1);
       printf("vl_dsift: subsampling steps: stepX=%d, stepY=%d\n", stepX,
       stepY);
       printf(
       "vl_dsift: num bins:          [numBinT, numBinX, numBinY] = [%d, %d, %d]\n",
       g.numBinT, g.numBinX, g.numBinY);
       printf("vl_dsift: descriptor size:   %d\n", descrSize);
       printf("vl_dsift: bin sizes:         [binSizeX, binSizeY] = [%d, %d]\n",
       g.binSizeX, g.binSizeY);
       printf("vl_dsift: flat window:       %s\n", VL_YESNO(useFlatWindow));
       printf("vl_dsift: window size:       %g\n",
       vl_dsift_get_window_size(dsift_model_));
       printf("vl_dsift: num of features:   %d\n", numFrames);
       }
       */

      const float sigma = 1.0 * sz / magnif_;
      float* smooth_img = (float*) malloc(sizeof(float) * len_img);
      memset(smooth_img, 0, sizeof(float) * len_img);
      vl_imsmooth_f(smooth_img, width, gray_img, width, height, width, sigma,
                    sigma);

      vl_dsift_process(dsift_model_, smooth_img);

      free(smooth_img);

      const int num_key_pts = vl_dsift_get_keypoint_num(dsift_model_);
      const VlDsiftKeypoint* key_points = vl_dsift_get_keypoints(dsift_model_);
      *dim = vl_dsift_get_descriptor_size(dsift_model_);
      const float* features = vl_dsift_get_descriptors(dsift_model_);

      float* f = (float*) malloc(sizeof(float) * num_key_pts * (*dim));
      cblas_scopy(num_key_pts * (*dim), features, 1, f, 1);
      cblas_sscal(num_key_pts * (*dim), 512.0f, f, 1);

      for (uint32_t d = 0; d < num_key_pts; ++d)
      {
        const float norm = (key_points + d)->norm;
        if (norm < contr_thrd_)
        {
          // remove low contrast
          memset(f + d * (*dim), 0, sizeof(float) * (*dim));
        }
        else
        {
          for (uint32_t j = d * (*dim); j < (d + 1) * (*dim); ++j)
          {
            float tmp = VL_MIN(f[j], 255.0);
            if (!float_desc_)
              tmp = (int) tmp;
            f[j] = tmp;
          }
        }
      }

      if (i == 0)
      {
        if (frames != NULL)
          frames->reserve(num_key_pts * sizes_.size());

        descrs->reserve(num_key_pts * sizes_.size() * (*dim));
      }

      if (frames != NULL)
        frames->insert(frames->end(), key_points, key_points + num_key_pts);

      descrs->insert(descrs->end(), f, f + num_key_pts * (*dim));

      free(f);
    }
  }
}

