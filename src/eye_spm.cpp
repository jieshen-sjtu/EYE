/*
 * eye_spm.cpp
 *
 *  Created on: Mar 16, 2014
 *      Author: jieshen
 */

#include "EYE/eye_spm.hpp"

#include <mkl.h>

#include <iostream>
#include <cstring>
#include <cmath>

using std::cerr;
using std::endl;

namespace EYE
{
  SPM::SPM()
      : img_width_(0),
        img_height_(0),
        finest_num_blk_(0),
        total_num_blk_(0),
        has_setup_(false)
  {
    init_with_default_parameter();
  }

  void SPM::init_with_default_parameter()
  {
    num_spm_level_ = DEFAULT_SPM_LEVEL;
  }

  void SPM::SetUp(const uint32_t width, const uint32_t height)
  {
    img_width_ = width;
    img_height_ = height;

    total_num_blk_ = 0;
    level_num_blk_.resize(num_spm_level_, 0);
    for (uint32_t i = 0; i < num_spm_level_; ++i)
    {
      level_num_blk_[i] = std::pow(2, i);
      total_num_blk_ += level_num_blk_[i] * level_num_blk_[i];
      //cerr << "blk = " << level_num_blk_[i] << endl;
    }

    finest_num_blk_ = level_num_blk_[num_spm_level_ - 1];

    blk_start_x_.resize(num_spm_level_);
    blk_end_x_.resize(num_spm_level_);
    blk_start_y_.resize(num_spm_level_);
    blk_end_y_.resize(num_spm_level_);
    for (uint32_t i = 0; i < num_spm_level_; ++i)
    {
      const uint32_t num_blk = level_num_blk_[i];
      const float blk_width = img_width_ * 1.0 / num_blk;
      const float blk_height = img_height_ * 1.0 / num_blk;

      vector<float>& start_x = blk_start_x_[i];
      vector<float>& end_x = blk_end_x_[i];
      vector<float>& start_y = blk_start_y_[i];
      vector<float>& end_y = blk_end_y_[i];

      start_x.resize(num_blk, 0);
      end_x.resize(num_blk, 0);
      start_y.resize(num_blk, 0);
      end_y.resize(num_blk, 0);

      for (uint32_t bidx = 0; bidx < num_blk; ++bidx)
      {
        if (bidx == 0)
        {
          start_x[bidx] = 0;
          start_y[bidx] = 0;
        }
        else
        {
          start_x[bidx] = end_x[bidx - 1];
          start_y[bidx] = end_y[bidx - 1];
        }

        end_x[bidx] = start_x[bidx] + blk_width;
        end_y[bidx] = start_y[bidx] + blk_height;
      }
    }  // level

    level_start_idx_.resize(num_spm_level_, 0);
    for (int i = num_spm_level_ - 2; i >= 0; --i)
    {
      const int prev_lv = i + 1;
      level_start_idx_[i] = level_start_idx_[prev_lv]
          + level_num_blk_[prev_lv] * level_num_blk_[prev_lv];

      //cerr << "lv = " << level_start_idx_[i] << endl;
    }

    has_setup_ = true;
  }

  uint32_t SPM::get_block_start_idx(const uint32_t level, const uint32_t yidx,
                                    const uint32_t xidx)
  {
    return (level_start_idx_[level] + yidx * level_num_blk_[level] + xidx);
  }

  void SPM::build_cell_blk_map(const float* const pos, const uint32_t num_data)
  {
    map_blk_cell_.clear();
    map_cell_blk_.clear();

    for (uint32_t i = 0; i < num_data; ++i)
    {
      const float x = pos[2 * i];
      const float y = pos[2 * i + 1];

      //std::cout << x << " " << y << endl;

      vector<uint32_t> blk_idx(num_spm_level_, 0);
      for (uint32_t lv = 0; lv < num_spm_level_; ++lv)
      {
        const uint32_t num_blk = level_num_blk_[lv];
        const float blk_width = img_width_ * 1.0 / num_blk;
        const float blk_height = img_height_ * 1.0 / num_blk;

        const uint32_t xidx = x / blk_width;
        const uint32_t yidx = y / blk_height;
        /*
         cerr << "lv = " << lv << "xidx = " << xidx << " yidx = " << yidx
         << endl;*/

        blk_idx[lv] = get_block_start_idx(lv, yidx, xidx);

        map<uint32_t, vector<uint32_t> >::iterator it = map_blk_cell_.find(
            blk_idx[lv]);
        if (it == map_blk_cell_.end())
        {
          vector<uint32_t> cells;
          cells.reserve(512);
          cells.push_back(i);
          map_blk_cell_.insert(std::make_pair(blk_idx[lv], cells));
        }
        else
        {
          it->second.push_back(i);
        }
      }

      map_cell_blk_.insert(std::make_pair(i, blk_idx));
    }
  }

  void SPM::MaxPooling(const float* const data, const uint32_t feat_dim,
                       const uint32_t num_data, const float* const pos,
                       float* const spm_code)
  {
    if (!has_setup_)
    {
      cerr << "Call SetUp() first" << endl;
      exit(-1);
    }
    if (data == NULL
        || feat_dim == 0|| num_data ==0 || pos == NULL || spm_code==NULL)
    {
      cerr << "ERROR: Input for MaxPooling" << endl;
      exit(-1);
    }

    const uint32_t spm_code_len = total_num_blk_ * feat_dim;
    memset(spm_code, 0, sizeof(float) * spm_code_len);

    //cerr << "start build" << endl;
    build_cell_blk_map(pos, num_data);

    //cerr << "build done" << endl;

    // compute the finest bin first
    {
      const int lv = num_spm_level_ - 1;
      const uint32_t num_blk = level_num_blk_[lv];

      for (int ybin = 0; ybin < num_blk; ++ybin)
        for (int xbin = 0; xbin < num_blk; ++xbin)
        {
          const int blk_id = get_block_start_idx(lv, ybin, xbin);
          const vector<uint32_t>& cells = map_blk_cell_.find(blk_id)->second;

          float* const out = spm_code + blk_id * feat_dim;
          cblas_scopy(feat_dim, data + cells[0] * feat_dim, 1, out, 1);

          for (size_t i = 1; i < cells.size(); ++i)
          {
            const float* const in = data + cells[i] * feat_dim;
            for (size_t dd = 0; dd < feat_dim; ++dd)
              out[dd] = std::max(out[dd], in[dd]);
          }
        }
    }

    // compute the remaining bins
    for (int lv = num_spm_level_ - 2; lv >= 0; --lv)
    {
      const uint32_t level_start_idx = level_start_idx_[lv];
      const uint32_t prev_level_start_idx = level_start_idx_[lv + 1];
      const uint32_t num_blk = level_num_blk_[lv];

      int idx(0);

      for (int ybin = 0; ybin < num_blk; ++ybin)
        for (int xbin = 0; xbin < num_blk; ++xbin)
        {
          const int out_idx = level_start_idx + idx;
          float* const out_code = spm_code + feat_dim * out_idx;

          for (int y_subbin = 0; y_subbin < 2; ++y_subbin)
            for (int x_subbin = 0; x_subbin < 2; ++x_subbin)
            {
              // figure out which subbin we should use for current pooling
              const int in_subbin_idx = prev_level_start_idx + 4 * idx
                  + y_subbin * 2 + x_subbin;
              //cerr << "in index: " << in_subbin_idx << endl;
              const float* const in_code = spm_code + in_subbin_idx * feat_dim;

              if (y_subbin == 0 && x_subbin == 0)
                cblas_scopy(feat_dim, in_code, 1, out_code, 1);
              else
              {
                for (int dd = 0; dd < feat_dim; ++dd)
                  out_code[dd] = std::max(out_code[dd], in_code[dd]);
              }
            }

          ++idx;
        }
    }

  }

  void SPM::MaxPooling(const float* const data, const uint32_t feat_dim,
                       const uint32_t num_data, const float* const pos,
                       shared_ptr<float>* const spm_code)
  {
    if (spm_code == NULL)
    {
      cerr << "SPM::MaxPooling. ERROR: Null pointer of output" << endl;
      exit(-1);
    }

    float* code = new float[total_num_blk_ * feat_dim];
    MaxPooling(data, feat_dim, num_data, pos, code);

    spm_code->reset(code);
  }

}
