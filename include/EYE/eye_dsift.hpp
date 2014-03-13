/*
 * eye_dsift.hpp
 *
 *  Created on: Feb 11, 2014
 *      Author: jieshen
 */

#ifndef __EYE_EYE_DSIFT_HPP__
#define __EYE_EYE_DSIFT_HPP__

#include <vl/dsift.h>

#include <stdint.h>
#include <vector>
#include <cmath>
#include <climits>
#include <boost/shared_ptr.hpp>

namespace EYE
{
#define FLT_EPS 1e-10

  using std::vector;
  using std::abs;
  using boost::shared_ptr;

  class DSift
  {
     public:
      DSift();
      ~DSift();

     public:
      void SetUp(const uint32_t width, const uint32_t height);
      void Extract(const float* gray_img, const uint32_t width,
                   const uint32_t height, vector<VlDsiftKeypoint>* frames,
                   vector<float>* descrs, uint32_t* dim);
      void Clear();

      // accessing data
     public:
      inline const vector<uint32_t>& get_sizes() const
      {
        return sizes_;
      }
      inline const bool get_fast() const
      {
        return fast_;
      }
      inline const uint32_t get_step() const
      {
        return step_;
      }
      inline const bool get_float_desc() const
      {
        return float_desc_;
      }
      inline const float get_magnif() const
      {
        return magnif_;
      }
      inline const float get_win_size() const
      {
        return win_size_;
      }
      inline const float get_contr_thrd() const
      {
        return contr_thrd_;
      }
      inline void get_bound(int* minx, int* miny, int* maxx, int* maxy) const
      {
        *minx = bound_minx_;
        *miny = bound_miny_;
        *maxx = bound_maxx_;
        *maxy = bound_maxy_;
      }

      inline void set_sizes(const vector<uint32_t>& sizes)
      {
        sizes_.assign(sizes.begin(), sizes.end());
        has_setup_ = false;
      }
      inline void set_fast(const bool fast)
      {
        if (fast == fast_)
          return;
        fast_ = fast;
        has_setup_ = false;
      }
      inline void set_step(const uint32_t step)
      {
        if (step == step_)
          return;
        step_ = step;
        has_setup_ = false;
      }
      inline void set_float_desc(const bool float_desc)
      {
        if (float_desc == float_desc_)
          return;
        float_desc_ = float_desc;
        has_setup_ = false;
      }
      inline void set_magnif(const float magnif)
      {
        if (abs(magnif - magnif_) < FLT_EPS)
          return;
        magnif_ = magnif;
        has_setup_ = false;
      }
      inline void set_win_size(const float win_sz)
      {
        if (abs(win_sz - win_size_) < FLT_EPS)
          return;
        win_size_ = win_sz;
        has_setup_ = false;
      }
      inline void set_contr_thrd(const float contr_thrd)
      {
        if (abs(contr_thrd - contr_thrd_) < FLT_EPS)
          return;
        contr_thrd_ = contr_thrd;
        has_setup_ = false;
      }
      inline void set_bound(const int* minx, const int* miny, const int* maxx,
                            const int* maxy)
      {
        if (!minx && !miny && !maxx && !maxy)
          return;

        if (minx)
          bound_minx_ = *minx;
        if (miny)
          bound_miny_ = *miny;
        if (maxx)
          bound_maxx_ = *maxx;
        if (maxy)
          bound_maxy_ = *maxy;
        has_setup_ = false;
      }

     private:
      void init_with_default_parameter();
      void clear_data();

     public:
#define DEFAULT_FAST true
#define DEFAULT_STEP 2
#define DEFAULT_FLT_DESC false
#define DEFAULT_MAGNIF 6
#define DEFAULT_WIN_SIZE 1.5
#define DEFAULT_CONTR_THRD 0.005
#define DEFAULT_BOUND_MINX 0
#define DEFAULT_BOUND_MINY 0
#define DEFAULT_BOUND_MAXX INT_MAX
#define DEFAULT_BOUND_MAXY INT_MAX
      enum
      {
        DEFAULT_NUM_BIN_X = 4,
        DEFAULT_NUM_BIN_Y = 4,
        DEFAULT_NUM_BIN_T = 8,
      };

     private:
      VlDsiftFilter* dsift_model_;
      uint32_t width_;
      uint32_t height_;

      // param
      vector<uint32_t> sizes_;
      bool fast_;
      uint32_t step_;
      bool float_desc_;
      float magnif_;
      float win_size_;
      float contr_thrd_;
      int bound_minx_;
      int bound_miny_;
      int bound_maxx_;
      int bound_maxy_;

      bool has_setup_;
  };

}

#endif /* __EYE_EYE_DSIFT_HPP__ */
