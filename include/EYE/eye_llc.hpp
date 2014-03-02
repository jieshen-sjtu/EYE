/*
 * eye_llc.hpp
 *
 *  Created on: Feb 11, 2014
 *      Author: jieshen
 */

#ifndef __EYE_EYE_LLC_HPP__
#define __EYE_EYE_LLC_HPP__

#include <stdint.h>
#include <cmath>
#include <vl/kdtree.h>
#include <boost/shared_ptr.hpp>

namespace EYE
{
  using boost::shared_ptr;

  class LLC
  {
      // constructor and destructor
     public:
      LLC();
      LLC(const shared_ptr<float>& base, const uint32_t dim,
          const uint32_t num_base);
      ~LLC();

      // must call this function before encoder!!!
      void SetUp();
      void Clear();

     public:
      void Encode(const float* const data, const uint32_t dim,
                  const uint32_t num_frame,
                  shared_ptr<float>* const codes) const;
      void Encode_with_max_pooling(const float* const data, const uint32_t dim,
                                   const uint32_t num_frame,
                                   shared_ptr<float>* const codes) const;

      // !Note: Must allocate memory outside before calling these two
      void Encode(const float* const data, const uint32_t dim,
                  const uint32_t num_frame,
                  float* const codes) const;
      void Encode_with_max_pooling(const float* const data, const uint32_t dim,
                                   const uint32_t num_frame,
                                   float* const codes) const;

     private:
      void init_with_default_parameter();
      void clear_data();

     public:
      // setting and accessing
      void set_base(const shared_ptr<float>& base, const uint32_t dim,
                    const uint32_t num_base);

      inline void set_thrd_method(const VlKDTreeThresholdingMethod method)
      {
        if (method == thrd_method_)
          return;
        thrd_method_ = method;
        has_setup_ = false;
      }
      inline void set_dist_method(const VlVectorComparisonType method)
      {
        if (method == dist_method_)
          return;
        dist_method_ = method;
        has_setup_ = false;
      }
      inline void set_num_tree(const uint32_t num_tree)
      {
        if (num_tree == num_tree_)
          return;
        num_tree_ = num_tree;
        has_setup_ = false;
      }
      inline void set_num_knn(const uint32_t num_knn)
      {
        if (num_knn == num_knn_)
          return;
        num_knn_ = num_knn;
        has_setup_ = false;
      }
      inline void set_max_comp(const uint32_t max_comp)
      {
        if (max_comp == max_comp_)
          return;
        max_comp_ = max_comp;
        has_setup_ = false;
      }
      inline void set_beta(const float beta)
      {
        if (std::abs(beta - beta_) < 1e-10)
          return;
        beta_ = beta;
        has_setup_ = false;
      }

      inline const float* get_base() const
      {
        return base_.get();
      }
      inline const uint32_t get_dim() const
      {
        return dim_;
      }
      inline const uint32_t get_num_base() const
      {
        return num_base_;
      }
      inline const VlKDTreeThresholdingMethod get_thrd_method() const
      {
        return thrd_method_;
      }
      inline const VlVectorComparisonType get_dist_method() const
      {
        return dist_method_;
      }
      inline const uint32_t get_num_tree() const
      {
        return num_tree_;
      }
      inline const uint32_t get_num_knn() const
      {
        return num_knn_;
      }
      inline const uint32_t get_max_comp() const
      {
        return max_comp_;
      }
      inline const float get_beta() const
      {
        return beta_;
      }

     public:
      enum
      {
        DEFAULT_NUM_TREE = 1,
        DEFAULT_NUM_KNN = 5,
        DEFAULT_MAX_COMP = 500,
      };
#define DEFAULT_THRD_METHOD VL_KDTREE_MEDIAN
#define DEFAULT_DIST_METHOD VlDistanceL2
#define DEFAULT_BETA 1e-4

     private:
      // base data
      shared_ptr<float> base_;  // the actual base, get from shared_ptr
      uint32_t dim_;
      uint32_t num_base_;

      // kd-forest data
      VlKDForest* kdforest_model_;
      VlKDTreeThresholdingMethod thrd_method_;
      VlVectorComparisonType dist_method_;
      uint32_t num_tree_;
      uint32_t num_knn_;
      uint32_t max_comp_;

      // LLC parameter
      float beta_;

      // tag
      bool has_setup_;
  };
}

#endif /* __EYE_EYE_LLC_HPP__ */
