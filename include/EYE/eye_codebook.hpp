/*
 * eye_codebook.hpp
 *
 *  Created on: Feb 11, 2014
 *      Author: jieshen
 */

#ifndef __EYE_EYE_CODEBOOK_HPP__
#define __EYE_EYE_CODEBOOK_HPP__

#include <vl/kmeans.h>

#include <stdint.h>
#include <cstddef>
#include <cstdio>
#include <boost/shared_ptr.hpp>

namespace EYE
{
  using boost::shared_ptr;

  class CodeBook
  {
  public:
    enum
    {
      DEFAULT_MAX_KMEANS_ITER = 100,
      DEFAULT_NUM_KDTREES = 3,
      DEFAULT_MAX_COMP = 500,
    };
#define DEFAULT_DIST_COMP VlDistanceL2

    // constructor and destructor
  public:
    CodeBook();
    ~CodeBook();

  public:
    void SetUp();
    void Clear();

  private:
    void init_with_default_parameter();

    // setting and accessing
  public:
    inline void set_max_iter(const uint32_t max_iter)
    {
      if (max_iter == max_iter_)
        return;
      max_iter_ = max_iter;
      has_setup_ = false;
    }
    inline void set_num_kdtrees(const uint32_t num_kdtree)
    {
      if (num_kdtree == num_kdtrees_)
        return;
      num_kdtrees_ = num_kdtree;
      has_setup_ = false;
    }
    inline void set_max_comp(const uint32_t max_comp)
    {
      if (max_comp == max_comp_)
        return;
      max_comp_ = max_comp;
      has_setup_ = false;
    }
    inline void set_dist_type(const VlVectorComparisonType type)
    {
      if (type == dist_type_)
        return;
      dist_type_ = type;
      has_setup_ = false;
    }

    inline const float* get_clusters() const
    {
      if (kmeans_model_ == NULL)
        return NULL;
      return ((const float*) vl_kmeans_get_centers(kmeans_model_));
    }
    inline const uint32_t get_max_iter() const
    {
      return max_iter_;
    }
    inline const uint32_t get_num_kdtrees() const
    {
      return num_kdtrees_;
    }
    inline const uint32_t get_max_comp() const
    {
      return max_comp_;
    }
    inline const VlVectorComparisonType get_dist_type() const
    {
      return dist_type_;
    }

    // IO operation
  public:
    static void save(FILE* output, const shared_ptr<float>& clusters,
                     const uint32_t dim, const uint32_t K);
    static void save(FILE* output, const float* clusters, const uint32_t dim,
                     const uint32_t K);
    static void load(FILE* input, shared_ptr<float>& clusters, uint32_t* dim,
                     uint32_t* K);

  private:
    VlKMeans* kmeans_model_;

    // parameter
    uint32_t max_iter_;
    uint32_t num_kdtrees_;
    uint32_t max_comp_;
    VlVectorComparisonType dist_type_;

    // setup
    bool has_setup_;

  public:
    // ensure that the clusters should be allocate the memory outside
    void GenKMeans(const shared_ptr<float>& data, const uint32_t num_data,
                   const uint32_t dim, const uint32_t K);
    void GenKMeans(const float* data, const uint32_t num_data,
                   const uint32_t dim, const uint32_t K);
  };
}

#endif /* __EYE_EYE_CODEBOOK_HPP__ */
