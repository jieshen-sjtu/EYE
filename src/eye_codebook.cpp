/*
 * codebook.cpp
 *
 *  Created on: Feb 1, 2014
 *      Author: jieshen
 */

#include "EYE/eye_codebook.hpp"

#include <vl/kmeans.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace EYE
{
  CodeBook::CodeBook()
      : kmeans_model_(NULL), has_setup_(false)
  {
    init_with_default_parameter();
  }

  CodeBook::~CodeBook()
  {
    Clear();
  }

  void CodeBook::Clear()
  {
    init_with_default_parameter();

    if (kmeans_model_ != NULL)
    {
      vl_kmeans_delete(kmeans_model_);
      kmeans_model_ = NULL;
    }

    has_setup_ = false;
  }

  void CodeBook::init_with_default_parameter()
  {
    max_iter_ = DEFAULT_MAX_KMEANS_ITER;
    num_kdtrees_ = DEFAULT_NUM_KDTREES;
    max_comp_ = DEFAULT_MAX_COMP;
    dist_type_ = DEFAULT_DIST_COMP;
  }

  void CodeBook::save(FILE* output, const shared_ptr<float>& cluster,
                      const uint32_t dim, const uint32_t K)
  {
    const float* clusters = cluster.get();
    if (clusters == NULL)
    {
      fprintf(stderr, "Check the clusters\n");
      exit(-1);
    }

    fprintf(output, "K:%u dim:%u\n", K, dim);
    for (uint32_t i = 0; i < K * dim; ++i)
    {
      fprintf(output, "%f ", clusters[i]);
      if ((i + 1) % dim == 0)
        fprintf(output, "\n");
    }
  }

  void CodeBook::load(FILE* input, shared_ptr<float>& _clusters, uint32_t* _dim,
                      uint32_t* _K)
  {

    fscanf(input, "K:%u dim:%u\n", _K, _dim);
    const uint32_t K = *_K;
    const uint32_t dim = *_dim;

    float* clusters = (float*) malloc(sizeof(float) * K * dim);

    for (uint32_t i = 0; i < K * dim; ++i)
    {
      fscanf(input, "%f ", clusters + i);
      if ((i + 1) % dim == 0)
        fscanf(input, "\n");
    }

    _clusters.reset(clusters);
  }

  void CodeBook::SetUp()
  {
    if (kmeans_model_ != NULL)
      vl_kmeans_delete(kmeans_model_);

    kmeans_model_ = vl_kmeans_new(VL_TYPE_FLOAT, dist_type_);
    // use the ANN for fast computation
    vl_kmeans_set_max_num_iterations(kmeans_model_, max_iter_);
    vl_kmeans_set_algorithm(kmeans_model_, VlKMeansANN);
    vl_kmeans_set_num_trees(kmeans_model_, num_kdtrees_);
    vl_kmeans_set_max_num_comparisons(kmeans_model_, max_comp_);

    has_setup_ = true;
  }

  void CodeBook::GenKMeans(const shared_ptr<float>& org_data,
                           const uint32_t num_data, const uint32_t dim,
                           const uint32_t K)
  {
    const float* data = org_data.get();

    if (data == NULL)
    {
      fprintf(stderr, "NULL pointer for data\n");
      exit(-1);
    }

    if (!has_setup_)
      SetUp();

// initialize centers
    vl_kmeans_init_centers_with_rand_data(kmeans_model_, data, dim, num_data,
                                          K);

    vl_kmeans_refine_centers(kmeans_model_, data, num_data);
  }
}

