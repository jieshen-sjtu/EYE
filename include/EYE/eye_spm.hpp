/*
 * eye_spm.hpp
 *
 *  Created on: Mar 16, 2014
 *      Author: jieshen
 */

#ifndef __EYE_EYE_SPM_HPP__
#define __EYE_EYE_SPM_HPP__

#include <stdint.h>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>
using std::vector;
using std::map;
using std::pair;
using boost::shared_ptr;

namespace EYE
{
  class SPM
  {
     public:
      SPM();
      void SetUp(const uint32_t width, const uint32_t height);

     public:
      // accessing data
      void set_num_spm_level(const uint32_t num_spm_level)
      {
        if (num_spm_level == num_spm_level_)
          return;
        num_spm_level_ = num_spm_level;
        has_setup_ = false;
      }

      void set_same_geom(const bool same)
      {
        same_geom_ = same;
      }

      uint32_t get_num_spm_level() const
      {
        return num_spm_level_;
      }

      bool get_same_geom() const
      {
        return same_geom_;
      }

      uint32_t get_total_num_blk() const
      {
        return total_num_blk_;
      }

      const map<uint32_t, vector<uint32_t> >& get_map_cell_blks() const
      {
        return map_cell_blk_;
      }
      const map<uint32_t, vector<uint32_t> >& get_map_blk_cells() const
      {
        return map_blk_cell_;
      }

     public:
      void MaxPooling(const float* const data, const uint32_t feat_dim,
                      const uint32_t num_data, const float* const pos,
                      float* const spm_code);
      void MaxPooling(const float* const data, const uint32_t feat_dim,
                      const uint32_t num_data, const float* const pos,
                      shared_ptr<float>* const spm_code);
      void build_cell_blk_map(const float* const pos, const uint32_t num_data);
     private:
      void init_with_default_parameter();
      uint32_t get_block_start_idx(const uint32_t level, const uint32_t yidx,
                                   const uint32_t xidx);

     public:
#define DEFAULT_SPM_LEVEL 3

     private:
      // param
      int num_spm_level_;

      uint32_t finest_num_blk_;
      uint32_t total_num_blk_;
      vector<uint32_t> level_num_blk_;

      vector<vector<float> > blk_start_x_;
      vector<vector<float> > blk_end_x_;
      vector<vector<float> > blk_start_y_;
      vector<vector<float> > blk_end_y_;

      // image related data
      uint32_t img_width_;
      uint32_t img_height_;

      // aux data
      vector<uint32_t> level_start_idx_;

      map<uint32_t, vector<uint32_t> > map_cell_blk_;
      map<uint32_t, vector<uint32_t> > map_blk_cell_;

      // to save time for map_cell_blk and map_blk_cell
      bool same_geom_;
      bool has_built_map_;

      bool has_setup_;
  };
}

#endif /* __EYE_EYE_SPM_HPP__ */
