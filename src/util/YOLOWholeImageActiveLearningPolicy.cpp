/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ActiveLearningPolicy.h"

#include <cmath>

namespace Conv {

datum YOLOWholeImageActiveLearningPolicy::Score(Tensor &output, DatasetMetadataPointer *metadata, unsigned int index) {
#ifdef BUILD_OPENCL
  output.MoveToCPU();
#endif
  DetectionMetadataPointer* detection_metadata = (DetectionMetadataPointer*) metadata;
  DetectionMetadata& proposals = *(detection_metadata[index]);

  unsigned int total_maps = output.maps();
  unsigned int maps_per_cell = total_maps / (horizontal_cells_ * vertical_cells_);
  unsigned int classes_ = maps_per_cell - (5 * boxes_per_cell_);

  // Prepare indices into the prediction array
  unsigned int sample_index = index * output.maps();
  unsigned int class_index = sample_index + (vertical_cells_ * horizontal_cells_ * boxes_per_cell_ * 5);
  unsigned int iou_index = sample_index;

  datum total_score = 0;

  // Loop over all cells
  for (unsigned int vcell = 0; vcell < vertical_cells_; vcell++) {
    for (unsigned int hcell = 0; hcell < horizontal_cells_; hcell++) {
      unsigned int cell_id = vcell * horizontal_cells_ + hcell;

      datum max_class_score = 0;

      // Loop over all classes
      for (unsigned int c = 0; c < classes_; c++) {
        datum class_prob = output.data_ptr_const()[class_index + (horizontal_cells_ * vertical_cells_ * c) + cell_id];
        if(class_prob > max_class_score) max_class_score = class_prob;
      }

      datum cell_score = 0;
      // Loop over all possible boxes
      for (unsigned int b = 0; b < boxes_per_cell_; b++) {

        // Get predicted IOU
        const datum iou = output.data_ptr_const()[iou_index + (cell_id * boxes_per_cell_ + b) * 5 + 4];
        const datum box_score = iou - max_class_score;
        cell_score += (box_score * box_score);
      }

      total_score += cell_score;
    }
  }

  return total_score;
}

}