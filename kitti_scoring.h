#ifndef _KITTI_SCORING_H_
#define _KITTI_SCORING_H_

struct AveragePrecisionResults {
    std::vector<double> car_aps;
    std::vector<double> pedestrian_aps;
    std::vector<double> cyclist_aps;
};

// gt_dir - path to ground truth directory.
// image_set - set of image numbers (e.g. 6590).
// detection_strings - map of image numbers to KITTI-formatted detection strings.
// results - output average precisions (easy, medium, hard) for evaluated classes.
bool score_kitti(const std::string& gt_dir,
                 const std::set<int>& image_set,
                 const std::string& results_dir,
                 //const std::map<int, std::string>& detection_strings,
                 AveragePrecisionResults& results);

#endif
