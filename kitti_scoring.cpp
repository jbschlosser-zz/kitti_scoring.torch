#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <numeric>
#include <set>
#include <sstream>
#include <stdio.h>
#include <string>
#include <strings.h>
#include <vector>

#include "TH.h"
#include "luaT.h"
extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

#include "kitti_scoring.h"

using namespace std;

/*=======================================================================
STATIC EVALUATION PARAMETERS
=======================================================================*/

const string IMAGE_SET_FILE = "data/kitti/object/ImageSets/val.txt";

// easy, moderate and hard evaluation level
enum DIFFICULTY{EASY=0, MODERATE=1, HARD=2};

// evaluation parameter
const int32_t MIN_HEIGHT[3]     = {40, 25, 25};     // minimum height for evaluated groundtruth/detections
const int32_t MAX_OCCLUSION[3]  = {0, 1, 2};        // maximum occlusion level of the groundtruth used for evaluation
const double  MAX_TRUNCATION[3] = {0.15, 0.3, 0.5}; // maximum truncation level of the groundtruth used for evaluation


// parameters varying per class
vector<string> CLASS_NAMES;
const double   MIN_OVERLAP[3] = {0.7, 0.5, 0.5};                  // the minimum overlap required for evaluation

// no. of recall steps that should be evaluated (discretized)
const double N_SAMPLE_PTS = 41;

// holds the number of test images on the validation set; initialized below
int32_t N_TESTIMAGES = 0;

// initialize class names
void initGlobals (const int32_t num_images) {
  N_TESTIMAGES = num_images;
  CLASS_NAMES.push_back("car");
  CLASS_NAMES.push_back("pedestrian");
  CLASS_NAMES.push_back("cyclist");
}

/*=======================================================================
DATA TYPES FOR EVALUATION
=======================================================================*/

// holding data needed for precision-recall and precision-aos
struct tPrData {
  vector<double> v;           // detection score for computing score thresholds
  double         similarity;  // orientation similarity
  int32_t        tp;          // true positives
  int32_t        fp;          // false positives
  int32_t        fn;          // false negatives
  tPrData () :
    similarity(0), tp(0), fp(0), fn(0) {}
};

// holding bounding boxes for ground truth and detections
struct tBox {
  string  type;     // object type as car, pedestrian or cyclist,...
  double   x1;      // left corner
  double   y1;      // top corner
  double   x2;      // right corner
  double   y2;      // bottom corner
  double   alpha;   // image orientation
  tBox (string type, double x1,double y1,double x2,double y2,double alpha) :
    type(type),x1(x1),y1(y1),x2(x2),y2(y2),alpha(alpha) {}
};

// holding ground truth data
struct tGroundtruth {
  tBox    box;        // object type, box, orientation
  double  truncation; // truncation 0..1
  int32_t occlusion;  // occlusion 0,1,2 (non, partly, fully)
  tGroundtruth () :
    box(tBox("invalild",-1,-1,-1,-1,-10)),truncation(-1),occlusion(-1) {}
  tGroundtruth (tBox box,double truncation,int32_t occlusion) :
    box(box),truncation(truncation),occlusion(occlusion) {}
  tGroundtruth (string type,double x1,double y1,double x2,double y2,double alpha,double truncation,int32_t occlusion) :
    box(tBox(type,x1,y1,x2,y2,alpha)),truncation(truncation),occlusion(occlusion) {}
};

// holding detection data
struct tDetection {
  tBox    box;    // object type, box, orientation
  double  thresh; // detection score
  tDetection ():
    box(tBox("invalid",-1,-1,-1,-1,-10)),thresh(-1000) {}
  tDetection (tBox box,double thresh) :
    box(box),thresh(thresh) {}
  tDetection (string type,double x1,double y1,double x2,double y2,double alpha,double thresh) :
    box(tBox(type,x1,y1,x2,y2,alpha)),thresh(thresh) {}
};

/*=======================================================================
FUNCTIONS TO LOAD DETECTION AND GROUND TRUTH DATA ONCE, SAVE RESULTS
=======================================================================*/

template <class T>
vector<tDetection> loadDetections(T& input_stream, bool &compute_aos, bool &success) {

  // holds all detections (ignored detections are indicated by an index vector
  vector<tDetection> detections;
  while(input_stream.good()) {
    std::string line;
    std::getline(input_stream, line);
    tDetection d;
    double trash;
    char str[255];
    if (sscanf(line.c_str(), "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &trash,    &trash,    &d.box.alpha,
                   &d.box.x1,   &d.box.y1, &d.box.x2, &d.box.y2,
                   &trash,      &trash,    &trash,    &trash,
                   &trash,      &trash,    &trash,    &d.thresh )==16) {
      d.box.type = str;
      detections.push_back(d);

      // orientation=-10 is invalid, AOS is not evaluated if at least one orientation is invalid
      if(d.box.alpha==-10)
        compute_aos = false;
    }
  }
  //fclose(fp);
  success = input_stream.eof();
  return detections;
}

template <class T>
vector<tGroundtruth> loadGroundtruth(T& input_stream, bool &success) {

  // holds all ground truth (ignored ground truth is indicated by an index vector
  vector<tGroundtruth> groundtruth;
  /*FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return groundtruth;
  }
  while (!feof(fp)) {*/
  while(input_stream.good()) {
    std::string line;
    std::getline(input_stream, line);
    tGroundtruth g;
    double trash;
    char str[255];
    if (sscanf(line.c_str(), "%s %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &g.truncation, &g.occlusion, &g.box.alpha,
                   &g.box.x1,   &g.box.y1,     &g.box.x2,    &g.box.y2,
                   &trash,      &trash,        &trash,       &trash,
                   &trash,      &trash,        &trash )==15) {
      g.box.type = str;
      groundtruth.push_back(g);
    }
  }
  //fclose(fp);
  success = input_stream.eof();
  return groundtruth;
}

/*void saveStats (const vector<double> &precision, const vector<double> &aos, FILE *fp_det, FILE *fp_ori) {

  // save precision to file
  if(precision.empty())
    return;
  for (int32_t i=0; i<precision.size(); i++)
    fprintf(fp_det,"%f ",precision[i]);
  fprintf(fp_det,"\n");

  // save orientation similarity, only if there were no invalid orientation entries in submission (alpha=-10)
  if(aos.empty())
    return;
  for (int32_t i=0; i<aos.size(); i++)
    fprintf(fp_ori,"%f ",aos[i]);
  fprintf(fp_ori,"\n");
}*/

/*=======================================================================
EVALUATION HELPER FUNCTIONS
=======================================================================*/

// criterion defines whether the overlap is computed with respect to both areas (ground truth and detection)
// or with respect to box a or b (detection and "dontcare" areas)
inline double boxoverlap(tBox a, tBox b, int32_t criterion=-1){

  // overlap is invalid in the beginning
  double o = -1;

  // get overlapping area
  double x1 = max(a.x1, b.x1);
  double y1 = max(a.y1, b.y1);
  double x2 = min(a.x2, b.x2);
  double y2 = min(a.y2, b.y2);

  // compute width and height of overlapping area
  double w = x2-x1;
  double h = y2-y1;

  // set invalid entries to 0 overlap
  if(w<=0 || h<=0)
    return 0;

  // get overlapping areas
  double inter = w*h;
  double a_area = (a.x2-a.x1) * (a.y2-a.y1);
  double b_area = (b.x2-b.x1) * (b.y2-b.y1);

  // intersection over union overlap depending on users choice
  if(criterion==-1)     // union
    o = inter / (a_area+b_area-inter);
  else if(criterion==0) // bbox_a
    o = inter / a_area;
  else if(criterion==1) // bbox_b
    o = inter / b_area;

  // overlap
  return o;
}

vector<double> getThresholds(vector<double> &v, double n_groundtruth){

  // holds scores needed to compute N_SAMPLE_PTS recall values
  vector<double> t;

  // sort scores in descending order
  // (highest score is assumed to give best/most confident detections)
  sort(v.begin(), v.end(), greater<double>());

  // get scores for linearly spaced recall
  double current_recall = 0;
  for(int32_t i=0; i<v.size(); i++){

    // check if right-hand-side recall with respect to current recall is close than left-hand-side one
    // in this case, skip the current detection score
    double l_recall, r_recall, recall;
    l_recall = (double)(i+1)/n_groundtruth;
    if(i<(v.size()-1))
      r_recall = (double)(i+2)/n_groundtruth;
    else
      r_recall = l_recall;

    if( (r_recall-current_recall) < (current_recall-l_recall) && i<(v.size()-1))
      continue;

    // left recall is the best approximation, so use this and goto next recall step for approximation
    recall = l_recall;

    // the next recall step was reached
    t.push_back(v[i]);
    current_recall += 1.0/(N_SAMPLE_PTS-1.0);
  }
  return t;
}

void cleanData(const std::string& class_name, const vector<tGroundtruth> &gt, const vector<tDetection> &det, vector<int32_t> &ignored_gt, vector<tGroundtruth> &dc, vector<int32_t> &ignored_det, int32_t &n_gt, DIFFICULTY difficulty){

  // extract ground truth bounding boxes for current evaluation class
  for(int32_t i=0;i<gt.size(); i++){

    // only bounding boxes with a minimum height are used for evaluation
    double height = gt[i].box.y2 - gt[i].box.y1;

    // neighboring classes are ignored ("van" for "car" and "person_sitting" for "pedestrian")
    // (lower/upper cases are ignored)
    int32_t valid_class;

    // all classes without a neighboring class
    if(!strcasecmp(gt[i].box.type.c_str(), class_name.c_str()))
      valid_class = 1;

    // classes with a neighboring class
    else if(!strcasecmp(class_name.c_str(), "Pedestrian") && !strcasecmp("Person_sitting", gt[i].box.type.c_str()))
      valid_class = 0;
    else if(!strcasecmp(class_name.c_str(), "Car") && !strcasecmp("Van", gt[i].box.type.c_str()))
      valid_class = 0;

    // classes not used for evaluation
    else
      valid_class = -1;

    // ground truth is ignored, if occlusion, truncation exceeds the difficulty or ground truth is too small
    // (doesn't count as FN nor TP, although detections may be assigned)
    bool ignore = false;
    if(gt[i].occlusion>MAX_OCCLUSION[difficulty] || gt[i].truncation>MAX_TRUNCATION[difficulty] || height<MIN_HEIGHT[difficulty])
      ignore = true;

    // set ignored vector for ground truth
    // current class and not ignored (total no. of ground truth is detected for recall denominator)
    if(valid_class==1 && !ignore){
      ignored_gt.push_back(0);
      n_gt++;
    }

    // neighboring class, or current class but ignored
    else if(valid_class==0 || (ignore && valid_class==1))
      ignored_gt.push_back(1);

    // all other classes which are FN in the evaluation
    else
      ignored_gt.push_back(-1);
  }

  // extract dontcare areas
  for(int32_t i=0;i<gt.size(); i++)
    if(!strcasecmp("DontCare", gt[i].box.type.c_str()))
      dc.push_back(gt[i]);

  // extract detections bounding boxes of the current class
  for(int32_t i=0;i<det.size(); i++){

    // neighboring classes are not evaluated
    int32_t valid_class;
    if(!strcasecmp(det[i].box.type.c_str(), class_name.c_str()))
      valid_class = 1;
    else
      valid_class = -1;

    // set ignored vector for detections
    if(valid_class==1)
      ignored_det.push_back(0);
    else
      ignored_det.push_back(-1);
  }
}

tPrData computeStatistics(double minOverlap, const vector<tGroundtruth> &gt, const vector<tDetection> &det, const vector<tGroundtruth> &dc, const vector<int32_t> &ignored_gt, const vector<int32_t>  &ignored_det, bool compute_fp, bool compute_aos=false, double thresh=0, bool debug=false){

  tPrData stat = tPrData();
  const double NO_DETECTION = -10000000;
  vector<double> delta;            // holds angular difference for TPs (needed for AOS evaluation)
  vector<bool> assigned_detection; // holds wether a detection was assigned to a valid or ignored ground truth
  assigned_detection.assign(det.size(), false);
  vector<bool> ignored_threshold;
  ignored_threshold.assign(det.size(), false); // holds detections with a threshold lower than thresh if FP are computed

  // detections with a low score are ignored for computing precision (needs FP)
  if(compute_fp)
    for(int32_t i=0; i<det.size(); i++)
      if(det[i].thresh<thresh)
        ignored_threshold[i] = true;

  // evaluate all ground truth boxes
  for(int32_t i=0; i<gt.size(); i++){

    // this ground truth is not of the current or a neighboring class and therefore ignored
    if(ignored_gt[i]==-1)
      continue;

    /*=======================================================================
    find candidates (overlap with ground truth > 0.5) (logical len(det))
    =======================================================================*/
    int32_t det_idx          = -1;
    double valid_detection = NO_DETECTION;
    double max_overlap     = 0;

    // search for a possible detection
    bool assigned_ignored_det = false;
    for(int32_t j=0; j<det.size(); j++){

      // detections not of the current class, already assigned or with a low threshold are ignored
      if(ignored_det[j]==-1)
        continue;
      if(assigned_detection[j])
        continue;
      if(ignored_threshold[j])
        continue;

      // find the maximum score for the candidates and get idx of respective detection
      double overlap = boxoverlap(det[j].box, gt[i].box);

      // for computing recall thresholds, the candidate with highest score is considered
      if(!compute_fp && overlap>minOverlap && det[j].thresh>valid_detection){
        det_idx         = j;
        valid_detection = det[j].thresh;
      }

      // for computing pr curve values, the candidate with the greatest overlap is considered
      // if the greatest overlap is an ignored detection (min_height), the overlapping detection is used
      else if(compute_fp && overlap>minOverlap && (overlap>max_overlap || assigned_ignored_det) && ignored_det[j]==0){
        max_overlap     = overlap;
        det_idx         = j;
        valid_detection = 1;
        assigned_ignored_det = false;
      }
      else if(compute_fp && overlap>minOverlap && valid_detection==NO_DETECTION && ignored_det[j]==1){
        det_idx              = j;
        valid_detection      = 1;
        assigned_ignored_det = true;
      }
    }

    /*=======================================================================
    compute TP, FP and FN
    =======================================================================*/

    // nothing was assigned to this valid ground truth
    if(valid_detection==NO_DETECTION && ignored_gt[i]==0)
      stat.fn++;

    // only evaluate valid ground truth <=> detection assignments (considering difficulty level)
    else if(valid_detection!=NO_DETECTION && (ignored_gt[i]==1 || ignored_det[det_idx]==1))
      assigned_detection[det_idx] = true;

    // found a valid true positive
    else if(valid_detection!=NO_DETECTION){

      // write highest score to threshold vector
      stat.tp++;
      stat.v.push_back(det[det_idx].thresh);

      // compute angular difference of detection and ground truth if valid detection orientation was provided
      if(compute_aos)
        delta.push_back(gt[i].box.alpha - det[det_idx].box.alpha);

      // clean up
      assigned_detection[det_idx] = true;
    }
  }

  // if FP are requested, consider stuff area
  if(compute_fp){

    // count fp
    for(int32_t i=0; i<det.size(); i++){

      // count false positives if required (height smaller than required is ignored (ignored_det==1)
      if(!(assigned_detection[i] || ignored_det[i]==-1 || ignored_det[i]==1 || ignored_threshold[i]))
        stat.fp++;
    }

    // do not consider detections overlapping with stuff area
    int32_t nstuff = 0;
    for(int32_t i=0; i<dc.size(); i++){
      for(int32_t j=0; j<det.size(); j++){

        // detections not of the current class, already assigned, with a low threshold or a low minimum height are ignored
        if(assigned_detection[j])
          continue;
        if(ignored_det[j]==-1 || ignored_det[j]==1)
          continue;
        if(ignored_threshold[j])
          continue;

        // compute overlap and assign to stuff area, if overlap exceeds class specific value
        double overlap = boxoverlap(det[j].box, dc[i].box, 0);
        if(overlap>minOverlap){
          assigned_detection[j] = true;
          nstuff++;
        }
      }
    }

    // FP = no. of all not to ground truth assigned detections - detections assigned to stuff areas
    stat.fp -= nstuff;

    // if all orientation values are valid, the AOS is computed
    if(compute_aos){
      vector<double> tmp;

      // FP have a similarity of 0, for all TP compute AOS
      tmp.assign(stat.fp, 0);
      for(int32_t i=0; i<delta.size(); i++)
        tmp.push_back((1.0+cos(delta[i]))/2.0);

      // be sure, that all orientation deltas are computed
      assert(tmp.size()==stat.fp+stat.tp);
      assert(delta.size()==stat.tp);

      // get the mean orientation similarity for this image
      if(stat.tp>0 || stat.fp>0)
        stat.similarity = accumulate(tmp.begin(), tmp.end(), 0.0);

      // there was neither a FP nor a TP, so the similarity is ignored in the evaluation
      else
        stat.similarity = -1;
    }
  }
  return stat;
}

/*=======================================================================
EVALUATE CLASS-WISE
=======================================================================*/

bool eval_class (const ClassData& current_class,const vector< vector<tGroundtruth> > &groundtruth,const vector< vector<tDetection> > &detections, bool compute_aos, vector<double> &precision, vector<double> &aos, DIFFICULTY difficulty) {

  // init
  int32_t n_gt=0;                                     // total no. of gt (denominator of recall)
  vector<double> v, thresholds;                       // detection scores, evaluated for recall discretization
  vector< vector<int32_t> > ignored_gt, ignored_det;  // index of ignored gt detection for current class/difficulty
  vector< vector<tGroundtruth> > dontcare;            // index of dontcare areas, included in ground truth

  // for all test images do
  for (int32_t i=0; i<N_TESTIMAGES; i++){

    // holds ignored ground truth, ignored detections and dontcare areas for current frame
    vector<int32_t> i_gt, i_det;
    vector<tGroundtruth> dc;

    // only evaluate objects of current class and ignore occluded, truncated objects
    cleanData(current_class.name, groundtruth[i], detections[i], i_gt, dc, i_det, n_gt, difficulty);
    ignored_gt.push_back(i_gt);
    ignored_det.push_back(i_det);
    dontcare.push_back(dc);

    // compute statistics to get recall values
    tPrData pr_tmp = tPrData();
    pr_tmp = computeStatistics(current_class.minOverlap, groundtruth[i], detections[i], dc, i_gt, i_det, false);

    // add detection scores to vector over all images
    for(int32_t j=0; j<pr_tmp.v.size(); j++)
      v.push_back(pr_tmp.v[j]);
  }

  // get scores that must be evaluated for recall discretization
  thresholds = getThresholds(v, n_gt);

  // compute TP,FP,FN for relevant scores
  vector<tPrData> pr;
  pr.assign(thresholds.size(),tPrData());
  for (int32_t i=0; i<N_TESTIMAGES; i++){

    // for all scores/recall thresholds do:
    for(int32_t t=0; t<thresholds.size(); t++){
      tPrData tmp = tPrData();
      tmp = computeStatistics(current_class.minOverlap, groundtruth[i], detections[i], dontcare[i],
                              ignored_gt[i], ignored_det[i], true, compute_aos, thresholds[t], t==38);

      // add no. of TP, FP, FN, AOS for current frame to total evaluation for current threshold
      pr[t].tp += tmp.tp;
      pr[t].fp += tmp.fp;
      pr[t].fn += tmp.fn;
      if(tmp.similarity!=-1)
        pr[t].similarity += tmp.similarity;
    }
  }

  // compute recall, precision and AOS
  vector<double> recall;
  precision.assign(N_SAMPLE_PTS, 0);
  if(compute_aos)
    aos.assign(N_SAMPLE_PTS, 0);
  double r=0;
  for (int32_t i=0; i<thresholds.size(); i++){
    r = pr[i].tp/(double)(pr[i].tp + pr[i].fn);
    recall.push_back(r);
    precision[i] = pr[i].tp/(double)(pr[i].tp + pr[i].fp);
    if(compute_aos)
      aos[i] = pr[i].similarity/(double)(pr[i].tp + pr[i].fp);
  }

  // filter precision and AOS using max_{i..end}(precision)
  for (int32_t i=0; i<thresholds.size(); i++){
    precision[i] = *max_element(precision.begin()+i, precision.end());
    if(compute_aos)
      aos[i] = *max_element(aos.begin()+i, aos.end());
  }

  // save statisics and finish with success
  //saveStats(precision, aos, fp_det, fp_ori);
  return true;
}

void saveAndPlotPlots(string dir_name,string file_name,string obj_type,vector<double> vals[],bool is_aos){
  // Create plot directory if it doesn't exist.
  system(("mkdir -p " + dir_name).c_str());

  char command[1024];

  // save plot data to file
  FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(),"w");
  for (int32_t i=0; i<(int)N_SAMPLE_PTS; i++)
    fprintf(fp,"%f %f %f %f\n",(double)i/(N_SAMPLE_PTS-1.0),vals[0][i],vals[1][i],vals[2][i]);
  fclose(fp);

  // create png + eps
  for (int32_t j=0; j<2; j++) {

    // open file
    FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(),"w");

    // save gnuplot instructions
    if (j==0) {
      fprintf(fp,"set term png size 450,315 font \"Helvetica\" 11\n");
      fprintf(fp,"set output \"%s.png\"\n",file_name.c_str());
    } else {
      fprintf(fp,"set term postscript eps enhanced color font \"Helvetica\" 20\n");
      fprintf(fp,"set output \"%s.eps\"\n",file_name.c_str());
    }

    // set labels and ranges
    fprintf(fp,"set size ratio 0.7\n");
    fprintf(fp,"set xrange [0:1]\n");
    fprintf(fp,"set yrange [0:1]\n");
    fprintf(fp,"set xlabel \"Recall\"\n");
    if (!is_aos) fprintf(fp,"set ylabel \"Precision\"\n");
    else         fprintf(fp,"set ylabel \"Orientation Similarity\"\n");
    obj_type[0] = toupper(obj_type[0]);
    fprintf(fp,"set title \"%s\"\n",obj_type.c_str());

    // line width
    int32_t   lw = 5;
    if (j==0) lw = 3;

    // plot error curve
    fprintf(fp,"plot ");
    fprintf(fp,"\"%s.txt\" using 1:2 title 'Easy' with lines ls 1 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:3 title 'Moderate' with lines ls 2 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:4 title 'Hard' with lines ls 3 lw %d",file_name.c_str(),lw);

    // close file
    fclose(fp);

    // run gnuplot => create png + eps
    sprintf(command,"cd %s; gnuplot %s",dir_name.c_str(),(file_name + ".gp").c_str());
    system(command);
  }

  // create pdf and crop
  sprintf(command,"cd %s; ps2pdf %s.eps %s_large.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; pdfcrop %s_large.pdf %s.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; rm %s_large.pdf",dir_name.c_str(),file_name.c_str());
  system(command);
}

std::vector<double> calcAP(const std::vector<double> precision[3]) {
    std::vector<double> aps;
    for(int i = 0; i < 3; ++i) {
        double ap = 0.0;
        double size = (double)precision[i].size();
        for(int j = 0; j < precision[i].size(); j += 4) {
            double prec = precision[i][j];
            double this_ap = prec / 11.0; // 11 precision steps for every 4
            ap += this_ap;
        }
        aps.push_back(ap);
    }

    return aps;
}

// gt_dir - path to ground truth directory.
// image_set - set of image numbers (e.g. 6590)
bool score_kitti(const std::string& gt_dir,
                 const std::set<int>& image_set,
                 const std::string& results_dir,
                 const std::vector<ClassData>& classes_to_score,
                 //const std::map<int, std::string>& detection_strings,
                 AveragePrecisionResults& results) {

  // set some global parameters
  initGlobals(image_set.size());

  // ground truth and result directories
  string plot_dir = "./plot";

  // hold detections and ground truth in memory
  vector< vector<tGroundtruth> > groundtruth;
  vector< vector<tDetection> >   detections;

  // holds whether orientation similarity shall be computed (might be set to false while loading detections)
  // and which labels where provided by this submission
  bool compute_aos=true;
  vector<bool> shouldEvalClass;
  for(int i = 0; i < classes_to_score.size(); ++i) {
      shouldEvalClass.push_back(false);
  }

  // for all images read groundtruth and detections
  for(std::set<int>::const_iterator image_num = image_set.begin();
      image_num != image_set.end();
      ++image_num) {

    std::stringstream file_name_stream;
    file_name_stream << setfill('0') << setw(6) << *image_num << ".txt";
    std::string file_name = file_name_stream.str();

    // read ground truth and result poses
    bool gt_success,det_success;
    std::ifstream gt_file((gt_dir + "/" + file_name).c_str());
    vector<tGroundtruth> gt = loadGroundtruth(gt_file, gt_success);
    gt_file.close();

    // Load detections from file.
    std::ifstream result_file((results_dir + "/" + file_name).c_str());
    vector<tDetection> det = loadDetections(result_file, compute_aos, det_success);

    // Check which classes should be evaluated (only ones that have been seen at least
    // once in the detections).
    for(vector<tDetection>::const_iterator d = det.begin(); d != det.end(); ++d) {
        for(int i = 0; i < classes_to_score.size(); ++i) {
            if(!strcasecmp(classes_to_score[i].name.c_str(), d->box.type.c_str())) {
                shouldEvalClass[i] = true;
            }
        }
    }
    result_file.close();
    groundtruth.push_back(gt);
    detections.push_back(det);

    // check for errors
    if (!gt_success) {
      return false;
    }
    if (!det_success) {
      return false;
    }
  }

  // Evaluate each of the classes.
  for(int i = 0; i < classes_to_score.size(); ++i) {
      if(shouldEvalClass[i]) {
          vector<double> precision[3], aos[3];
          if(!eval_class(classes_to_score[i],groundtruth,detections,compute_aos,precision[0],aos[0],EASY)
            || !eval_class(classes_to_score[i],groundtruth,detections,compute_aos,precision[1],aos[1],MODERATE)
            || !eval_class(classes_to_score[i],groundtruth,detections,compute_aos,precision[2],aos[2],HARD))
          {
              return false;
          }

          std::vector<double> aps = calcAP(precision);
          results.aps.push_back(std::make_pair(classes_to_score[i].name, aps));
      }
  }

  // success
  return true;
}

// Lua stuff.
static int lua_score_kitti(lua_State *L) {
    std::string usage_msg = "Usage: score_kitti <gt_dir> <image_set> <results_dir> <classes_to_score>";

    // Parse the gt dir.
    size_t gt_dir_len;
    const char *gt_dir_c_str = lua_tolstring(L, 1, &gt_dir_len);
    if(gt_dir_c_str == NULL) {
        return luaL_error(L, usage_msg.c_str());
    }
    std::string gt_dir(gt_dir_c_str, gt_dir_len);

    // Parse the image set.
    std::set<int> image_set;
    if(lua_istable(L, 2) == 0) {
        return luaL_error(L, usage_msg.c_str());
    }
    lua_pushnil(L); // first key
    while (lua_next(L, 2) != 0) {
        // uses 'key' (at index -2) and 'value' (at index -1)
        if(lua_isnumber(L, -1) == 0) {
            return luaL_error(L, usage_msg.c_str());
        }
        int image_num = lua_tointeger(L, -1);
        image_set.insert(image_num);
        // removes 'value'; keeps 'key' for next iteration
        lua_pop(L, 1);
    }

    // Parse the results dir.
    size_t results_dir_len;
    const char *results_dir_c_str = lua_tolstring(L, 3, &results_dir_len);
    if(results_dir_c_str == NULL) {
        return luaL_error(L, usage_msg.c_str());
    }
    std::string results_dir(results_dir_c_str, results_dir_len);

    // Parse the classes to score; expecting name and min overlap for each.
    std::vector<ClassData> classes_to_score;
    if(lua_istable(L, 4) == 0) {
        return luaL_error(L, usage_msg.c_str());
    }
    lua_pushnil(L); // first key
    while(lua_next(L, 4) != 0) {
        // Sub-table.
        if(lua_istable(L, -1) == 0) {
            return luaL_error(L, usage_msg.c_str());
        }

        // Get name from sub-table.
        ClassData classData;
        lua_pushstring(L, "name");
        lua_gettable(L, -2);
        size_t name_len;
        const char *name_c_str = lua_tolstring(L, -1, &name_len);
        if(name_c_str == NULL) {
            return luaL_error(L, usage_msg.c_str());
        }
        classData.name = std::string(name_c_str, name_len);
        lua_pop(L, 1);

        // Get min overlap from sub-table.
        lua_pushstring(L, "minOverlap");
        lua_gettable(L, -2);
        if(lua_isnumber(L, -1) == 0) {
            return luaL_error(L, usage_msg.c_str());
        }
        classData.minOverlap = lua_tonumber(L, -1);
        lua_pop(L, 1);

        // Store class data.
        classes_to_score.push_back(classData);

        // Go to next sub-table.
        lua_pop(L, 1);
    }

    // Run the scoring function.
    AveragePrecisionResults results;
    if(!score_kitti(gt_dir, image_set, results_dir, classes_to_score, results)) {
        return luaL_error(L, "An error occurred when calculating scores");
    }

    lua_createtable(L, results.aps.size(), 0);
    for(std::vector<std::pair<std::string, std::vector<double> > >::const_iterator result = results.aps.begin();
        result != results.aps.end();
        ++result)
    {
        // Create the table of results.
        lua_pushstring(L, result->first.c_str());
        lua_newtable(L);
        for(int i = 0; i < result->second.size(); ++i) {
            lua_pushnumber(L, i + 1);
            lua_pushnumber(L, result->second[i]);
            lua_settable(L, -3);
        }
        lua_settable(L, -3);
    }

    return 1;

    // Run the scoring function.
    /*AveragePrecisionResults results;
    if(!score_kitti(gt_dir, image_set, results_dir, results)) {
        return luaL_error(L, "An error occurred when calculating scores");
    }

    // Create car results.
    lua_createtable(L, 3, 0);
    lua_pushstring(L, "car");
    lua_newtable(L);
    for(int i = 0; i < results.car_aps.size(); ++i) {
        lua_pushnumber(L, i + 1);
        lua_pushnumber(L, results.car_aps[i]);
        lua_settable(L, -3);
    }
    lua_settable(L, -3);

    // Create pedestrian results.
    lua_pushstring(L, "pedestrian");
    lua_newtable(L);
    for(int i = 0; i < results.pedestrian_aps.size(); ++i) {
        lua_pushnumber(L, i + 1);
        lua_pushnumber(L, results.pedestrian_aps[i]);
        lua_settable(L, -3);
    }
    lua_settable(L, -3);
    
    // Create cyclist results.
    lua_pushstring(L, "cyclist");
    lua_newtable(L);
    for(int i = 0; i < results.cyclist_aps.size(); ++i) {
        lua_pushnumber(L, i + 1);
        lua_pushnumber(L, results.cyclist_aps[i]);
        lua_settable(L, -3);
    }
    lua_settable(L, -3);

    // Return the table.
    return 1;*/
}

static const struct luaL_reg kitti_scoring[] = {
    {"score_kitti", lua_score_kitti},
    {NULL, NULL}
};

LUA_EXTERNC int luaopen_libkitti_scoring(lua_State *L);

int luaopen_libkitti_scoring(lua_State *L) {
    luaL_openlib(L, "kitti_scoring", kitti_scoring, 0);
    return 1;
}
