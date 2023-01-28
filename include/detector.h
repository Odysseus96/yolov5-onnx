#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#define MOD(num, divi) (num - divi * int(num / divi))

struct Pred
{
    std::vector<cv::Mat> detection;
};

struct Config
{
    float confThreshold;
    float nmsThreshold;
    std::string modelPath;
    std::string classNamePath;
    int infer_w;
    int infer_h;
    float maxDet;
    bool _auto;
};

class Detector
{
private:
    cv::dnn::Net model;
    float nmsThreshold;
    float confThreshold;
    float maxDet;
    std::string modelPath;
    std::string classNamePath;
    int infer_w;
    int infer_h;
    bool _auto;
public:
    Detector(Config &config);
    ~Detector();
    int loadModel(const std::string &modelPath, cv::dnn::Net &model);
    void letterbox(const cv::Mat &img, int infer_w, int infer_h, cv::Mat &dst, bool _auto);
    void inference(cv::Mat &img, cv::Mat &pred, cv::dnn::Net &model);
    void nonMaxSuppression();
    void postProcess();
};

#endif