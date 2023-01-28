#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <cmath>

#include "detector.h"

Detector::Detector(Config &config)
{
    
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->maxDet = config.maxDet;
    this->modelPath = config.modelPath;
    this->classNamePath = config.classNamePath;
    this->infer_w = config.infer_w;
    this->infer_h = config.infer_h;
    this->_auto = config._auto;

    printf("Construct Yolov5 Detector\n");
    // printf("Load Model\n");
    // if (!loadModel(this->modelPath, this->model))
    // {
    //     printf("load model failure!\n");
    // }
    // else
    // {
    //     printf("Load model successfully\n");
    //     printf("Construct Yolov5 Detector\n");
    // }
}

int Detector::loadModel(const std::string &modelPath, cv::dnn::Net &model)
{
    int ret = -1;
    model = cv::dnn::readNetFromONNX(modelPath);
    if (model.empty())
    {
        fprintf(stdout, "read onnx model data failure...\n");
    }
    else
    {
        ret = 1;
    }
    return ret;
}

void Detector::letterbox(const cv::Mat &img, int infer_w, int infer_h, cv::Mat &dst, bool _auto)
{
    dst = cv::Mat::zeros(cv::Size(infer_w, infer_h), CV_32F);

    float ratio_w = 1.0f * infer_w / img.cols;
    float ratio_h = 1.0f * infer_h / img.rows;
    cv::Mat img_resized;

    float min_ratio = ratio_w > ratio_h ? ratio_h : ratio_w;

    int new_w = (int)(min_ratio * img.cols);
    int new_h = (int)(min_ratio * img.rows);
    // printf("new size: (%d, %d)\n", new_w, new_h);
    cv::resize(img, img_resized, cv::Size(new_w, new_h));

    int dw = infer_w - new_w;
    int dh = infer_h - new_h;

    // printf("dw:%d, dh:%d\n", dw, dh);

    if (_auto)
    {
        dw = MOD(dw, 32);
        dh = MOD(dh, 32);
    }
    
    dw = int(dw / 2);
    dh = int(dh / 2);

    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));
    // printf("top:%d, bottom:%d, left:%d, right:%d\n", top, bottom, left, right);
    cv::copyMakeBorder(img_resized, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    // printf("image pad shape [%d, %d]\n", dst.cols, dst.rows);
    cv::imwrite("./img_pad.jpg", dst);
}

void Detector::inference(cv::Mat &img, cv::Mat &pred, cv::dnn::Net &model)
{
    cv::Mat img_infer;
    letterbox(img, this->infer_w, this->infer_h, img_infer, this->_auto);
    cv::Mat blob;
    cv::dnn::blobFromImage(img_infer, blob, 1 / 255.0f, 
                           cv::Size(img_infer.cols, img_infer.rows),
                           cv::Scalar(0, 0, 0), true, false);

    std::vector<std::string> outLayerNames = model.getUnconnectedOutLayersNames();
    // std::vector<cv::Mat> outs;
    model.setInput(blob);
    model.forward(pred, outLayerNames);
    // std::cout << "outs sizes: " << outs.size() << std::endl;
}

Detector::~Detector()
{
}