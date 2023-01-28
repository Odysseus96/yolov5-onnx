#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "detector.h"
#include "common.h"

int main(int argc, char* argv[])
{
    const char* imagePath = "data/images/bus.jpg";
    std::string modelPath = "models/yolov5s.onnx";
    if (argc >= 2)
    {
        imagePath = argv[1];
        modelPath = argv[2];
    }
    Config config = {0.25f, 0.45f, modelPath, "data/coco.names", 640, 640, true};
    Detector detector(config);
    cv::Mat img = cv::imread(imagePath);
    cv::Mat dst;
    printf("read image: %s\n", imagePath);
    // detector.letterbox(img, config.infer_w, config.infer_h, dst, config._auto);
    cv::Mat pred;
    cv::dnn::Net yolov5Model;
    printf("Load Model\n");
    if (!detector.loadModel(modelPath, yolov5Model))
    {
        printf("load model failure!\n");
        return -1;
    }
    fprintf(stdout, "Load model success!\n");
    
    detector.inference(img, pred, yolov5Model);
    std::cout << "pred shape: " << pred.size << std::endl;
    return 0;
}