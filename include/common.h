#ifndef COMMON_H
#define COMMON_H

#include <queue>
#include <mutex>
#include <thread>
#include <opencv2/core/utility.hpp>

template <typename T>
class QueueFPS : public std::queue<T>
{
public:
    QueueFPS() : counter(0) {}

    void push(const T & entry);
    T get();
    float getFPS();
    void clear();

    unsigned int counter;
private:
    cv::TickMeter tm;
    std::mutex mutex;
};

#endif