#ifndef __TIMER_H
#define __TIMER_H

#include <chrono>
#include <cstdint>

/**
 * A simple timer class. Uses std::chrono as backend.
 * The duration returned by this class is in nanoseconds (= 10^(-9))
 */
class Timer {
public:
    explicit Timer() : _timestamp() {
        Remember();
    }
    virtual ~Timer() = default;

    void    Remember();
    int64_t Diff();
    void Print();

private:
    std::chrono::high_resolution_clock::time_point _timestamp;
};

#endif // __TIMER_H
