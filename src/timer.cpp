#include "timer.hpp"

#include <iostream>

using namespace std;
using namespace std::chrono;

#define NS          1000000000      // 10^9

void Timer::Remember() {
    _timestamp = high_resolution_clock::now();
}

int64_t Timer::Diff() {
    high_resolution_clock::time_point before = _timestamp;
    Remember();
    duration<double> timeSpan = duration_cast<duration<double>>(_timestamp - before);
    return static_cast<int64_t>(timeSpan.count() * NS);
}

void Timer::Print() {
    cout << "CPU-Timer: " << Diff() << "ns" << endl;
}
