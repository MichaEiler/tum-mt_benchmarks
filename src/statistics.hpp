#ifndef __BENCH_STATISTICS_HPP
#define __BENCH_STATISTICS_HPP

#include <cmath>
#include <cstdint>
#include <vector>

/**
 * Handles some statistic evaluations.
 * Calculates things like mean value and standard deviation.
 */
template <typename T>
class Statistics {
private:
    std::vector<T> _values;
    T _sumValue;
    T _min;
    T _max;
public:
    explicit Statistics()
        : _values()
        , _sumValue(static_cast<T>(0))
        , _min(static_cast<T>(INT32_MAX))
        , _max(static_cast<T>(INT32_MIN)) {

    }

    virtual ~Statistics() { }

    void Clear() {
        _sumValue = static_cast<T>(0);
        _values.clear();
        _min = static_cast<T>(INT32_MAX);
        _max = static_cast<T>(INT32_MIN);
    }

    void Add(T value) {
        _sumValue += value;
        _values.push_back(value);

        if (value > _max)
            _max = value;

        if (value < _min)
            _min = value;
    }

    T Min() {
        return _min;
    }

    T Max() {
        return _max;
    }

    T Mean() {
        return _sumValue / static_cast<T>(_values.size());
    }

    T Sum() {
        return _sumValue;
    }

    template <typename TValue>
    TValue Deviation() {
        TValue result = static_cast<TValue>(0);
        TValue mean = static_cast<TValue>(Mean());

        for (auto& value : _values) {
            // sum of (x_i - mean)^2 
            result += static_cast<TValue>(pow(static_cast<double>(value) - mean, 2));
        }

        result /= _values.size();   // variance

        return static_cast<TValue>(sqrt(result)); // standard deviation
    }

};

#endif // __BENCH_STATISTICS_HPP