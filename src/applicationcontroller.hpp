#ifndef __BENCH_APPLICATIONCONTROLLER_HPP
#define __BENCH_APPLICATIONCONTROLLER_HPP

#include <memory>
#include <vector>

#include "benchmarkbase.hpp"
#include "computecontroller.hpp"

/**
 * Handles user interface interaction.
 * Parses the arguments.
 * Calls the ComputeController to initialize the OpenCL context.
 * Calls the selected benchmarks.
 */
class ApplicationController {
private:
    std::shared_ptr<ComputeController> _controller = nullptr;
    std::vector<std::shared_ptr<benchmarks::BenchmarkBase>> _tests;

    std::vector<std::string> _runSpecificTests;

    bool _optimizeForSpeed = false;
    bool _disableOptimization = false;

    /**
     * Create a tests if it was selected by a specific argument when
     * launching the application or if all benchmarks should be executed.
     */
    template <typename TClass>
    void CreateTestInstance(const std::string& name);

    /**
     * Create all benchmarks and iterate over them to execute them.
     */
    void CreateAndExecuteTests();

public:
    explicit ApplicationController();
    virtual ~ApplicationController();

    /**
     * Parse application arguments and call other controllers
     * to execute the benchmark.
     * 
     * @return zero on success
     */
    int Run(int argc, char** argv);

    ApplicationController(const ApplicationController&) = delete;
    ApplicationController& operator=(const ApplicationController&) = delete;
    ApplicationController(const ApplicationController&&) = delete;
    ApplicationController& operator=(const ApplicationController&&) = delete;
};

#endif // __BENCH_CONTROLLER_HPP