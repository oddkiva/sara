/**
 *  @file timer.cpp
 *  @brief Timer class implementation
 *  @author Rodolfo Lima
 *  @date January, 2011
 */

//== INCLUDES =================================================================

#include <cmath>
#include <stack>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <ctime>
#include <sys/time.h>
#endif

#include <error.h>
#include <timer.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

timer_pool timers;

size_t g_start_count = 0;

base_timer::base_timer(const char *type_label, size_t data_size,
                       const std::string &unit)
    : m_type_label(type_label)
    , m_elapsed(0)
    , m_started(false)
    , m_data_size(data_size)
    , m_unit(unit)
{
}

void base_timer::start()
{
    if(m_started)
        stop();

    do_start();

    m_elapsed = 0;
    m_started = true;
    ++g_start_count;
}

void base_timer::stop()
{
    if(!m_started)
        return;

    do_stop();

    m_started = false;
    --g_start_count;
}

double base_timer::elapsed()
{ 
    if(m_elapsed == 0)
    {
        stop();
        m_elapsed = do_get_elapsed();
    }

    return m_elapsed; 
}


gpu_timer::gpu_timer(size_t data_size, const std::string &unit, bool start)
    : base_timer("[GPU]", data_size, unit)
    , m_start(NULL), m_stop(NULL)
{
    if(start)
        this->start();
}

gpu_timer::~gpu_timer()
{
    if(m_stop)
    {
        cudaEventDestroy(m_stop);
        check_cuda_error("Timer event destruction");
    }
    if(m_start)
    {
        cudaEventDestroy(m_start);
        check_cuda_error("Timer event destruction");
    }
}

void gpu_timer::do_start()
{
    if(m_start == NULL)
    {
        cudaEventCreate(&m_start);
        check_cuda_error("Timer event creation");
    }
    if(m_stop == NULL)
    {
        cudaEventCreate(&m_stop);
        check_cuda_error("Timer event creation");
    }


    cudaEventRecord(m_start, 0);
    check_cuda_error("Event recording");
}

void gpu_timer::do_stop()
{
    cudaEventRecord(m_stop, 0);
    check_cuda_error("Event recording");
}

double gpu_timer::do_get_elapsed() const
{ 
    float elapsed;

    cudaEventSynchronize(m_stop);
    check_cuda_error("Event synchronize");
    cudaEventElapsedTime(&elapsed, m_start, m_stop);
    check_cuda_error("Event elapsed time");

    return elapsed/1000.f;
}

cpu_timer::cpu_timer(size_t data_size, const std::string &unit, bool start)
    : base_timer("[CPU]", data_size, unit)
    , m_start_time(0), m_stop_time(0)
{
    if(start)
        this->start();
}

double cpu_timer::get_cpu_time() const
{
#ifdef _WIN32
    LARGE_INTEGER counter, freq;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&freq);
    return (1.0*counter.QuadPart)/(1.0*freq.QuadPart);
#else
    struct timeval v;
    gettimeofday(&v, (struct timezone *) NULL);
    return v.tv_sec + v.tv_usec/1.0e6;
#endif
}

void cpu_timer::do_start()
{
    m_start_time = get_cpu_time();
}

void cpu_timer::do_stop()
{
    m_stop_time = get_cpu_time();
}

double cpu_timer::do_get_elapsed() const
{
    return m_stop_time - m_start_time;
}

scoped_timer_stop::scoped_timer_stop(base_timer &timer) 
    : m_timer(&timer)
{
}

gpu_timer &timer_pool::gpu_add(const std::string &label, size_t data_size,
                               const std::string &unit)
{
    gpu_timer *timer = new gpu_timer(data_size, unit, false);

    timer_data data;
    data.timer = timer;
    data.label = label;
    data.level = g_start_count;

    m_timers.push_back(data);

    timer->start();
    return *timer;
}

cpu_timer &timer_pool::cpu_add(const std::string &label, size_t data_size,
                               const std::string &unit)
{
    cpu_timer *timer = new cpu_timer(data_size, unit, false);

    timer_data data;
    data.timer = timer;
    data.label = label;
    data.level = g_start_count;

    m_timers.push_back(data);

    timer->start();
    return *timer;
}

std::string unit_value(double v, double base)
{
    const char *units[] = {"","k","M","G","T",NULL};

    const char **unit;
    for(unit = units; unit; ++unit)
    {
        if(std::abs(v) < base)
            break;

        v /= base;
    }

    if(*unit == NULL)
    {
        --unit;
        v *= base;
    }

    std::stringstream ss;
    ss << v << " " << *unit;
    return ss.str();
}

void timer_pool::flush()
{
    std::stack<float> parent_totals;

    for(timer_list::iterator it=m_timers.begin(); it!=m_timers.end(); ++it)
    {
        std::string padding(it->level*2,' ');
        std::cout << it->timer->type_label() << padding;

        while(parent_totals.size() > it->level)
            parent_totals.pop();

        std::cout << std::setw(4)
                  << (int)(parent_totals.empty() ? 100 : it->timer->elapsed()/parent_totals.top()*100)
                  << "% - " << std::flush;

        std::cout << it->label << ": ";
        if(!it->timer->is_stopped())
            std::cout << "FORCED STOP - ";
        std::cout << std::scientific
                  << it->timer->elapsed() << " s"
                  << std::resetiosflags( std::ios_base::floatfield );

        if(it->timer->data_size() != 0)
        {
            double base;
            if(!it->timer->unit().empty() && it->timer->unit()[0] == 'i')
                base = 1024;
            else
                base = 1000;

            std::cout << " - " 
                      << unit_value(it->timer->data_size()/it->timer->elapsed(),
                                    base) 
                      << it->timer->unit() << "/s";
        }
        std::cout << std::endl;

        parent_totals.push(it->timer->elapsed());

        delete it->timer;
    }

    m_timers.clear();
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
//vi: ai sw=4 ts=4
