/**
 *  @file timer.h
 *  @brief Timer utility class definition
 *  @author Rodolfo Lima
 *  @date January, 2011
 */

#ifndef TIMER_H
#define TIMER_H

//== INCLUDES =================================================================

#include <string>
#include <list>
#include <cuda_runtime.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== CLASS DEFINITION =========================================================

/**
 *  @class base_timer timer.h
 *  @ingroup utils
 *  @brief Base timer class
 *
 *  Base timer is the abstract class for timing computation.
 */
class base_timer {

public:

    /**
     *  Constructor
     *  @param[in] type_label The label string for this timer type
     *  @param[in] data_size Data size associated with this timer
     *  @param[in] unit The unit of the associated data
     */
    base_timer( const char *type_label,
                size_t data_size = 0,
                const std::string &unit = "" );

    /**
     *  @brief Start counting timing
     */
    void start();

    /**
     *  @brief Stop counting timing
     */
    void stop();

    /**
     *  @brief Time elapsed
     *  @return Elapsed time in seconds
     */
    double elapsed();

    /**
     *  @brief Check if this timer is not counting timing
     *  @return True if this timer is not counting timing
     */
    bool is_stopped() const { return !m_started; }

    /**
     *  @brief Data size associated with this timer
     *  @return The data size associated with this timer
     */
    size_t data_size() const { return m_data_size; }

    /**
     *  @brief Unit of the data associated with this timer
     *  @return The unit of the associated data
     */
    const std::string &unit() const { return m_unit; }

    /**
     *  @brief Type of this timer
     *  @return The label string for this timer type
     */
    const char *type_label() { return m_type_label; }

protected:

    /**
     *  @brief Do start this timer
     */
    virtual void do_start() = 0;

    /**
     *  @brief Do stop this timer
     */
    virtual void do_stop() = 0;

    /**
     *  @brief Do get time elapsed by this timer
     *  @return Time elapsed in seconds
     */
    virtual double do_get_elapsed() const = 0;

private:

    /**
     *  Copy Constructor (deleted)
     *  @brief Prevent the assignment declaration by derived classes
     *  @param[in] bt Object to copy to this object
     */
    base_timer( const base_timer& bt );

    /**
     *  @brief Assign operator (deleted)
     *  @brief Prevent the assignment declaration by derived classes
     *  @param[in] bt Object to copy from
     *  @return This timer with assigned values
     */
    base_timer& operator = ( const base_timer& bt );

    const char *m_type_label; ///< Label for this timer type

    double m_elapsed; ///< Elapsed time

    bool m_started; ///< Flag true if this timer is counting timing

    size_t m_data_size; ///< Data size associated with this timer

    std::string m_unit; ///< Unit of the associated data

};

//== CLASS DEFINITION =========================================================

/**
 *  @class gpu_timer timer.h
 *  @ingroup utils
 *  @brief GPU specialization of the timer class
 *
 *  GPU timer is the specialized class for GPU timing computation.
 */
class gpu_timer : public base_timer {

public:

    /**
     *  Constructor
     *  @param[in] data_size Data size associated with this timer
     *  @param[in] unit The unit of the associated data
     *  @param[in] start Start this timer promptly flag
     */
    gpu_timer( size_t data_size = 0,
               const std::string& unit = "",
               bool start = true );

    /**
     *  Destructor
     */
    ~gpu_timer();

private:

    /**
     *  @brief Do start this timer
     */
    virtual void do_start();

    /**
     *  @brief Do stop this timer
     */
    virtual void do_stop();

    /**
     *  @brief Do get time elapsed by this timer
     *  @return Time elapsed in seconds
     */
    virtual double do_get_elapsed() const;

    cudaEvent_t m_start; ///< Start CUDA events for timing computation
    cudaEvent_t m_stop; ///< Stop CUDA events for timing computation

};

//== CLASS DEFINITION =========================================================

/**
 *  @class cpu_timer timer.h
 *  @ingroup utils
 *  @brief CPU specialization of the timer class
 *
 *  CPU timer is the specialized class for CPU timing computation.
 */
class cpu_timer : public base_timer {

public:

    /**
     *  Constructor
     *  @param[in] data_size Data size associated with this timer
     *  @param[in] unit The unit of the associated data
     *  @param[in] start Start this timer promptly flag
     */
    cpu_timer( size_t data_size = 0,
               const std::string& unit = "",
               bool start = true );

    /**
     *  Destructor
     */
    ~cpu_timer();

private:

    /**
     *  @brief Do start this timer
     */
    virtual void do_start();

    /**
     *  @brief Do stop this timer
     */
    virtual void do_stop();

    /**
     *  @brief Do get time elapsed by this timer
     *  @return Time elapsed in seconds
     */
    virtual double do_get_elapsed() const;

    /**
     *  @brief Get the current CPU time
     *  @return Current time in the CPU in seconds
     */
    double get_cpu_time() const;

    double m_start_time; ///< Start CPU timings
    double m_stop_time; ///< Stop CPU timings

};

//== CLASS DEFINITION =========================================================

/**
 *  @class scoped_timer_stop timer.h
 *  @ingroup utils
 *  @brief Scope-limited timer class
 *
 *  Scoped timer stop is a timer class that stops counting timing when
 *  destroyed, i.e. at the end of its scope.
 */
class scoped_timer_stop {

public:

    /**
     *  Constructor
     *  @param[in] timer Base timer associated with this scope-limited timer
     */
    scoped_timer_stop( base_timer& timer );

    /**
     *  Destructor
     */
    ~scoped_timer_stop() { stop(); }

    /**
     *  @brief Stop counting timing
     */
    void stop() { m_timer->stop(); }

    /**
     *  @brief Time elapsed
     *  @return Elapsed time in seconds
     */
    double elapsed() const { return m_timer->elapsed(); }

private:

    base_timer *m_timer; ///< Base timer associated with this scope-limited timer

};

//== CLASS DEFINITION =========================================================

/**
 *  @class timer_pool timer.h
 *  @ingroup utils
 *  @brief Pool of timers
 *
 *  Pool of timers allows a number of timers to be composed and
 *  flushed to the standard output showing relative timing percentages
 *  for comparison.
 *
 */
class timer_pool {

public:

    /**
     *  Destructor
     */
    ~timer_pool() { }

    /**
     *  @brief Add a GPU timer to this pool
     *  @param[in] label The label string for this timer
     *  @param[in] data_size Data size associated with this timer
     *  @param[in] unit The unit of the associated data
     *  @return The added GPU timer
     */
    gpu_timer &gpu_add( const std::string& label,
                        size_t data_size = 0,
                        const std::string& unit = "" );

    /**
     *  @brief Add a CPU timer to this pool
     *  @param[in] label The label string for this timer
     *  @param[in] data_size Data size associated with this timer
     *  @param[in] unit The unit of the associated data
     *  @return The added CPU timer
     */
    cpu_timer &cpu_add( const std::string& label,
                        size_t data_size = 0,
                        const std::string& unit = "" );

    /**
     *  @brief Flush to standard output all timers and their relative percentages
     */
    void flush();

private:

    /**
     *  @struct timer_data
     *  @brief Each timer data in the pool of timers
     */
    struct timer_data {
        base_timer *timer; ///< Pointer to base timer
        std::string label; ///< Timer label
        int level; ///< Stack level when flushing timers
    };

    typedef std::list<timer_data> timer_list; ///< timer list definition

    timer_list m_timers; ///< Pool of timers

};

//== EXTERNS ==================================================================

/**
 *  @var timers
 *  @ingroup utils
 *  @brief Global pool of timers
 */
extern
timer_pool timers;

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // TIMER_H
//=============================================================================
//vi: ai sw=4 ts=4
