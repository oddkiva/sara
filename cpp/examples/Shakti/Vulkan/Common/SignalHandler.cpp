#pragma once

#include "SignalHandler.hpp"


bool SignalHandler::initialized = false;
std::atomic_bool SignalHandler::ctrl_c_hit = false;
#if !defined(_WIN32)
struct sigaction SignalHandler::sigint_handler = {};
#endif
