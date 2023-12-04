#pragma once

#include <signal.h>

#include <atomic>
#include <iostream>


struct SignalHandler
{
  static bool initialized;
  static std::atomic_bool ctrl_c_hit;
  static struct sigaction sigint_handler;

  static auto stop_render_loop(int) -> void
  {
    std::cout << "[CTRL+C HIT] Preparing to close the program!" << std::endl;
    ctrl_c_hit = true;
  }

  static auto init() -> void
  {
    if (initialized)
      return;

#if defined(_WIN32)
    signal(SIGINT, stop_render_loop);
    signal(SIGTERM, stop_render_loop);
    signal(SIGABRT, stop_render_loop);
#else
    sigint_handler.sa_handler = SignalHandler::stop_render_loop;
    sigemptyset(&sigint_handler.sa_mask);
    sigint_handler.sa_flags = 0;
    sigaction(SIGINT, &sigint_handler, nullptr);
#endif

    initialized = true;
  }
};
