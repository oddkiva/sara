#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void *initialize_graphics_application();
// void *initialize_graphics_application(int *argc, char **argv);

void deinitialize_graphics_application(void *);

void register_user_main(void *, void (*user_main)(void));

void exec_graphics_application(void *);

#ifdef __cplusplus
}
#endif
