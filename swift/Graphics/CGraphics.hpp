#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void* GraphicsApplication_initialize();
// void *GraphicsApplication_initialize(int *argc, char **argv);
void GraphicsApplication_registerUserMainFunc(void*, void (*user_main)(void));
void GraphicsApplication_exec(void*);

void* createWindow(int w, int h);
void closeWindow(void *window);
void millisleep(int ms);
int getKey();

void drawPoint(int x, int y, int r, int g, int b);

#ifdef __cplusplus
}
#endif
