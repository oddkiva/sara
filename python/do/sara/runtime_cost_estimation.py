# Latency: time needed to execute a CPU instruction (add, sub, mul, div...)
# Throughput: wait time before repeating a CPU instruction.

ns = 1e-9
us = 1e-6
ms = 1e-3

mhz = 1e6
ghz = 1e9


def cycle_time(frequency = 1 * ghz):
    return 1 / frequency;

cpu_freq = 1 * ghz
cpu_cycle = cycle_time(cpu_freq)
cpu_cores = 4

# GeForce GTX 1080 Ti
gpu_freq = 1481 * mhz
gpu_cycle = cycle_time(gpu_freq)
gpu_cores = 3580

# cf: https://www.mjr19.org.uk/lect1.pdf
cpu_float_arithmetics_op = {
    'load/store': [2 * cpu_cycle, 4 * cpu_cycle],
    '+': 4 * cpu_cycle,
    '-': 4 * cpu_cycle,
    '*': 4 * cpu_cycle,
    '/': [30 * cpu_cycle, 50 * cpu_cycle],
    'sqrt': [30 * cpu_cycle, 50 * cpu_cycle]
}

gpu_float_arithmetics_op = {
    'load/store': [2 * cpu_cycle, 4 * cpu_cycle],
    '+': 4 * gpu_cycle,
    '-': 4 * gpu_cycle,
    '*': 32 * gpu_cycle,
    '/': [128 * gpu_cycle],
    'sqrt': [30 * cpu_cycle, 50 * cpu_cycle]
}

cpu_central_gradient_op = cpu_float_arithmetics_op['+'] + \
    cpu_float_arithmetics_op['*']
gpu_central_gradient_op = gpu_float_arithmetics_op['+'] + \
    gpu_float_arithmetics_op['*']

cpu_image_gradient_op = 1920 * 1080 * cpu_central_gradient_op / cpu_cores
print('CPU image gradient (ms) =', cpu_image_gradient_op / ms)  # 4 ms

gpu_image_gradient_op = 1920 * 1080 * gpu_central_gradient_op / gpu_cores
print('GPU image gradient (ms) =', gpu_image_gradient_op / ms)  # 10 us

accel_factor = cpu_image_gradient_op / gpu_image_gradient_op
print('GPU accel factor =', accel_factor)

game_fps = 60
game_frame = 1 / game_fps
print('Frame rendering max time (ms) = ', game_frame / ms)  # 16 ms
import IPython; IPython.embed()
