nsys profile --wait=primary --delay=3 --duration=10 --cuda-graph-trace=node --stats=true -o nprof-eval-bias python -m pufferlib.pufferl train puffer_breakout
#nsys stats --report cuda_gpu_kern_sum:base nprof.nsys-rep
#nsys profile --wait=primary --delay=3 --duration=10 --cuda-graph-trace=graph --stats=true -o nprof-nograph python -m pufferlib.pufferl train puffer_breakout
#nsys profile --delay=3 --wait=primary --stats=true -o my_report python -m pufferlib.pufferl train puffer_breakout
