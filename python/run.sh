#/bin/bash

if [[ -z $1 ]]; then
    RUNPATH=$PWD
else
    RUNPATH=$1
fi



python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_nd_fft >> ${HOSTNAME}_cpu.data
python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_many_nd_fft >> ${HOSTNAME}_cpu.data
python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_nd_fft -t 24 >> ${HOSTNAME}_cpu.data
python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_many_nd_fft -t 24 >> ${HOSTNAME}_cpu.data

python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_gpu_nd_fft >> ${HOSTNAME}_gpu.data
python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_gpu_many_nd_fft >> ${HOSTNAME}_gpu.data
