#/bin/bash

if [[ -z $1 ]]; then
    RUNPATH=$PWD
else
    RUNPATH=$1
fi

if [[ -n $2 ]];then
    CPU_ID="-c $2"
else
    CPU_ID="-c "`egrep "^model name" /proc/cpuinfo |sed -e 's/.*\([Ei].*\) \@.*/\1/'|tr -d ' '`
fi

TAG=`echo $HOSTNAME|sed -e 's/\(.*\)\..*/\1/'`

python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_nd_fft $CPU_ID >>  ${TAG}_cpu.data
python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_many_nd_fft $CPU_ID >>  ${TAG}_cpu_many_fft.data

python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_nd_fft -t 24 $CPU_ID >>  ${TAG}_cpu.data
python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_many_nd_fft -t 24 $CPU_ID >>  ${TAG}_cpu_many_fft.data

python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_gpu_nd_fft >> ${TAG}_gpu.data
python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_gpu_many_nd_fft >> ${TAG}_gpu_many_fft.data
