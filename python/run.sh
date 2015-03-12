#/bin/bash

if [[ -z $1 ]]; then
    RUNPATH=$PWD
else
    RUNPATH=$1
fi

if [[ -n $2 ]];then
    CPU_ID="$2"
else
    CPU_ID=`egrep "^model name" /proc/cpuinfo |sed -e 's/.*\([Ei].*\) \@.*/\1/'|tr -d ' '|sort -u`
fi

if [[ $HOSTNAME == *.* ]];then
TAG=`echo $HOSTNAME|egrep -o "^[a-Z0-9]+\."|tr -d '.'`
else
TAG=$HOSTNAME
fi

NCORES=`grep "core id" /proc/cpuinfo |wc -l`

python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_nd_fft -c "1x$CPU_ID" >>  ${TAG}_cpu.data
python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_many_nd_fft -c "1x$CPU_ID" >>  ${TAG}_cpu_many_fft.data

python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_nd_fft -t ${NCORES} -c "${NCORES}x${CPU_ID}" >>  ${TAG}_cpu.data
python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_many_nd_fft -t ${NCORES} -c "${NCORES}x${CPU_ID}" >>  ${TAG}_cpu_many_fft.data

python $RUNPATH/../../python/sweep_gpu.py --prof $RUNPATH/bench_gpu_nd_fft >> ${TAG}_gpu_prof.data
python $RUNPATH/../../python/sweep_gpu.py --prof $RUNPATH/bench_gpu_many_nd_fft >> ${TAG}_gpu_many_fft_prof.data

cat ${TAG}_gpu_prof.data | cut -f1-10 -d' ' > ${TAG}_gpu.data
cat ${TAG}_gpu_many_fft_prof.data | cut -f1-10 -d' ' > ${TAG}_gpu_many_fft.data
