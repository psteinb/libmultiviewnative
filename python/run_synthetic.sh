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
TAG=`echo $HOSTNAME|egrep -o "^[^\.]+"`
else
TAG=$HOSTNAME
fi

NCORES=`grep "core id" /proc/cpuinfo |wc -l`

python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_deconvolve_synthetic -c "1x$CPU_ID" >  ${TAG}_cpu_deconvolve_synthetic.data
python $RUNPATH/../../python/sweep_gpu.py $RUNPATH/bench_cpu_deconvolve_synthetic -t ${NCORES} -c "${NCORES}x${CPU_ID}" |grep -v comment >>  ${TAG}_cpu_deconvolve_synthetic.data

python $RUNPATH/../../python/sweep_gpu.py --prof $RUNPATH/bench_gpu_deconvolve_synthetic > ${TAG}_gpu_deconvolve_synthetic_prof.data

cat ${TAG}_gpu_deconvolve_synthetic_prof.data | cut -f1-10 -d' ' > ${TAG}_gpu_deconvolve_synthetic.data
