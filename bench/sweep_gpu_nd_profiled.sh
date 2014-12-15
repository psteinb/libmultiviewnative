

if [[ -z $1 ]];then
    echo "usage : sweep_gpu_nd.sh <path_to generate_sizes.py> <path_to bench_gpu_nd_fft>"
    echo "        will produce 2 files: bench_gpu_3d_fft.log "
else

SIZES=""
if [[ -e $1 ]];then
SIZES=`python $1 11`
else
    echo "unable to sweep, generate_sizes.py not found at $1"
fi

BASENAME=`basename $2|sed -e 's/nd/3d/'`
echo "-> generating ${BASENAME}.log"

for i in $SIZES;
do echo $i;
    nvprof --print-gpu-summary --print-api-summary --print-summary --profile-from-start off $2 -s $i -t -a -r 20 >> ${BASENAME}.log 2>&1
done

for i in $SIZES;
do echo $i;
    nvprof --print-gpu-summary --print-api-summary --print-summary --profile-from-start off $2 -s $i -a -r 20 >> ${BASENAME}.log 2>&1
done

for i in $SIZES;
do echo $i;
    nvprof --print-gpu-summary --print-api-summary --print-summary --profile-from-start off $2 -s $i -r 20 >> ${BASENAME}.log 2>&1
done

fi
