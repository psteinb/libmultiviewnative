from subprocess import Popen, PIPE, STDOUT
import argparse
import sys

from generate_dims import produce_size_strings
from parse_nvprof import expected_cmd, extract_cuda_api_stats

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=__file__,
                                     description="""
                                     call <bench_executable> given a generated set of arguments
                                     """)

    parser.add_argument('bench_util',
                        nargs="?",
                        action='store',
                        type=str,
                        default="./bench_gpu_nd_fft",
                        help="""
                        benchmark utility, that has a cli complying to:
                        <util> -s <stack_size|generated> -t <optional> -a <optional>
                        (default: %(default)s)
                        """)

    parser.add_argument('-s,--start_exponent',
                        nargs="?",
                        dest='start_exponent',
                        action='store',
                        type=int,
                        default=6,
                        help="""
                        exponent to 2^<exponent> with which first stack size will be generated (default: %(default)s)
                        """)

    parser.add_argument('-e,--end_exponent',
                        nargs="?",
                        dest='end_exponent',
                        action='store',
                        type=int,
                        default=10,
                        help="""
                        exponent to 2^<exponent> with which last stack size will be generated (default: %(default)s)
                        """)

    # READING CL FLAGS
    args_parsed = parser.parse_args()

    if hasattr(args_parsed, "help"):
        parser.print_help()
        sys.exit(1)
    # else:
        # print __file__, " configuration received :"
        # max_len = max([ len(str(word)) for word in args_parsed.__dict__.values() ])
        # for k, v in args_parsed.__dict__.iteritems():
        #     print str("\t%15s %"+str(max_len+5)+"s") % (k, v)

    # PRODUCE stack DIMS
    size_str = produce_size_strings(args_parsed.start_exponent,
                                    args_parsed.end_exponent)

    modes = ["-t -a", "-a", ""]
    nvprof_cmd = expected_cmd.split()
    api_calls_to_check = "cudaFree cudaMemcpy cudaMalloc".split()

    colnames = ["gpu", "alloc", "tx", "repeats", "total_time_ms",
                "shape", "data_in_mb", "exp_gmem_in_mb"]

    colnames.extend([str(item+"_ms") for item in api_calls_to_check])

    output = ["\t".join(colnames)]
    for mode in modes:

        for size in size_str:
            cmd = nvprof_cmd + [args_parsed.bench_util, mode, "-s", size]
            cmd_to_give = " ".join(cmd)
            
            p = Popen(cmd_to_give,
                      stderr=PIPE,
                      stdout=PIPE,
                      shell=True)
            nvprof_output = p.stderr.read().split("\n")
            bench_output = p.stdout.read().strip("\n")
            for k in api_calls_to_check:
                res = extract_cuda_api_stats(nvprof_output, k)
                if res and len(res) > 1:
                    bench_output += "\t%s" % res[0]
                
            output.append(bench_output)

    print "\n".join(output)
