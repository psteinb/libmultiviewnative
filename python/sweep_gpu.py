from subprocess import Popen, PIPE
import argparse
import sys

from generate_dims import produce_size_strings
from parse_nvprof import expected_cmd, extract_cuda_api_totals

def produce_header_from_app(_app):
    
    if not _app.count("-H"):
        _app += " -H"


    p = Popen(_app,
              stderr=PIPE,
              stdout=PIPE,
              shell=True)
    
    header_output = p.stdout.read().strip("\n")
    
    return header_output.split(" ")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=__file__,
                                     description="""
                                     call <bench_executable> given a generated set of arguments
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
                        default=11,
                        help="""
                        exponent to 2^<exponent> with which last stack size will be generated (default: %(default)s)
                        """)

    parser.add_argument('-b,--basis',
                        nargs="?",
                        dest='basis',
                        action='store',
                        type=int,
                        default=2,
                        help="""
                        basis to created stacks of (default: %(default)s)
                        """)

    parser.add_argument('--prof',
                        dest='profile',
                        action='store_true',
                        default=False,
                        help="""
                        run all processes within nvprof (must be available) 
                        and collect additional information
                        """)

    parser.add_argument('-v,--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help="""
                        print verbose messages
                        """)

    parser.add_argument('-n,--noheader',
                        dest='noheader',
                        action='store_true',
                        default=False,
                        help="""
                        do not print stats header
                        """)


    parser.add_argument('-l,--logfile',
                        dest='logfile',
                        action='store',
                        type=str,
                        default=None,
                        help="""
                        file to pipe output to instead of stdout
                        """)

    parser.add_argument('bench_util',
                        nargs=argparse.REMAINDER,
                        action='store',
                        type=str,
                        default="./bench_gpu_nd_fft",
                        help="""
                        benchmark utility, that has a cli complying to:
                        <util> -s <stack_size|generated> -t <optional> -a <optional>
                        (default: %(default)s)
                        """)

    # READING CL FLAGS
    args_parsed, left_over = parser.parse_known_args()

    if hasattr(args_parsed, "help"):
        parser.print_help()
        sys.exit(1)

    # PRODUCE stack DIMS
    if args_parsed.verbose:
        if left_over:
            print "[WARNING]\tcli options not recognized: ", left_over

        if args_parsed.verbose:
            print "producing data sizes from %i^%i to %i^%i" % (args_parsed.basis,
                                                                args_parsed.start_exponent,
                                                                args_parsed.basis,
                                                                args_parsed.end_exponent)
    size_str = produce_size_strings(args_parsed.start_exponent,
                                    args_parsed.end_exponent)

    nvprof_cmd = expected_cmd.split()
    api_calls_to_check = "cudaFree cudaMemcpy cudaMalloc".split()
    logfile = args_parsed.logfile

    bench_util_cmd = " ".join(args_parsed.bench_util)

    #extract the column names
    colnames = produce_header_from_app(bench_util_cmd)
    
    if args_parsed.profile:
        colnames.extend([str(item+"_perc "+item+"_ms") for 
                         item in api_calls_to_check])

    if not args_parsed.noheader:
        output = [" ".join(colnames)]

    modes = [" "]
    if bench_util_cmd.count("bench_gpu_nd_fft"):
        modes = ["-o -t -a", "-o -t", "-o ", "-t -a", "-t", "", "-g", "-g -o", "-g -t -a"]

    if bench_util_cmd.count("bench_cpu_nd_fft"):
        modes = ["-o ", "-g -a ", "-g -a -o "]

    if bench_util_cmd.count("bench_gpu_many_nd_fft"):
        to_check = "sync async async2plans mapped mangd".split(" ")
        flag = ["-t"]*len(to_check)
        modes = [ str("%s %s" % item) for item in zip(flag, to_check)]


    index = 0
    n_runs = len(modes)*len(size_str)
    for mode in modes:

        for size in size_str:
            if args_parsed.profile:
                cmd = nvprof_cmd
                cmd = cmd + [bench_util_cmd, mode, "-s", size]
            else:
                cmd = [bench_util_cmd, mode, "-s", size]

            cmd_to_give = " ".join(cmd)

            if args_parsed.verbose:
                print "%i/%i %s " % (index, n_runs,cmd_to_give)
                index += 1
            
            p = Popen(cmd_to_give,
                      stderr=PIPE,
                      stdout=PIPE,
                      shell=True)

            nvprof_output = p.stderr.read().split("\n")
            bench_output = p.stdout.read().strip("\n")
            
            if bench_output:
                if args_parsed.profile:
                    for k in api_calls_to_check:
                        res = extract_cuda_api_totals(nvprof_output, k)
                        if res and len(res) > 1:
                            bench_output += " %s %s" % res
                        else:
                            bench_output += " %s %s" % (str(0), str(0))
                output.append(bench_output)
    
    results = "\n".join(output)
    if not logfile:
        print results
    else:
        results += "\n"
        lfile = open(logfile,"a")
        lfile.writelines(results)
        lfile.close()
