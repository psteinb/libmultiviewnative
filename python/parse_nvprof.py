"""
python module that parses the output of a call to
$ nvprof --normalized-time-unit ms --print-gpu-summary --print-api-summary --print-summary --profile-from-start off 
"""

expected_cmd = "nvprof --normalized-time-unit ms --print-gpu-summary --print-api-summary --print-summary --profile-from-start off"

def extract_cuda_api_stats(_lines, _api_str):
    """ 
    search the output for _api_str and return total, min, avg, max
    this method expects a _lines to contain lines as in
    Time(%)      Time     Calls       Avg       Min       Max  Name
    55.39%  58.524ms        80  731.55us  8.2520us  57.452ms  cudaLaunch
    14.42%  15.233ms       100  152.33us     588ns  661.61us  cudaFree
    """
    value = None
    try:
        api_tag = next(li for li in _lines if li.count("API calls"))
    except:
        return value

    lines = _lines[_lines.index(api_tag):]
    for line in lines:
        if line.split()[-1].count(_api_str):
            columns = line.strip("ms").split()
            if len(columns) > 5:
                if value:
                    new = (float(columns[1]), 
                             float(columns[3]), 
                             float(columns[4]), 
                             float(columns[5]))
                    for n in range(len(value)):
                        value[n] += new[n]
                else:
                    value = (float(columns[1]), 
                             float(columns[3]), 
                             float(columns[4]), 
                             float(columns[5]))
            else:
            
                value = None
            break
                
    return value

def extract_cuda_api_totals(_lines, _api_str):
    """ 
    search the output for _api_str and return total, min, avg, max
    this method expects a _lines to contain lines as in
    Time(%)      Time     Calls       Avg       Min       Max  Name
    55.39%  58.524ms        80  731.55us  8.2520us  57.452ms  cudaLaunch
    14.42%  15.233ms       100  152.33us     588ns  661.61us  cudaFree
    """
    value = None
    try:
        api_tag = next(li for li in _lines if li.count("API calls"))
    except:
        return value

    lines = _lines[_lines.index(api_tag):]
    for line in lines:
        if line.split()[-1].count(_api_str):
            columns = line.strip("ms").strip("%").split()
            if len(columns) > 5:
                if value:
                    new = (float(columns[0]), 
                             float(columns[1]))
                    for n in range(len(value)):
                        value[n] += new[n]
                else:
                    value = (float(columns[0]), 
                             float(columns[1]))
            else:
            
                value = None
            break
                
    return value

def approximate_runtime(_lines):
    """ 
    search the output for lines after "API calls" and return sum of the Time column
    """
    value = None
    api_tag = next(l for l in _lines if l.count("API calls"))
    lines = _lines[_lines.index(api_tag):]
    for line in lines:
        value += float(line.strip("ms").split()[1])
                
    return value
    

