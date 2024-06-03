## UMF

This repository maintains the implementation of an updated adaptive matrix factorization approach to online QoS prediction of cloud services. 

### Dependencies
- Python 2.7 (https://www.python.org)
- Cython 0.20.1 (http://cython.org)
- numpy 1.8.1 (http://www.scipy.org)
- scipy 0.13.3 (http://www.scipy.org)

### Usage

The UMF algorithm is implemented in C++ and further wrapped up as a python package for common use.

1. Install `UMF` package
  
   Download the repo then install the package `python setup.py install --user`.    

2. Change directory `cd` to `"benchmarks/"`, and configure the parameters in benchmark scripts
  
   `run_rt.py` and `run_tp.py` to run UMF with response time and throughput 
   Some common parameters:
    `'parallelMode': True` if you are running a multi-core machine.
    `'rounds': 1` for testing, which can make the execution finish soon.

3. Run the benchmark scripts
     
   ```    
     $ python run_rt.py
     $ python run_tp.py 
   ```
    
4. Check the evaluation results in `"benchmarks/result/"` directory.


### License
The UMF is based on AMF repo with MIT license.
[The MIT License (MIT)](./LICENSE)

Copyright &copy; 2017, [WS-DREAM](https://wsdream.github.io), CUHK

