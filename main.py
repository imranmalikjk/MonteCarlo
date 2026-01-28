
import json 
from BootsrapSE import get_bootstrap

import time 

def run(json_input):

    args = json.loads(json_input) if json_input else dict()

    beta0 = args.get('beta0',0)
    beta1 = args.get('beta1',2)
    sigma = args.get('sigma',1)
    N = args.get('N',10000)
    B = args.get('B',5000)
    useParallelism = args.get('useParallelism', False)

    start = time.time() 

    beta1_hat, se_boot, beta1_boot = get_bootstrap(
        beta0=beta0,
        beta1=beta1,
        sigma=sigma,
        N=N,
        B=B,
        useParallelism=useParallelism
    )

    print('Execution Time', round(time.time() - start,2),'secs')

    print("Beta1 predict: ", round(beta1_hat,2))
    print("Standard Error of Beta1 Bootstrap: ", se_boot,4)
    print("Beta1 Bootstrap", beta1_boot)

if __name__ == "__main__":
    run(
        json.dumps(
            {
                "beta0": 0,
                "beta1": 2,
                "sigma": 1,
                "N": 5000,
                "B": 1000,
                "useParallelism":True
            }
        )
    )