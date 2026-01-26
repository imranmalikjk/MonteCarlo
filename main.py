
import json 
from BootsrapSE import get_bootstrap

def run(json_input):

    args = json.loads(json_input) if json_input else dict()

    beta0 = args.get('beta0',0)
    beta1 = args.get('beta1',2)
    sigma = args.get('sigma',1)
    N = args.get('N',10000)
    B = args.get('B',1000)
    useParallelism = args.get('useParallelism', False)

    get_bootstrap(
        beta0=beta0,
        beta1=beta1,
        sigma=sigma,
        N=N,
        B=B,
        useParallelism=useParallelism
    )

if __name__ == "__main__":
    run(
        json.dumps(
            {
                "beta0": 0,
                "beta1": 2,
                "sigma": 1,
                "N": 10000,
                "B": 1000,
                "useParallelism":False
            }
        )
    )