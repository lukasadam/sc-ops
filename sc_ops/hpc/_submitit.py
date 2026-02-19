import submitit
from functools import partial

def create_executor(folder="submitit_logs", cpus_per_task=1, timeout_min=23*60, mem_gb=200, qos="1d", slurm_partition="batch_cpu"):
    """Create a submitit executor with specified parameters.
    
    Parameters
    ----------
    folder : str, optional
        The folder where submitit logs will be stored, by default "submitit_logs".
    cpus_per_task : int, optional
        The number of CPUs to allocate per task, by default 1.
    timeout_min : int, optional
        The timeout for the job in minutes, by default 23 hours (23*60).
    mem_gb : int, optional
        The amount of memory to allocate in GB, by default 200.
    qos : str, optional
        The quality of service for the job, by default "1d".
    slurm_partition : str, optional
        The SLURM partition to submit the job to, by default "batch_cpu".
    """
    executor = submitit.AutoExecutor(folder=folder)
    executor.update_parameters(
        cpus_per_task=cpus_per_task,       
        timeout_min=timeout_min,  
        mem_gb=mem_gb,             
        qos=qos,
        slurm_partition=slurm_partition 
    )
    return executor