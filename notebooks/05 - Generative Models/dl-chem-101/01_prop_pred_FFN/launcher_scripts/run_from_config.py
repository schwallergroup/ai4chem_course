""" Pythonic launcher

Use combinations config file to launch trials

python launcher_scripts/run_from_config.py configs/[config_name].yaml

To use this script, the config file should be structured using a set of
launcher args, universal args, and then an iterable iterative args that will
override the universal args.

Within these, there should be arguments to the python file mapping to a list of
values to iterate over.  The product of all these entries will be computed and
directly fed as arguments in different program calls. Any argument that should
be interpreted by the parser differently should be indicated with an underscore.

Any argument beginning with an underscore is manipulated by this launcher file
and transformed before being fed into the corresponding program.

    launcher_args: {experiment_name : str,
                    slurm_script : launcher_scripts/generic_slurm.sh
                    use_slurm : bool}
    universal_args:
        program-arg-1 : [combo_1, combo_2]
        program-arg-2: [combo_1, combo_2, combo_3]
        _slurm_args : [{time: time, _num_gpu : 1, job-name}]
    iterative_args:
        - universal_arg_replacement_1: [new_val1, new_val2]
          universal_arg_replacement_2: [new_val1, new_val2]
        - run_2_replacement : [new_val3, new_val4]
"""

from pathlib import Path
import shutil
import hashlib
import subprocess
import copy
import time
import itertools
from datetime import datetime
import argparse
import yaml

PYTHON_NAME = "python3"


def md5(key: str) -> str:
    """md5.

    Args:
        key (str): string to be hasehd
    Returns:
        Hashed encoding of str
    """
    return hashlib.md5(key.encode()).hexdigest()


def get_args():
    """parse yaml config"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Name of configuration file")
    args = parser.parse_args()
    print(f"Loading experiment from: {args.config_file}\n")
    args_new = yaml.safe_load(open(args.config_file, "r"))
    return args_new, args.config_file


def dump_config_file(save_dir: str, config: str):
    """dump_config_file.

    Try to dump the output config file continuously. If it doesn't work,
    increment it.

    Args:
        save_dir (str): Name of the save dir where to put this
        config (str): Location of the config file
    """

    # Dump experiment
    new_file = "experiment.yaml"
    save_dir = Path(save_dir)
    config_path = save_dir / new_file
    ctr = 1
    save_dir.mkdir(exist_ok=True)

    # Keep incrementing the counter
    while config_path.exists():
        new_file = f"experiment_{ctr}.yaml"
        config_path = save_dir / new_file
        ctr += 1

    shutil.copy2(config, config_path)
    time_stamp_date = datetime.now().strftime("%m_%d_%y")

    # Open timestamps file
    with open(save_dir / "timestamps.txt", "a") as fp:
        fp.write(f"Experiment {new_file} run on {time_stamp_date}.\n")


def build_python_string(
    experiment_folder: str,
    experiment_name,
    arg_dict: dict,
    launcher_args: dict,
    script: str = "run_training.py",
):
    """build_python_string."""

    python_string = f"{PYTHON_NAME} {script}"

    # Define OUT
    time_stamp_seconds = datetime.now().strftime("%Y_%m_%d-%H%M_%f")

    sub_dir = experiment_folder
    args_hash = md5(str(arg_dict))

    time_and_hash = f"{time_stamp_seconds}_{args_hash}"

    general_flags = "" 
    for arg_name, arg_value in arg_dict.items():
        # Handle specific arguments here
        if arg_name == "_slurm_args":
            slurm_args = construct_slurm_args(experiment_name, arg_value)
        elif arg_name == "_model":
            python_string = f"{python_string} {arg_value}".strip()
        elif arg_name == "_use-save-dir":
            pass
        else:
            new_flag = convert_flag(arg_name, arg_value)
            general_flags = f"{general_flags} {new_flag}".strip()
    python_string = f"{python_string} {general_flags}"

    if "save-dir" in arg_dict.keys():
        out = arg_dict["save-dir"]
    else:
        out = Path(sub_dir) / time_and_hash

    outdir = convert_flag("save-dir", out)
    python_string = f"{python_string} {outdir}".strip()

    return (slurm_args, python_string)


def construct_slurm_args(experiment_name: str, slurm_args: dict):
    """construct_slurm_args."""

    # Slurm args
    sbatch_args = f"--output=logs/{experiment_name}_%j.log"
    for k, v in slurm_args.items():
        if k == "_num_gpu":
            # Specify node number
            if v > 0:
                sbatch_args = f"{sbatch_args} --gres=gpu:{v}"
        elif k == "node":
            sbatch_args = f"{sbatch_args} -w {v}"

        else:
            new_flag = convert_flag(k, v)
            sbatch_args = f"{sbatch_args} {new_flag}".strip()
    return sbatch_args


def convert_flag(flag_key, flag_value):
    """Convert the actual key value pair into a python flag"""
    if isinstance(flag_value, bool):
        return_string = f"--{flag_key}" if flag_value else ""
    elif isinstance(flag_value, list):
        flag_value = [str(i) for i in flag_value]
        return_string = f"--{flag_key} {' '.join(flag_value)}"
    elif flag_value is None:
        return_string = ""
    elif isinstance(flag_value, str):
        return_string = f"--{flag_key} '{flag_value}'"
    else:
        return_string = f"--{flag_key} {flag_value}"
    return return_string


def get_launcher_log_name(experiment_folder):
    """Return an appropriate launcher log file"""
    launcher_path = Path(experiment_folder) /  "launcher_log_1.log"

    # Keep incrementing the counter
    ctr = 1
    while launcher_path.exists():
        new_file = f"launcher_log_{ctr}.log"
        launcher_path = Path(experiment_folder) /  new_file
        ctr += 1
    return launcher_path


def main(
    config_file: str,
    launcher_args: list,
    universal_args: dict,
    iterative_args: dict,
    comments: dict = None,
):
    """main."""
    # Create output experiment
    Path("logs").mkdir(exist_ok=True)

    experiment_name = launcher_args["experiment_name"]
    script_name = launcher_args.get("script_name", "run_training.py")
    experiment_folder = f"results/{experiment_name}/"
    dump_config_file(experiment_folder, config_file)

    ### Create launcher log
    launcher_path = get_launcher_log_name(experiment_folder)
    log = open(launcher_path, "w") if launcher_path is not None else None

    # List of current executions to run
    experiment_list = []

    # Loop over major arguments
    for arg_sublist in iterative_args:

        # Update universal args with iterative args
        # This overrides universal args with specific ones
        base_args = copy.deepcopy(universal_args)

        # Update each arg category
        base_args.update(arg_sublist)
        key, values = zip(*base_args.items())

        # Collapse this down
        combos = [dict(zip(key, val_combo)) for val_combo in itertools.product(*values)]
        experiment_list.extend(combos)

    program_strs = []

    for experiment_args in experiment_list:
        program_strs.append(
            build_python_string(
                experiment_folder,
                experiment_name,
                experiment_args,
                launcher_args,
                script_name,
            )
        )
    # Launch programs
    scripts_to_run = []
    launch_method = launcher_args.get("launch_method", "local")
    for str_num, (sbatch_args, python_str) in enumerate(program_strs):

        time.sleep(0.1)
        if launch_method == "slurm":
            slurm_script = launcher_args.get(
                "slurm_script", "launcher_scripts/generic_slurm.sh"
            )
            cmd_str = f'sbatch --export=CMD="{python_str}" {sbatch_args} {slurm_script}'
            scripts_to_run.append(cmd_str)

        elif launch_method == "local":
            vis_devices = launcher_args.get("visible_devices", None)
            if vis_devices is not None:
                vis_devices = ",".join(map(str, vis_devices))
                cmd_str = f"CUDA_VISIBLE_DEVICES={vis_devices} {python_str}"
            else:
                cmd_str = f"{python_str}"
            scripts_to_run.append(cmd_str)
        elif launch_method == "local_parallel":
            scripts_to_run.append(python_str)
        else:
            raise NotImplementedError()

    # Actually launch these
    if launch_method == "slurm":
        for cmd_str in scripts_to_run:
            print(f"Command String: ", cmd_str)
            subprocess.call(cmd_str, shell=True)
            if log is not None:
                log.write(cmd_str + "\n")
    elif launch_method == "local":
        for cmd_str in scripts_to_run:
            print(f"Command String: ", cmd_str)
            subprocess.call(cmd_str, shell=True)
            if log is not None:
                log.write(cmd_str + "\n")
    elif launch_method == "local_parallel":

        # Parallelize to several gpus
        vis_devices = launcher_args.get("visible_devices", None)
        if vis_devices is None:
            raise ValueError()

        sh_run_files = set()
        for str_num, cmd_str in enumerate(scripts_to_run):
            gpu_num = str_num % len(vis_devices)
            gpu = vis_devices[gpu_num]
            cmd_str_new = f"CUDA_VISIBLE_DEVICES={gpu} {cmd_str}"
            output_name = f"{launcher_path.parent / launcher_path.stem}_python_{gpu}.sh"

            with open(output_name, "a") as fp:
                fp.write(f"{cmd_str_new}\n")
            sh_run_files.add(output_name)

        # Single file to run alll launch scripts
        launch_all = f"{launcher_path.parent / launcher_path.stem}_launch_all.sh"
        with open(launch_all, "w") as fp:
            temp_str = [f"sh {i} > logs/{Path(i).name}.log &" 
                        for i in list(sh_run_files)]
            fp.write("\n".join(temp_str))
        print(f"Runnings script: {launch_all}")
        subprocess.call(f"source {launch_all}", shell=True)
    else:
        raise NotImplementedError()

    if log is not None:
        log.close()


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    args, config_file = get_args()
    main(config_file=config_file, **args)
