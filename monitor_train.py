import os
import sys
import time
import pty
import subprocess

SCRIPT_NAME = "train.py"
SCRIPT_DIR = "/home/adir/vscode/MammoCLIP"  # Adjust to your actual path if needed
CONDA_PATH = "~/anaconda3/bin/activate base"      # Path to conda "base" env
RESTART_DELAY_SECONDS = 60                       # Delay before restarting on crash

def run_command_with_tty(command, cwd=None):
    """
    Spawns a subprocess connected to a pseudo-tty (PTY).
    This allows TQDM to see a terminal (isatty=True) and display
    a proper in-place progress bar instead of printing new lines.
    """
    # Create a new pseudo-terminal
    master_fd, slave_fd = pty.openpty()

    # Launch the command in a shell, attaching both stdin & stdout to the PTY.
    process = subprocess.Popen(
        command,
        shell=True,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=cwd
    )

    # We close the slave fd in our parent, because only the child subprocess should use it.
    os.close(slave_fd)

    # Continuously read from the master fd and write to our own stdout
    try:
        while True:
            # Read up to 1024 bytes from PTY
            output = os.read(master_fd, 1024)
            if not output:
                break
            # Write directly to our Python script's stdout
            sys.stdout.buffer.write(output)
            sys.stdout.flush()
    except OSError:
        pass

    # Wait for the process to finish
    process.wait()
    exit_code = process.returncode

    # Close the master fd now that we're done reading
    os.close(master_fd)

    return exit_code

def main():
    os.chdir(SCRIPT_DIR)  # Ensure the working dir is where train-ugan.py resides.

    while True:
        print("============================================================")
        print(f"Launching {SCRIPT_NAME} with accelerate (PTY mode).")
        print("============================================================")

        # Build our shell command:
        # 1) source base environment
        # 2) run accelerate launch train-ugan.py
        command = (
            f"bash -c 'source {CONDA_PATH} && "
            f"accelerate launch {SCRIPT_NAME} --resume --dir ./mammoclip-v1'"
        )

        # Run inside a pseudo-tty to preserve TQDM's live progress bars
        exit_code = run_command_with_tty(command, cwd=SCRIPT_DIR)

        if exit_code == 0:
            # Script ended "successfully"
            print(f"\n[MONITOR] {SCRIPT_NAME} ended with exit code 0 (success).")
            # Decide whether to break or to keep restarting even on success
            break
        else:
            # Non-zero exit => crash or error
            print(f"\n[MONITOR] {SCRIPT_NAME} crashed with exit code {exit_code}.")
            print(f"[MONITOR] Restarting in {RESTART_DELAY_SECONDS} seconds...\n")
            time.sleep(RESTART_DELAY_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MONITOR] Monitoring interrupted by user.")
    except Exception as e:
        print(f"\n[MONITOR] Uncaught exception in monitoring loop: {e}")