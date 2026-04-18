from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
PYTHON = sys.executable
LOG_FILE = None
LOG_LOCK = threading.Lock()
DEBUG_MODE = False


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def init_logging() -> Path:
    global LOG_FILE
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    LOG_FILE = log_path.open("a", encoding="utf-8")
    return log_path


def close_logging() -> None:
    global LOG_FILE
    if LOG_FILE is not None:
        LOG_FILE.close()
        LOG_FILE = None


def write_log_line(line: str) -> None:
    with LOG_LOCK:
        if LOG_FILE is not None:
            print(line, file=LOG_FILE, flush=True)


def print_step(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] [run_project] {message}"
    if level != "DEBUG":
        print(line, flush=True)
    if DEBUG_MODE:
        write_log_line(line)


def log_environment(interval: float) -> None:
    print_step(f"Working directory: {ROOT}", "DEBUG")
    print_step(f"Python executable: {PYTHON}", "DEBUG")
    print_step(f"Simulator interval: {interval}", "DEBUG")


def stream_subprocess_output(name: str, process: subprocess.Popen) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_log_line(f"[{timestamp}] [DEBUG] [{name}] {line.rstrip()}")


def run_command(name: str, command: list[str]) -> None:
    print_step(f"Running command for {name}: {' '.join(command)}", "DEBUG")
    kwargs = {
        "cwd": ROOT,
        "check": True,
        "text": True,
    }
    if DEBUG_MODE:
        kwargs["stdout"] = LOG_FILE
        kwargs["stderr"] = subprocess.STDOUT
    else:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.STDOUT
    subprocess.run(command, **kwargs)


def wait_for_port(host: str, port: int, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def run_training() -> None:
    print_step("Training Isolation Forest and LSTM anomaly detectors...")
    run_command("Model training", [PYTHON, "train_model.py"])
    print_step("Model training complete.")


def start_process(name: str, command: list[str]) -> subprocess.Popen:
    print_step(f"Starting {name}: {' '.join(command)}")
    if DEBUG_MODE:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(
            target=stream_subprocess_output,
            args=(name, process),
            daemon=True,
        ).start()
        return process

    return subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )


def terminate_process(name: str, process: subprocess.Popen) -> None:
    if process.poll() is None:
        print_step(f"Stopping {name}...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print_step(f"Force killing {process.pid} after timeout.", "WARNING")
            process.kill()


def terminate_processes(processes: list[tuple[str, subprocess.Popen]]) -> None:
    for name, process in reversed(processes):
        terminate_process(name, process)


def main() -> None:
    global DEBUG_MODE

    parser = argparse.ArgumentParser(description="Run Smart Meter project services in direct mode")
    parser.add_argument(
        "--debug",
        default="false",
        help="Set to true to save full debug logs under logs/.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between simulator batches.",
    )
    parser.add_argument(
        "--skip-training",
        default="false",
        help="Set to true to reuse existing detector artifacts without retraining.",
    )
    args = parser.parse_args()
    DEBUG_MODE = parse_bool(args.debug)
    skip_training = parse_bool(args.skip_training)

    if DEBUG_MODE:
        log_path = init_logging()
        print_step(f"Saving logs to {log_path}")
        log_environment(args.interval)
    else:
        print_step("Running in normal mode. Use --debug true to save full logs.")

    print_step("Launching the project")
    if not skip_training:
        run_training()
    else:
        print_step("Skipping training and reusing existing detector artifacts.")

    processes: list[tuple[str, subprocess.Popen]] = []

    def handle_signal(signum, _frame) -> None:
        print_step(f"Received signal {signum}. Shutting down...")
        terminate_processes(processes)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        backend_process = start_process(
            "FastAPI backend",
            [PYTHON, "-m", "uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"],
        )
        processes.append(("FastAPI backend", backend_process))
        if not wait_for_port("127.0.0.1", 8000):
            raise RuntimeError("FastAPI backend did not become ready on port 8000.")

        flask_process = start_process("Flask dashboard", [PYTHON, "flask_app.py"])
        processes.append(("Flask dashboard", flask_process))

        simulator_process = start_process(
            "Meter simulator",
            [PYTHON, "meter_simulator.py", "--interval", str(args.interval)],
        )
        processes.append(("Meter simulator", simulator_process))

        print_step("Project is running.")
        print_step("Dashboard: http://localhost:5000")
        print_step("API: http://localhost:8000/data")
        print_step("Press Ctrl+C to stop all services.")

        while True:
            for name, process in processes:
                code = process.poll()
                if code is not None:
                    print_step(f"{name} exited unexpectedly with code {code}.", "ERROR")
                    terminate_processes(processes)
                    raise SystemExit(code)
            time.sleep(1)
    except Exception:
        print_step("Unhandled exception in launcher:", "ERROR")
        for line in traceback.format_exc().rstrip().splitlines():
            print_step(line, "ERROR")
        raise
    finally:
        terminate_processes(processes)
        close_logging()


if __name__ == "__main__":
    main()
