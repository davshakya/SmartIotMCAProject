import signal
import shutil
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
KAFKA_HOST = "localhost"
KAFKA_PORT = 9092
KAFKA_CONFIG = Path.home() / "kafka" / "config" / "kraft" / "server.properties"
LOG_FILE = None
LOG_LOCK = threading.Lock()


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
        print(line, flush=True)
        if LOG_FILE is not None:
            print(line, file=LOG_FILE, flush=True)


def print_step(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] [run_project] {message}"
    write_log_line(line)


def log_environment() -> None:
    print_step(f"Working directory: {ROOT}", "DEBUG")
    print_step(f"Python executable: {PYTHON}", "DEBUG")
    print_step(f"Python version: {sys.version.replace(chr(10), ' ')}", "DEBUG")
    print_step(f"Kafka config: {KAFKA_CONFIG}", "DEBUG")


def stream_subprocess_output(name: str, process: subprocess.Popen) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_log_line(f"[{timestamp}] [DEBUG] [{name}] {line.rstrip()}")


def run_command(name: str, command: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    print_step(f"Running command for {name}: {' '.join(command)}", "DEBUG")
    kwargs = {
        "cwd": ROOT,
        "check": True,
        "text": True,
    }
    if capture_output:
        kwargs["capture_output"] = True
    else:
        kwargs["stdout"] = LOG_FILE
        kwargs["stderr"] = subprocess.STDOUT
    return subprocess.run(command, **kwargs)


def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def detect_kafka_paths() -> tuple[Path, Path, Path] | None:
    kafka_home = Path.home() / "kafka"
    storage = kafka_home / "bin" / "kafka-storage.sh"
    server = kafka_home / "bin" / "kafka-server-start.sh"
    topics = kafka_home / "bin" / "kafka-topics.sh"
    config = kafka_home / "config" / "kraft" / "server.properties"

    if all(path.exists() for path in (storage, server, topics, config)):
        return storage, server, topics
    return None


def ensure_kafka_topic(topics_script: Path) -> None:
    run_command(
        "Kafka topic creation",
        [
            str(topics_script),
            "--create",
            "--if-not-exists",
            "--topic",
            "smart-meter",
            "--bootstrap-server",
            f"{KAFKA_HOST}:{KAFKA_PORT}",
            "--partitions",
            "1",
            "--replication-factor",
            "1",
        ],
    )


def format_kraft_storage_if_needed(storage_script: Path) -> None:
    meta_properties = Path("/tmp/kraft-combined-logs/meta.properties")
    if meta_properties.exists():
        print_step(f"KRaft storage already formatted at {meta_properties}", "DEBUG")
        return

    print_step("Formatting Kafka KRaft storage...")
    cluster_id = run_command(
        "Kafka random cluster id",
        [str(storage_script), "random-uuid"],
        capture_output=True,
    ).stdout.strip()
    print_step(f"Generated Kafka cluster id: {cluster_id}", "DEBUG")

    run_command(
        "Kafka KRaft format",
        [str(storage_script), "format", "-t", cluster_id, "-c", str(KAFKA_CONFIG)],
    )


def ensure_kafka_running() -> tuple[bool, subprocess.Popen | None]:
    if check_port(KAFKA_HOST, KAFKA_PORT):
        print_step(f"Kafka is reachable at {KAFKA_HOST}:{KAFKA_PORT}.")
        return True, None

    kafka_paths = detect_kafka_paths()
    if kafka_paths is None:
        print_step(
            "Kafka is not reachable at localhost:9092 and no local Kafka install was found. Falling back to direct mode without Kafka/Spark."
        )
        return False, None

    storage_script, server_script, topics_script = kafka_paths
    print_step(
        f"Detected local Kafka install: storage={storage_script}, server={server_script}, topics={topics_script}",
        "DEBUG",
    )
    format_kraft_storage_if_needed(storage_script)
    print_step("Starting local Kafka broker in KRaft mode...")
    process = subprocess.Popen(
        [str(server_script), str(KAFKA_CONFIG)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=stream_subprocess_output,
        args=("Kafka broker", process),
        daemon=True,
    ).start()

    for _ in range(20):
        if check_port(KAFKA_HOST, KAFKA_PORT):
            print_step(f"Kafka broker started at {KAFKA_HOST}:{KAFKA_PORT}.")
            ensure_kafka_topic(topics_script)
            return True, process
        if process.poll() is not None:
            break
        time.sleep(1)

    if process.poll() is None:
        process.terminate()
        process.wait(timeout=5)

    print_step(
        f"Kafka broker did not start cleanly. Exit code: {process.poll()}. Falling back to direct mode without Kafka/Spark.",
        "ERROR",
    )
    return False, None


def run_training() -> None:
    print_step("Training anomaly detection model...")
    run_command("Model training", [PYTHON, "train_model.py"])
    print_step("Model training complete.")


def start_process(name: str, command: list[str]) -> subprocess.Popen:
    print_step(f"Starting {name}: {' '.join(command)}")
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


def remove_process(processes: list[tuple[str, subprocess.Popen]], name: str) -> None:
    for index, (process_name, _) in enumerate(processes):
        if process_name == name:
            processes.pop(index)
            return


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
        if process.poll() is None:
            print_step(f"Stopping {name}...")
            process.terminate()

    deadline = time.time() + 8
    for _, process in processes:
        if process.poll() is None:
            remaining = max(0, deadline - time.time())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print_step(f"Force killing {process.pid} after timeout.", "WARNING")
                process.kill()


def main() -> None:
    log_path = init_logging()
    print_step(f"Saving logs to {log_path}")
    log_environment()
    use_kafka, kafka_process = ensure_kafka_running()
    run_training()

    processes: list[tuple[str, subprocess.Popen]] = []
    stop_requested = False

    def handle_signal(signum, _frame) -> None:
        nonlocal stop_requested
        if stop_requested:
            return
        stop_requested = True
        print_step(f"Received signal {signum}. Shutting down...")
        terminate_processes(processes)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if kafka_process is not None:
            processes.append(("Kafka broker", kafka_process))

        processes.append(
            (
                "FastAPI backend",
                start_process(
                    "FastAPI backend",
                    [PYTHON, "-m", "uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"],
                ),
            )
        )
        time.sleep(3)

        spark_process = None
        if use_kafka:
            spark_submit = shutil.which("spark-submit")
            if spark_submit is None:
                print_step(
                    "spark-submit is not available on PATH. Falling back to direct mode so the dashboard can stay up.",
                    "WARNING",
                )
                use_kafka = False
            else:
                spark_process = start_process("Spark streaming job", [spark_submit, "spark_stream.py"])
                processes.append(("Spark streaming job", spark_process))
                time.sleep(5)
                if spark_process.poll() is not None:
                    print_step(
                        f"Spark streaming job exited early with code {spark_process.poll()}. Falling back to direct mode.",
                        "WARNING",
                    )
                    remove_process(processes, "Spark streaming job")
                    use_kafka = False

        processes.append(
            (
                "Meter simulator",
                start_process(
                    "Meter simulator",
                    [PYTHON, "meter_simulator.py"]
                    if use_kafka
                    else [PYTHON, "meter_simulator.py", "--mode", "direct"],
                ),
            )
        )
        time.sleep(2)

        processes.append(
            (
                "Flask dashboard",
                start_process("Flask dashboard", [PYTHON, "flask_app.py"]),
            )
        )

        print_step("Project is running.")
        print_step(f"Mode: {'Kafka + Spark' if use_kafka else 'Direct fallback'}")
        print_step("Dashboard: http://localhost:5000")
        print_step("API: http://localhost:8000/data")
        print_step("Press Ctrl+C to stop all services.")

        while True:
            for name, process in processes:
                code = process.poll()
                if code is not None:
                    if name == "Spark streaming job" and use_kafka:
                        print_step(
                            f"Spark streaming job exited with code {code}. Switching to direct mode to keep the app running.",
                            "WARNING",
                        )
                        remove_process(processes, "Spark streaming job")
                        use_kafka = False
                        for process_name, current_process in list(processes):
                            if process_name == "Meter simulator":
                                terminate_process(process_name, current_process)
                                remove_process(processes, process_name)
                                break
                        direct_simulator = start_process(
                            "Meter simulator",
                            [PYTHON, "meter_simulator.py", "--mode", "direct"],
                        )
                        processes.append(("Meter simulator", direct_simulator))
                        break
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
