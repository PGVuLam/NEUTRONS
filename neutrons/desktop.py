from __future__ import annotations

import logging
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import httpx

from neutrons.config import settings
from neutrons.logging_config import configure_logging

WINDOW_TITLE = "NEUTRONS"
BACKGROUND = "#111318"
SIDEBAR = "#171a21"
PANEL = "#1c2029"
ACCENT = "#7c9cff"
TEXT = "#eef1f7"
MUTED = "#9aa3b2"
API_BASE_URL = f"http://127.0.0.1:{settings.api_port}"
logger = logging.getLogger(__name__)


class DesktopStartupError(RuntimeError):
    """Raised when the desktop app cannot connect to its local backend."""


class ProcessHandle(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class ManagedBackend:
    def __init__(self, process: ProcessHandle, output: BinaryIO) -> None:
        self.process = process
        self.output = output

    def diagnostic_output(self) -> str:
        self.output.flush()
        self.output.seek(0)
        return self.output.read().decode("utf-8", errors="replace")[-4000:].strip()

    def stop(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        except ProcessLookupError:
            pass

    def close_output(self) -> None:
        self.output.close()


def backend_is_healthy(client: httpx.Client) -> bool:
    try:
        response = client.get(f"{API_BASE_URL}/healthz")
        response.raise_for_status()
    except httpx.HTTPError:
        return False

    try:
        payload = response.json()
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("status") == "ok" and payload.get("app") == settings.app_name


def _port_is_open() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", settings.api_port), timeout=0.25):
            return True
    except OSError:
        return False


def _startup_error_message(message: str, backend: ManagedBackend | None = None) -> str:
    diagnostics = backend.diagnostic_output() if backend else ""
    if diagnostics:
        return f"{message}\n\nBackend log:\n{diagnostics}"
    return message


def ensure_backend() -> ManagedBackend | None:
    with httpx.Client(timeout=1.0) as client:
        if backend_is_healthy(client):
            return None

    if _port_is_open():
        raise DesktopStartupError(
            f"Port {settings.api_port} đang được một ứng dụng khác sử dụng, "
            "hoặc dịch vụ ở đó không phải backend NEUTRONS."
        )

    output = tempfile.TemporaryFile(mode="w+b")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "neutrons.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(settings.api_port),
            ],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        output.close()
        raise DesktopStartupError(f"Không thể tạo tiến trình backend: {exc}") from exc
    backend = ManagedBackend(process, output)

    deadline = time.monotonic() + 20
    with httpx.Client(timeout=1.0) as client:
        while time.monotonic() < deadline:
            if backend.process.poll() is not None:
                message = _startup_error_message(
                    "Backend NEUTRONS dừng ngay sau khi khởi động.", backend
                )
                backend.stop()
                backend.close_output()
                raise DesktopStartupError(message)
            if backend_is_healthy(client):
                logger.info(
                    "Started local backend",
                    extra={
                        "event": "desktop_backend_started",
                        "context": {"port": settings.api_port},
                    },
                )
                return backend
            time.sleep(0.25)

    backend.stop()
    message = _startup_error_message(
        f"Backend không phản hồi sau 20 giây tại port {settings.api_port}.", backend
    )
    backend.close_output()
    raise DesktopStartupError(message)


class NeutronsDesktop:
    def __init__(self, root: tk.Tk, backend_process: ManagedBackend | None) -> None:
        self.root = root
        self.backend_process = backend_process
        self.results: queue.Queue[tuple[str, str]] = queue.Queue()
        self.history: list[dict[str, str]] = []
        self.busy = False

        self.root.title(f"{WINDOW_TITLE} | Trợ lý AI cục bộ")
        self.root.geometry("1020x720")
        self.root.minsize(760, 560)
        self.root.configure(bg=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_styles()
        self._build_layout()
        self._show_welcome()
        self.root.after(100, self._poll_results)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("Sidebar.TFrame", background=SIDEBAR)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure(
            "Send.TButton",
            background=ACCENT,
            foreground="#101522",
            borderwidth=0,
            padding=(18, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Send.TButton", background=[("disabled", "#4b5364"), ("active", "#98b0ff")])
        style.configure(
            "Quiet.TButton",
            background=SIDEBAR,
            foreground=TEXT,
            borderwidth=0,
            padding=(10, 7),
            font=("Segoe UI", 9),
        )
        style.map("Quiet.TButton", background=[("active", PANEL)])

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", padding=(20, 24))
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.configure(width=220)
        sidebar.grid_propagate(False)

        tk.Label(
            sidebar,
            text="✦  NEUTRONS",
            bg=SIDEBAR,
            fg=TEXT,
            font=("Segoe UI Semibold", 17),
        ).pack(anchor="w")
        tk.Label(
            sidebar,
            text="TRỢ LÝ AI CÁ NHÂN",
            bg=SIDEBAR,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", pady=(5, 28))

        self._sidebar_section(sidebar, "KHÔNG GIAN")
        self._sidebar_item(sidebar, "◉", "Trò chuyện mới", active=True)
        self._sidebar_section(sidebar, "HỆ THỐNG")
        self._sidebar_item(sidebar, "●", "Chạy cục bộ", color="#70d6a4")
        self._sidebar_item(sidebar, "◈", settings.model_name)

        spacer = tk.Frame(sidebar, bg=SIDEBAR)
        spacer.pack(fill="both", expand=True)
        tk.Label(
            sidebar,
            text="Dữ liệu trao đổi được gửi\nđến Ollama trên máy này.",
            bg=SIDEBAR,
            fg=MUTED,
            font=("Segoe UI", 9),
            justify="left",
        ).pack(anchor="w", pady=(12, 0))

        main = ttk.Frame(self.root, padding=(28, 22, 28, 20))
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        header = tk.Frame(main, bg=BACKGROUND)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Trò chuyện",
            bg=BACKGROUND,
            fg=TEXT,
            font=("Segoe UI Semibold", 19),
        ).grid(row=0, column=0, sticky="w")
        self.status_label = tk.Label(
            header,
            text=f"●  ĐANG DÙNG  {settings.model_name}",
            bg=BACKGROUND,
            fg="#70d6a4",
            font=("Segoe UI", 9, "bold"),
        )
        self.status_label.grid(row=0, column=1, sticky="e", padx=(12, 18))
        ttk.Button(header, text="Xóa hội thoại", style="Quiet.TButton", command=self.clear_chat).grid(
            row=0, column=2, sticky="e"
        )

        chat_panel = tk.Frame(main, bg=PANEL, highlightthickness=1, highlightbackground="#292e39")
        chat_panel.grid(row=1, column=0, sticky="nsew")
        chat_panel.grid_rowconfigure(0, weight=1)
        chat_panel.grid_columnconfigure(0, weight=1)

        self.chat = scrolledtext.ScrolledText(
            chat_panel,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            selectbackground="#39445f",
            relief=tk.FLAT,
            borderwidth=0,
            padx=24,
            pady=22,
            font=("Segoe UI", 11),
            spacing1=3,
            spacing3=9,
        )
        self.chat.grid(row=0, column=0, sticky="nsew")
        self.chat.tag_configure("user_label", foreground=ACCENT, font=("Segoe UI", 10, "bold"), spacing1=12)
        self.chat.tag_configure("assistant_label", foreground="#70d6a4", font=("Segoe UI", 10, "bold"), spacing1=12)
        self.chat.tag_configure("body", foreground=TEXT, lmargin1=12, lmargin2=12, spacing3=12)
        self.chat.tag_configure("welcome", foreground=MUTED, justify="center", spacing1=8, spacing3=8)

        compose = tk.Frame(main, bg=BACKGROUND)
        compose.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        compose.grid_columnconfigure(0, weight=1)
        input_panel = tk.Frame(compose, bg=PANEL, highlightthickness=1, highlightbackground="#343a48")
        input_panel.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        input_panel.grid_columnconfigure(0, weight=1)
        self.input = tk.Text(
            input_panel,
            height=3,
            wrap=tk.WORD,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief=tk.FLAT,
            borderwidth=0,
            padx=14,
            pady=12,
            font=("Segoe UI", 11),
        )
        self.input.grid(row=0, column=0, sticky="ew")
        self.input.bind("<Return>", self._on_enter)
        ttk.Button(compose, text="Gửi  ➤", style="Send.TButton", command=self.send_message).grid(
            row=0, column=1, sticky="se"
        )
        tk.Label(
            main,
            text="Enter để gửi · Shift+Enter để xuống dòng",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Segoe UI", 8),
        ).grid(row=3, column=0, sticky="e", pady=(7, 0))

    @staticmethod
    def _sidebar_section(parent: tk.Widget, text: str) -> None:
        tk.Label(
            parent,
            text=text,
            bg=SIDEBAR,
            fg="#70798a",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", pady=(0, 8))

    @staticmethod
    def _sidebar_item(
        parent: tk.Widget, icon: str, text: str, active: bool = False, color: str = MUTED
    ) -> None:
        background = "#242a36" if active else SIDEBAR
        row = tk.Frame(parent, bg=background, padx=10, pady=9)
        row.pack(fill="x", pady=(0, 5))
        tk.Label(row, text=icon, bg=background, fg=color, font=("Segoe UI", 10)).pack(side="left")
        tk.Label(
            row,
            text=text,
            bg=background,
            fg=TEXT if active else MUTED,
            font=("Segoe UI", 9, "bold" if active else "normal"),
        ).pack(side="left", padx=(9, 0))

    def _show_welcome(self) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(
            tk.END,
            "\n\n\nNEUTRONS sẵn sàng\n\n"
            "Hãy bắt đầu bằng một câu hỏi hoặc mô tả việc bạn muốn làm.\n"
            "Tin nhắn sẽ được xử lý bởi model Ollama đang chạy trên máy.",
            "welcome",
        )
        self.chat.configure(state=tk.DISABLED)

    def _append_message(self, speaker: str, message: str, is_user: bool) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(
            tk.END,
            f"\n{speaker}\n",
            "user_label" if is_user else "assistant_label",
        )
        self.chat.insert(tk.END, f"{message.strip()}\n", "body")
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _on_enter(self, event: tk.Event[Any]) -> str | None:
        if event.state & 0x0001:
            return None
        self.send_message()
        return "break"

    def send_message(self) -> None:
        if self.busy:
            return
        message = self.input.get("1.0", tk.END).strip()
        if not message:
            return
        self.input.delete("1.0", tk.END)
        self._append_message("Bạn", message, is_user=True)
        request_history = self.history[-40:]
        self.history.append({"role": "user", "content": message})
        self.busy = True
        self.status_label.configure(text="●  ĐANG SUY NGHĨ…", fg="#f0c674")
        self.root.update_idletasks()
        threading.Thread(
            target=self._request_chat,
            args=(message, request_history),
            daemon=True,
        ).start()

    def _request_chat(self, message: str, history: list[dict[str, str]]) -> None:
        try:
            response = httpx.post(
                f"{API_BASE_URL}/api/v1/chat",
                json={"message": message, "history": history},
                timeout=settings.request_timeout_seconds + 10,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Backend trả về dữ liệu không hợp lệ.")
            answer = payload.get("response")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Backend trả về câu trả lời không hợp lệ.")
            self.results.put(("ok", answer.strip()))
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            try:
                payload = exc.response.json()
                if isinstance(payload, dict):
                    detail = payload.get("detail", detail)
            except ValueError:
                pass
            self.results.put(("error", f"Backend trả lỗi {exc.response.status_code}: {detail}"))
        except (httpx.HTTPError, ValueError) as exc:
            self.results.put(("error", f"Không thể nhận câu trả lời: {exc}"))

    def _poll_results(self) -> None:
        while True:
            try:
                status, result = self.results.get_nowait()
            except queue.Empty:
                break
            self.busy = False
            if status == "ok":
                self._append_message("NEUTRONS", result, is_user=False)
                self.history.append({"role": "assistant", "content": result})
                self.status_label.configure(
                    text=f"●  ĐANG DÙNG  {settings.model_name}", fg="#70d6a4"
                )
            else:
                self._append_message("Thông báo lỗi", result, is_user=False)
                if self.history and self.history[-1]["role"] == "user":
                    self.history.pop()
                self.status_label.configure(text="●  CẦN KIỂM TRA", fg="#ff8888")
        self.root.after(100, self._poll_results)

    def clear_chat(self) -> None:
        if self.busy:
            return
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)
        self.history.clear()
        self._show_welcome()

    def close(self) -> None:
        try:
            if self.backend_process is not None:
                try:
                    self.backend_process.stop()
                finally:
                    self.backend_process.close_output()
        finally:
            self.root.destroy()


def main() -> None:
    configure_logging()
    try:
        backend_process = ensure_backend()
    except DesktopStartupError as exc:
        logger.error(
            "Desktop backend startup failed",
            extra={"event": "desktop_backend_startup_failed"},
        )
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Không thể khởi động NEUTRONS", str(exc))
        root.destroy()
        raise SystemExit(1) from exc

    try:
        root = tk.Tk()
        NeutronsDesktop(root, backend_process)
        root.mainloop()
    except Exception:
        if backend_process is not None:
            try:
                backend_process.stop()
            finally:
                backend_process.close_output()
        raise


if __name__ == "__main__":
    main()
