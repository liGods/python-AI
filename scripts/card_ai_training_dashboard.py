from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "card_ai"
TRAINING_ROOT = DATA_ROOT / "training"
STATE_PATH = TRAINING_ROOT / "continuous_state.json"
PROGRESS_PATH = TRAINING_ROOT / "training_progress.json"
POINTERS_PATH = DATA_ROOT / "models" / "pointers.json"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def training_processes() -> list[dict[str, Any]]:
    result = []
    own_pid = os.getpid()
    for process in psutil.process_iter(["pid", "cmdline", "create_time", "memory_info"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if process.info["pid"] == own_pid or "ok_tasks.card_ai continuous" not in command:
                continue
            memory = process.info.get("memory_info")
            result.append(
                {
                    "pid": process.info["pid"],
                    "created": float(process.info.get("create_time") or 0),
                    "memory_gib": round((memory.rss if memory else 0) / 1024**3, 2),
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, TypeError):
            continue
    return result


def gpu_metrics() -> dict[str, float]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command, text=True, encoding="utf-8", errors="replace", timeout=3, creationflags=CREATE_NO_WINDOW
        ).strip().splitlines()[0]
        utilization, used, total, power = (float(value.strip()) for value in output.split(","))
        return {"utilization": utilization, "memory_used": used, "memory_total": total, "power": power}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"utilization": 0.0, "memory_used": 0.0, "memory_total": 0.0, "power": 0.0}


def candidate_models() -> list[dict[str, Any]]:
    candidates_root = TRAINING_ROOT / "candidates"
    candidates = []
    if not candidates_root.is_dir():
        return candidates
    for path in sorted(candidates_root.rglob("training_metadata.json")):
        metadata = read_json(path, {})
        if not isinstance(metadata, dict):
            continue
        candidates.append(
            {
                "name": path.parent.name,
                "backbone": str(metadata.get("backbone", "-")),
                "seed": str(metadata.get("seed", "-")),
                "validation_mse": metadata.get("validation_mse"),
                "teammate_accuracy": metadata.get("teammate_accuracy"),
                "device": str(metadata.get("device", "-")),
            }
        )
    return candidates


def latest_error_log() -> dict[str, Any]:
    paths = []
    for pattern in ("*.err.log", "*error*.log", "*errors*.log"):
        paths.extend(TRAINING_ROOT.glob(pattern))
        logs_root = TRAINING_ROOT / "logs"
        if logs_root.is_dir():
            paths.extend(logs_root.rglob(pattern))
    files = [path for path in paths if path.is_file()]
    if not files:
        return {"path": None, "size": 0, "tail": "未发现错误日志。"}
    path = max(files, key=lambda item: item.stat().st_mtime)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    return {"path": str(path), "size": path.stat().st_size, "tail": "\n".join(lines[-8:]) or "错误日志为空。"}


def collect_metrics() -> dict[str, Any]:
    state = read_json(STATE_PATH, {})
    config = state.get("config", {}) if isinstance(state, dict) else {}
    target = int(config.get("target_games", 200_000) or 200_000)
    completed = int(state.get("completed_games", 0) or 0)
    candidates = candidate_models()
    processes = training_processes()
    gpu = gpu_metrics()
    training_progress = read_json(PROGRESS_PATH, {})
    status = str(state.get("status", "unknown"))

    if status == "completed":
        phase, phase_index = "训练与评测已完成", 4
    elif not processes:
        phase, phase_index = "训练进程已停止", 0
    elif len(candidates) >= 6:
        phase, phase_index = "固定牌局评测", 3
    elif gpu["utilization"] >= 20 and completed >= target * 0.9:
        phase, phase_index = "GPU 模型训练", 2
    else:
        phase, phase_index = "自我对战生成牌局", 1

    display_completed = target if phase_index >= 2 else completed
    started = min((item["created"] for item in processes), default=0)
    elapsed_seconds = max(0, int(datetime.now().timestamp() - started)) if started else 0
    return {
        "state": state,
        "status": status,
        "phase": phase,
        "phase_index": phase_index,
        "completed": completed,
        "display_completed": display_completed,
        "target": target,
        "cycle": int(state.get("cycle", 0) or 0),
        "processes": processes,
        "elapsed_seconds": elapsed_seconds,
        "gpu": gpu,
        "candidates": candidates,
        "training_progress": training_progress,
        "pointers": read_json(POINTERS_PATH, {"stable": None, "candidate": None, "rollback": None}),
        "error": latest_error_log(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def format_duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class GpuChart(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[float] = []
        self.setMinimumHeight(145)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def add_value(self, value: float) -> None:
        self.values = (self.values + [max(0.0, min(100.0, value))])[-90:]
        self.update()

    def paintEvent(self, event: Any) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111a2c"))
        margin = 15
        width = max(1, self.width() - margin * 2)
        height = max(1, self.height() - margin * 2)
        painter.setPen(QPen(QColor("#263450"), 1))
        for step in range(5):
            y = margin + height * step / 4
            painter.drawLine(margin, int(y), margin + width, int(y))
        if len(self.values) < 2:
            painter.setPen(QColor("#74829b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "正在采集 GPU 数据…")
            return
        points = QPolygonF()
        for index, value in enumerate(self.values):
            x = margin + width * index / max(1, len(self.values) - 1)
            y = margin + height * (1.0 - value / 100.0)
            points.append(QPointF(x, y))
        painter.setPen(QPen(QColor("#3ee6b1"), 2.5))
        painter.drawPolyline(points)


class MetricCard(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 15, 18, 15)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        self.value_label = QLabel("—")
        self.value_label.setObjectName("metricValue")
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("metricDetail")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)

    def set_values(self, value: str, detail: str = "") -> None:
        self.value_label.setText(value)
        self.detail_label.setText(detail)


class TrainingDashboard(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("百将牌 AI 训练监控")
        self.resize(1120, 790)
        self.setMinimumSize(940, 680)
        self._build_ui()
        self._apply_style()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)
        self.refresh()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 22)
        layout.setSpacing(14)

        heading_row = QHBoxLayout()
        heading = QLabel("百将牌 AI · 训练控制台")
        heading.setObjectName("heading")
        self.updated_label = QLabel("等待数据")
        self.updated_label.setObjectName("updated")
        heading_row.addWidget(heading)
        heading_row.addStretch()
        heading_row.addWidget(self.updated_label)
        layout.addLayout(heading_row)

        self.phase_label = QLabel("正在读取训练状态…")
        self.phase_label.setObjectName("phase")
        layout.addWidget(self.phase_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 200_000)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        cards = QGridLayout()
        cards.setSpacing(12)
        self.games_card = MetricCard("自我对战")
        self.gpu_card = MetricCard("GPU 负载")
        self.model_card = MetricCard("候选模型")
        self.process_card = MetricCard("训练进程")
        for index, card in enumerate((self.games_card, self.gpu_card, self.model_card, self.process_card)):
            cards.addWidget(card, 0, index)
        layout.addLayout(cards)

        middle = QHBoxLayout()
        middle.setSpacing(14)
        chart_frame = QFrame()
        chart_frame.setObjectName("panel")
        chart_layout = QVBoxLayout(chart_frame)
        chart_title = QLabel("GPU 利用率 · 最近 3 分钟")
        chart_title.setObjectName("panelTitle")
        self.gpu_chart = GpuChart()
        chart_layout.addWidget(chart_title)
        chart_layout.addWidget(self.gpu_chart)
        middle.addWidget(chart_frame, 3)

        registry_frame = QFrame()
        registry_frame.setObjectName("panel")
        registry_layout = QVBoxLayout(registry_frame)
        registry_title = QLabel("训练内部过程")
        registry_title.setObjectName("panelTitle")
        self.training_detail_label = QLabel()
        self.training_detail_label.setWordWrap(True)
        self.training_detail_label.setObjectName("registry")
        self.model_training_progress = QProgressBar()
        self.model_training_progress.setRange(0, 1000)
        registry_layout.addWidget(registry_title)
        registry_layout.addWidget(self.training_detail_label)
        registry_layout.addWidget(self.model_training_progress)
        version_title = QLabel("模型版本状态")
        version_title.setObjectName("panelTitle")
        self.registry_label = QLabel()
        self.registry_label.setWordWrap(True)
        self.registry_label.setObjectName("registry")
        registry_layout.addWidget(version_title)
        registry_layout.addWidget(self.registry_label)
        registry_layout.addStretch()
        middle.addWidget(registry_frame, 2)
        layout.addLayout(middle, 3)

        lower = QHBoxLayout()
        lower.setSpacing(14)
        self.model_table = QTableWidget(0, 5)
        self.model_table.setHorizontalHeaderLabels(["模型", "骨干", "种子", "验证 MSE", "队友预测"])
        self.model_table.horizontalHeader().setStretchLastSection(True)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.model_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        lower.addWidget(self.model_table, 3)
        self.error_text = QTextEdit()
        self.error_text.setReadOnly(True)
        self.error_text.setPlaceholderText("错误与诊断信息")
        lower.addWidget(self.error_text, 2)
        layout.addLayout(lower, 3)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b1220; color: #eaf0fb; font-family: 'Microsoft YaHei UI'; }
            QLabel#heading { font-size: 25px; font-weight: 700; color: #f5f8ff; }
            QLabel#updated { color: #7786a3; font-size: 12px; }
            QLabel#phase { color: #70e8c0; font-size: 16px; font-weight: 600; }
            QFrame#metricCard, QFrame#panel { background: #111a2c; border: 1px solid #202d46; border-radius: 10px; }
            QLabel#metricTitle { color: #8392ae; font-size: 12px; }
            QLabel#metricValue { color: #f4f7ff; font-size: 23px; font-weight: 700; }
            QLabel#metricDetail { color: #71809c; font-size: 11px; }
            QLabel#panelTitle { color: #cdd7e8; font-size: 13px; font-weight: 600; }
            QLabel#registry { color: #94a4bf; font-size: 13px; line-height: 1.5; }
            QProgressBar { background: #111a2c; border: 1px solid #263450; border-radius: 8px; height: 24px; text-align: center; color: #f6f9ff; font-weight: 600; }
            QProgressBar::chunk { background: #31cfa0; border-radius: 7px; }
            QTableWidget, QTextEdit { background: #111a2c; alternate-background-color: #0e1728; border: 1px solid #202d46; border-radius: 9px; color: #d8e1f0; gridline-color: #202d46; }
            QHeaderView::section { background: #18243a; color: #91a1bc; border: none; padding: 7px; font-weight: 600; }
            QTableWidget::item { padding: 5px; }
            QScrollBar:vertical { background: #111a2c; width: 10px; }
            QScrollBar::handle:vertical { background: #30405d; border-radius: 5px; min-height: 28px; }
            """
        )

    def refresh(self) -> None:
        metrics = collect_metrics()
        completed = metrics["display_completed"]
        target = metrics["target"]
        raw_completed = metrics["completed"]
        gpu = metrics["gpu"]
        processes = metrics["processes"]
        candidates = metrics["candidates"]

        self.phase_label.setText(f"● {metrics['phase']}  ·  第 {metrics['cycle']} 轮")
        self.progress.setRange(0, max(1, target))
        self.progress.setValue(min(completed, target))
        self.progress.setFormat(f"{completed:,} / {target:,} 局  ·  {completed / max(1, target):.1%}")
        self.games_card.set_values(f"{raw_completed:,} 局", f"目标 {target:,} · 自我对战阶段已完成")
        self.gpu_card.set_values(f"{gpu['utilization']:.0f}%", f"显存 {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MiB · {gpu['power']:.0f} W")
        self.model_card.set_values(f"{len(candidates)} / 6", "LSTM 3 个 + Transformer 3 个")
        pid_text = ", ".join(str(item["pid"]) for item in processes) or "—"
        memory = sum(float(item["memory_gib"]) for item in processes)
        self.process_card.set_values("运行中" if processes else "已停止", f"PID {pid_text} · {memory:.1f} GiB · {format_duration(metrics['elapsed_seconds'])}")
        self.gpu_chart.add_value(gpu["utilization"])

        pointers = metrics["pointers"]
        progress = metrics.get("training_progress") or {}
        phase_names = {
            "loading_samples": "读取并编码牌局样本",
            "teacher": "完整信息教师模型",
            "student": "可见信息学生模型",
            "validation": "分批验证模型",
            "model_complete": "模型导出完成",
            "model_reused": "复用已完成模型",
            "screening": "候选模型快速初筛",
            "evaluation": "固定牌局对战评测",
        }
        if progress:
            epoch = progress.get("epoch")
            epoch_text = f"Epoch {epoch}/{progress.get('epochs')}" if epoch else ""
            batch = progress.get("batch")
            batch_text = f"Batch {batch}/{progress.get('batches')}" if batch else ""
            if progress.get("phase") in {"screening", "evaluation"}:
                evaluated = int(progress.get("evaluation_completed", 0) or 0)
                evaluation_total = int(progress.get("evaluation_deals", 0) or 0)
                epoch_text = f"评测牌局 {evaluated:,}/{evaluation_total:,}"
                backend = str(progress.get("inference_backend", "auto")).upper()
                batch_text = f"{int(progress.get('evaluation_workers', 4) or 4)}进程 {backend} 推理评测"
            losses = progress.get("losses") or {}
            loss_text = "  ".join(f"{name}: {float(value):.4f}" for name, value in losses.items())
            self.training_detail_label.setText(
                f"模型 {progress.get('model_index', '—')}/{progress.get('total_models', 6)} · "
                f"{str(progress.get('backbone', '—')).upper()} · Seed {progress.get('seed', '—')}\n"
                f"{phase_names.get(progress.get('phase'), progress.get('phase', '—'))}\n"
                f"{epoch_text}  {batch_text}\n{loss_text}\n"
                f"牌局 → 样本编码 → 教师 → 学生 → ONNX/OpenVINO → 5万局评测"
            )
            overall = float(progress.get("overall_progress", progress.get("model_progress", 0.0)) or 0.0)
            self.model_training_progress.setValue(max(0, min(1000, int(overall * 1000))))
            self.model_training_progress.setFormat(f"模型流水线 {overall:.1%}")
        else:
            self.training_detail_label.setText(
                "当前训练由旧版本进程启动，尚未记录精确 Epoch。\n"
                "可观测状态：GPU 正在执行首个候选模型的前向计算、反向传播与参数更新。\n\n"
                "牌局 → 样本编码 → 教师 → 学生 → ONNX/OpenVINO → 5万局评测"
            )
            self.model_training_progress.setValue(0)
            self.model_training_progress.setFormat("下次训练将显示 Epoch / Batch / Loss")
        self.registry_label.setText(
            f"稳定版本\n{pointers.get('stable') or '规则模型 stable_rule_v3'}\n\n"
            f"候选版本\n{pointers.get('candidate') or '等待训练与评测'}\n\n"
            f"回滚版本\n{pointers.get('rollback') or '尚未生成'}"
        )

        self.model_table.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            mse = candidate["validation_mse"]
            teammate = candidate["teammate_accuracy"]
            values = [
                candidate["name"], candidate["backbone"], candidate["seed"],
                "—" if mse is None else f"{float(mse):.4f}",
                "—" if teammate is None else f"{float(teammate):.1%}",
            ]
            for column, value in enumerate(values):
                self.model_table.setItem(row, column, QTableWidgetItem(str(value)))

        error = metrics["error"]
        state_message = str(metrics["state"].get("message", "") or "")
        self.error_text.setPlainText(
            f"诊断状态\n训练进程：{'正常' if processes else '未运行'}\n"
            f"错误日志：{error['path'] or '无'}\n大小：{error['size']} bytes\n"
            f"{state_message}\n\n{error['tail']}"
        )
        self.updated_label.setText(f"每 2 秒自动刷新 · {metrics['updated_at']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="百将牌 AI 训练可视化监控窗口")
    parser.add_argument("--snapshot", action="store_true", help="输出一次状态快照后退出")
    args = parser.parse_args()
    if args.snapshot:
        print(json.dumps(collect_metrics(), ensure_ascii=False, indent=2))
        return 0
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 9))
    window = TrainingDashboard()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
