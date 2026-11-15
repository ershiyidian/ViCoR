#!/usr/bin/env python3
"""
RoboDojo Headless Simulator Server with 6-DoF TCP-Centric Virtual Gamepad & Proprioceptive Cockpit HUD.
Runs inside Conda `RoboDojo` (Python 3.11 + Isaac Sim 5.1 + CuRobo) on dedicated GPU,
exposing a clean HTTP gamepad/frame/state bridge for the unified GameWorld/ViCoR harness.

Interface layer (strictly zero privileged scene/object coordinates):
  Unified Harness -> Gamepad key press / streaming hold -> Fingertip TCP 6-DoF delta + CuRobo IK
  Proprioceptive Cockpit HUD -> Projects ONLY robot's own joint kinematics (TCP `+`, jaw ticks, & plumb `o`)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import queue
import importlib
import io
import json
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import sys as _sys_early

_phys_gpu = os.environ.get("ROBODOJO_PHYS_GPU")
if _phys_gpu is None:
    _phys_gpu = "1"
    for _i, _a in enumerate(_sys_early.argv):
        if _a == "--device_id" and _i + 1 < len(_sys_early.argv):
            _phys_gpu = _sys_early.argv[_i + 1]
        elif _a.startswith("--device_id="):
            _phys_gpu = _a.split("=", 1)[1]
os.environ["ROBODOJO_PHYS_GPU"] = str(_phys_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(_phys_gpu)

import sys
import threading
import time
import traceback
from typing import Any

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
ROBODOJO_ROOT = os.path.join(REPO_DIR, "RoboDojo")
if os.path.isdir(ROBODOJO_ROOT):
    os.chdir(ROBODOJO_ROOT)
    if ROBODOJO_ROOT not in sys.path:
        sys.path.insert(0, ROBODOJO_ROOT)
    xpol = os.path.join(ROBODOJO_ROOT, "XPolicyLab")
    if os.path.isdir(xpol) and xpol not in sys.path:
        sys.path.append(xpol)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RoboDojo 6-DoF Virtual-Gamepad Simulator Server")
parser.add_argument("--task_name", type=str, required=True)
parser.add_argument("--env_cfg_type", type=str, default="arx_x5")
parser.add_argument("--device_id", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--http_host", type=str, default="127.0.0.1")
parser.add_argument("--http_port", type=int, default=19091)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from env.global_configs import BENCHMARK, ROOT_DIR

task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")


def _physx_monitor_needed(task_name: str) -> bool:
    cfg_path = task_registry.task_config_path(os.path.join(ROOT_DIR, "task", BENCHMARK, "config"), task_name)
    try:
        import yaml

        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return bool(cfg.get("Articulation"))
    except Exception:
        return True


if not os.environ.get("ROBODOJO_RUN_ID"):
    os.environ["ROBODOJO_RUN_ID"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

enable_monitor = _physx_monitor_needed(args_cli.task_name)
if enable_monitor:
    from src.eval_client.physx_warning_monitor import get_monitor

    get_monitor().start(enabled=True)

from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

add_zero_delay_kit_args(args_cli)
args_cli.device = "cuda:0"
_phys_id = os.environ.get("ROBODOJO_PHYS_GPU", str(args_cli.device_id))
extra_kit = (
    " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    f" --/renderer/activeGpu={_phys_id} --/renderer/multiGpu/enabled=false"
    " --/app/hangDetector/enabled=false"
)
args_cli.kit_args = (getattr(args_cli, "kit_args", "") or "") + extra_kit

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
import transforms3d as t3d

from env.global_configs import ENV_CONFIG_PATH
from src.eval_client.eval_env import create_eval_env
from utils.cluttered_generator import UnStableError
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config, process_randomization


class _NoopModelClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def call(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


import src.eval_client.eval_env as _eval_env_mod

_eval_env_mod.WsModelClient = _NoopModelClient


# Robot self-kinematics constants (from ARX X5 URDF & robot_config.yml, zero scene/object data)
TCP_LOCAL = np.array([0.145, 0.0, 0.0], dtype=np.float64)
Z_TABLE = 0.765
Z_PLUMB_REF = Z_TABLE + 0.015
TOPDOWN_PITCH_DEG = 66.0
READY_LEFT_TCP = np.array([-0.32, -0.25, 0.89], dtype=np.float64)
READY_RIGHT_TCP = np.array([0.32, -0.25, 0.89], dtype=np.float64)

# Virtual Gamepad physical velocity constants (duration_s * velocity = displacement)
V_XY_MPS = 0.18
V_Z_MPS = 0.15
OMEGA_YAW_RPS = float(np.deg2rad(90.0))
OMEGA_PITCH_RPS = float(np.deg2rad(60.0))
OMEGA_ROLL_RPS = float(np.deg2rad(75.0))

WS_X_TCP_LIM = (-0.54, 0.54)
WS_Y_TCP_LIM = (-0.36, 0.30)
WS_Z_TCP_LIM = (Z_TABLE + 0.003, 1.15)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _qmult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return _quat_normalize(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def _quat_from_axis_angle(axis: np.ndarray | list[float], angle_rad: float) -> np.ndarray:
    if abs(angle_rad) < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a = a / n
    half = 0.5 * float(angle_rad)
    s = float(np.sin(half))
    return _quat_normalize(np.array([float(np.cos(half)), a[0] * s, a[1] * s, a[2] * s], dtype=np.float64))


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    q0n = _quat_normalize(q0)
    q1n = _quat_normalize(q1)
    dot = float(np.dot(q0n, q1n))
    if dot < 0.0:
        q1n = -q1n
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize(q0n + a * (q1n - q0n))
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_0 = float(np.sin(theta_0))
    theta = theta_0 * a
    s0 = float(np.cos(theta) - dot * np.sin(theta) / sin_0)
    s1 = float(np.sin(theta) / sin_0)
    return _quat_normalize(s0 * q0n + s1 * q1n)


def _ee_to_tcp(ee_pose: np.ndarray) -> np.ndarray:
    R = t3d.quaternions.quat2mat(ee_pose[3:7])
    return ee_pose[:3] + R @ TCP_LOCAL


LEFT_SHOULDER_XY = np.array([-0.30, -0.45], dtype=np.float64)
RIGHT_SHOULDER_XY = np.array([0.30, -0.45], dtype=np.float64)


def _natural_topdown_pitch_deg(tcp_pos: np.ndarray, is_right: bool) -> float:
    sh = RIGHT_SHOULDER_XY if is_right else LEFT_SHOULDER_XY
    r_xy = float(np.hypot(float(tcp_pos[0]) - float(sh[0]), float(tcp_pos[1]) - float(sh[1])))
    # Keep 66 deg (~86 deg orthographic vertical wrist cam) across primary workspace (r_xy <= 0.46m), gently easing only at far reach
    return float(np.clip(TOPDOWN_PITCH_DEG - max(0.0, r_xy - 0.46) * 160.0, 42.0, TOPDOWN_PITCH_DEG))


def _compose_wrist_quat(home_quat: np.ndarray, pitch_deg: float, user_yaw: float, user_roll: float) -> np.ndarray:
    q = _qmult(_quat_from_axis_angle([1.0, 0.0, 0.0], -np.deg2rad(float(pitch_deg))), home_quat)
    if abs(user_yaw) > 1e-7:
        q = _qmult(_quat_from_axis_angle([0.0, 0.0, 1.0], float(user_yaw)), q)
    if abs(user_roll) > 1e-7:
        f_axis = t3d.quaternions.quat2mat(q)[:, 0]
        q = _qmult(_quat_from_axis_angle(f_axis, float(user_roll)), q)
    return q


def _ee_to_jaws(ee_pose: np.ndarray, jaw_mm: float) -> tuple[np.ndarray, np.ndarray]:
    R = t3d.quaternions.quat2mat(ee_pose[3:7])
    half_span = float(np.clip(0.5 * (jaw_mm / 1000.0), 0.006, 0.052))
    p_a = ee_pose[:3] + R @ np.array([0.145, half_span, 0.0], dtype=np.float64)
    p_b = ee_pose[:3] + R @ np.array([0.145, -half_span, 0.0], dtype=np.float64)
    return p_a, p_b


def _tcp_to_ee_pos(tcp_pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    R = t3d.quaternions.quat2mat(quat_wxyz)
    return tcp_pos - R @ TCP_LOCAL


class RoboDojoGamepadSession:
    def __init__(self, task_name: str, env_cfg_type: str, device_id: int, seed: int) -> None:
        self.task_name = task_name
        self.env_cfg_type = env_cfg_type
        self.lock = threading.Lock()

        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, env_cfg_type + ".yml"))
        eval_cfg["task_name"] = task_name
        eval_cfg["num_envs"] = 1
        eval_cfg["device_id"] = device_id
        eval_cfg["eval_batch"] = False
        eval_cfg["policy_name"] = "vicor_gamepad"
        eval_cfg["additional_info"] = "gamepad"
        eval_cfg["seed"] = int(os.environ.get("ROBODOJO_EVAL_SET", "0"))
        eval_cfg["physx_monitor_enabled"] = enable_monitor

        deploy_cfg = {
            "policy_name": "vicor_gamepad",
            "port": args_cli.http_port,
            "host": "127.0.0.1",
            "protocol": "ws",
            "policy_server_url": f"ws://127.0.0.1:{args_cli.http_port}",
            "evaluation_id": os.environ["ROBODOJO_RUN_ID"],
            "trial_id": f"{task_name}-{os.environ['ROBODOJO_RUN_ID']}",
            "action_case_id": f"{task_name}_case",
            "repeat_index": None,
        }
        bench_path = os.path.join(ROOT_DIR, "task", BENCHMARK)
        env_cfg = OmegaConf.create(
            {
                "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
                "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
                "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")),
                "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")),
                "task_env": load_yaml(task_registry.task_config_path(os.path.join(bench_path, "config"), task_name)),
                "eval_cfg": eval_cfg,
                "deploy_cfg": deploy_cfg,
            }
        )
        OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
        env_cfg = process_randomization(env_cfg)
        env_cfg, eval_num = process_config(env_cfg, task_name=task_name)
        eval_cfg["eval_num"] = eval_num
        OmegaConf.update(
            env_cfg,
            "camera.default_frequency",
            eval_cfg["observation"].get("collect_freq", 0),
            force_add=True,
        )
        env_cfg.sim.seed = [0]

        self.env = create_eval_env(env_cfg, simulation_app, resume_state=None)
        self.env.model_client = _NoopModelClient()
        self.env._stream_vision = lambda env_idx, frame: None

        self.official_step_lim = int(self.env.step_lim)
        self.unlimited_steps = os.environ.get("ROBODOJO_UNLIMITED_STEPS", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if self.unlimited_steps:
            self.env.step_lim = 100000

        self.available_seeds = list(self.env.seed_manager.seed_list)
        if not self.available_seeds:
            self.available_seeds = [0]
        self.initial_seed = int(seed) if int(seed) in self.available_seeds else int(self.available_seeds[0])
        self.seed_cursor = self.available_seeds.index(self.initial_seed)
        self.current_layout_id = self.initial_seed
        self.multi_layout_mode = os.environ.get("ROBODOJO_MULTI_LAYOUT", "0").strip().lower() in ("1", "true", "yes")

        self.obs: dict[str, Any] = {}
        self._calib_home_left_pose: np.ndarray | None = None
        self._calib_home_right_pose: np.ndarray | None = None
        self.home_left_pose = np.zeros(7, dtype=np.float64)
        self.home_right_pose = np.zeros(7, dtype=np.float64)
        self.topdown_left_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.topdown_right_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.target_left_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.target_right_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.left_grip = 1.0
        self.right_grip = 1.0
        self.user_yaw_l = 0.0
        self.user_pitch_l = 0.0
        self.user_roll_l = 0.0
        self.user_yaw_r = 0.0
        self.user_pitch_r = 0.0
        self.user_roll_r = 0.0
        self.stance_mode = "TOPDOWN"

        self.gamepad_steps = 0
        self.sim_steps = 0
        self.max_proc_score = 0.0
        self.score_within_official_lim = 0.0
        self.last_cmd_label = "READY"
        self._last_hold_seq = -1
        self._accum_hold_dur = 0.0
        self._hold_start_l_tcp = np.zeros(3, dtype=np.float64)
        self._hold_start_r_tcp = np.zeros(3, dtype=np.float64)
        self._accum_dl = np.zeros(3, dtype=np.float64)
        self._accum_dr = np.zeros(3, dtype=np.float64)
        self._latched_l_xy = READY_LEFT_TCP[:2].copy()
        self._latched_r_xy = READY_RIGHT_TCP[:2].copy()
        self._auto_pitch_rel_l = 0.0
        self._auto_pitch_rel_r = 0.0
        self._cached_png: bytes | None = None
        self.is_paused = False
        self._clock_started = False
        self._hold_left_joints: list[float] | None = None
        self._hold_right_joints: list[float] | None = None
        self.cmd_queue: queue.Queue = queue.Queue()
        self.latest_state: dict[str, Any] = {}
        self.latest_png: bytes = b""

        self.reset_episode(advance_seed=False)

    def _enter_topdown_ready_stance_unlocked(self) -> None:
        if self.env_cfg_type != "arx_x5":
            return
        st_cur = self.obs.get("state", {})
        cur_l = np.asarray(st_cur.get("left_ee_pose", self.home_left_pose), dtype=np.float64).copy()
        cur_r = np.asarray(st_cur.get("right_ee_pose", self.home_right_pose), dtype=np.float64).copy()
        tgt_l_ee = _tcp_to_ee_pos(READY_LEFT_TCP, self.topdown_left_quat)
        tgt_r_ee = _tcp_to_ee_pos(READY_RIGHT_TCP, self.topdown_right_quat)
        n_prep = 8
        for s_idx in range(1, n_prep + 1):
            if bool(self.env.end_flag[0]):
                break
            alpha = float(s_idx) / float(n_prep)
            p_l = cur_l[:3] + alpha * (tgt_l_ee - cur_l[:3])
            p_r = cur_r[:3] + alpha * (tgt_r_ee - cur_r[:3])
            q_l = _quat_slerp(cur_l[3:7], self.topdown_left_quat, alpha)
            q_r = _quat_slerp(cur_r[3:7], self.topdown_right_quat, alpha)
            self.env.take_action(
                {
                    "left_ee_pose": np.concatenate([p_l, q_l]).tolist(),
                    "left_ee_joint_state": [1.0],
                    "right_ee_pose": np.concatenate([p_r, q_r]).tolist(),
                    "right_ee_joint_state": [1.0],
                }
            )
        try:
            sm = self.env.scene_manager
            for obj_dict in (
                sm._rigid_and_dynamic_objects[0],
                sm._articulation_objects[0],
                sm._garment_objects[0],
                sm._geometry_objects[0],
                sm._fluid_objects[0],
            ):
                for obj in obj_dict.values():
                    obj.apply_saved_pose()
            for _ in range(2):
                sm.sim.sim_step(render=False)
            sm.sim.sim_step(render=True)
            self.env.end_flag[0] = False
            self.env.run_reward()
            if hasattr(self.env, "get_score"):
                self.env.get_score()
        except Exception as _pose_exc:
            print(f"[RoboDojoSimServer] scene t=0 restore warn: {_pose_exc}", flush=True)
        self.env.take_action_cnt[0] = 0
        self.obs = self.env.get_obs()
        self.target_left_quat = self.topdown_left_quat.copy()
        self.target_right_quat = self.topdown_right_quat.copy()
        self.stance_mode = "TOPDOWN"

    def reset_episode(self, advance_seed: bool = False, explicit_seed: int | None = None, paused: bool | None = None) -> dict[str, Any]:
        with self.lock:
            if not self.multi_layout_mode:
                self.seed_cursor = self.available_seeds.index(self.initial_seed)
            elif explicit_seed is not None and int(explicit_seed) in self.available_seeds:
                self.seed_cursor = self.available_seeds.index(int(explicit_seed))
            elif advance_seed and len(self.available_seeds) > 1:
                self.seed_cursor = (self.seed_cursor + 1) % len(self.available_seeds)

            attempts = 0
            while attempts < max(5, len(self.available_seeds)):
                cand_seed = int(self.available_seeds[self.seed_cursor % len(self.available_seeds)])
                if enable_monitor:
                    from src.eval_client.physx_warning_monitor import get_monitor

                    get_monitor().reset()
                try:
                    # Hard-reset robot joint positions & velocities to default_joint_pos BEFORE scene reload/sim_step so arms never whip through conveyor/objects on Episode 2+
                    try:
                        rm0 = self.env.robot_manager
                        if getattr(rm0, "robot_key", None):
                            for idx_r, rob_r in enumerate(rm0.robot_list):
                                if rob_r.robot_type == "arm" and idx_r < len(rm0.robot_key):
                                    art = rm0.robot_key[idx_r]
                                    q_def = art.data.default_joint_pos.clone()
                                    v_def = art.data.default_joint_vel.clone()
                                    art.write_joint_state_to_sim(q_def, v_def)
                                    art.set_joint_position_target(q_def)
                                    art.set_joint_velocity_target(v_def)
                    except Exception as _re_exc:
                        print(f"[RoboDojoSimServer] pre-reset joint write warn: {_re_exc}", flush=True)
                    if self.unlimited_steps:
                        self.env.step_lim = 100000
                    self.env.reset(seed=[cand_seed])
                    if self.unlimited_steps:
                        self.env.step_lim = 100000
                    self.env.run_reward()
                    if hasattr(self.env, "get_score"):
                        self.env.get_score()
                    if getattr(self.env, "interact", False) and hasattr(self.env, "query_support_arm_traj"):
                        self.env.query_support_arm_traj(env_idx=0)
                    self.current_layout_id = cand_seed
                    break
                except (UnStableError, AttributeError, KeyError):
                    self.seed_cursor = (self.seed_cursor + 1) % len(self.available_seeds)
                    attempts += 1

            self.obs = self.env.get_obs()
            st = self.obs.get("state", {})
            if self._calib_home_left_pose is None or self._calib_home_right_pose is None:
                self._calib_home_left_pose = np.asarray(st["left_ee_pose"], dtype=np.float64).copy()
                self._calib_home_right_pose = np.asarray(st["right_ee_pose"], dtype=np.float64).copy()
            self.home_left_pose = self._calib_home_left_pose.copy()
            self.home_right_pose = self._calib_home_right_pose.copy()

            self.user_yaw_l = -0.5 * float(np.pi)
            self.user_yaw_r = 0.5 * float(np.pi)
            self.topdown_left_quat = _compose_wrist_quat(self.home_left_pose[3:7], TOPDOWN_PITCH_DEG, self.user_yaw_l, 0.0)
            self.topdown_right_quat = _compose_wrist_quat(self.home_right_pose[3:7], TOPDOWN_PITCH_DEG, self.user_yaw_r, 0.0)

            self.left_grip = 1.0
            self.right_grip = 1.0
            self.user_pitch_l = 0.0
            self.user_roll_l = 0.0
            self.user_pitch_r = 0.0
            self.user_roll_r = 0.0
            self.gamepad_steps = 0
            self.sim_steps = 0
            self.max_proc_score = 0.0
            self.score_within_official_lim = 0.0
            self._last_hold_seq = -1
            self._accum_hold_dur = 0.0
            self._accum_dl = np.zeros(3, dtype=np.float64)
            self._accum_dr = np.zeros(3, dtype=np.float64)
            self._auto_pitch_rel_l = 0.0
            self._auto_pitch_rel_r = 0.0

            self._enter_topdown_ready_stance_unlocked()
            st_post = self.obs.get("state", {})
            self._latched_l_xy = _ee_to_tcp(np.asarray(st_post.get("left_ee_pose", self.home_left_pose), dtype=np.float64))[:2].copy()
            self._latched_r_xy = _ee_to_tcp(np.asarray(st_post.get("right_ee_pose", self.home_right_pose), dtype=np.float64))[:2].copy()

            if paused is not None:
                self.is_paused = bool(paused)
            self._latch_hold_joints_unlocked()
            self._clock_started = False
            self.last_cmd_label = f"READY_TOPDOWN (layout={self.current_layout_id})"
            self._cached_png = None
            return self._refresh_snapshots_unlocked()

    def _read_jaw_mm_unlocked(self) -> tuple[float, float]:
        try:
            rm = self.env.robot_manager
            r_l, r_r = rm.robot_list[0], rm.robot_list[1]
            jl = np.asarray(rm.get_end_effector_real_val(r_l, env_idx_list=[0])[0], dtype=np.float64).reshape(-1)
            jr = np.asarray(rm.get_end_effector_real_val(r_r, env_idx_list=[0])[0], dtype=np.float64).reshape(-1)
            mm_l = max(0.0, float(np.sum(jl)) * 1000.0)
            mm_r = max(0.0, float(np.sum(jr)) * 1000.0)
            return mm_l, mm_r
        except Exception as exc:
            print(f"[RoboDojoSimServer] jaw read warn: {exc}", flush=True)
            return (88.0 if self.left_grip >= 0.5 else 25.0), (88.0 if self.right_grip >= 0.5 else 25.0)

    def _read_process_score_unlocked(self) -> float:
        try:
            scores = self.env.reward_manager.get_score()
            if scores is not None and len(scores) > 0:
                return float(scores[0]) / 100.0
        except Exception:
            pass
        return 0.0

    def _build_state_unlocked(self) -> dict[str, Any]:
        end_flag = bool(self.env.end_flag[0])
        is_win = bool(end_flag and self.env.success[0])
        is_fail = bool(end_flag and not self.env.success[0])
        proc_score = self._read_process_score_unlocked()
        if proc_score > self.max_proc_score:
            self.max_proc_score = proc_score
        effective_score = 1.0 if is_win else max(self.max_proc_score, proc_score)
        if self.sim_steps <= self.official_step_lim and effective_score > self.score_within_official_lim:
            self.score_within_official_lim = effective_score
        instruction = str(self.obs.get("instruction", "") or "")

        st = self.obs.get("state", {})
        cur_l = np.asarray(st.get("left_ee_pose", self.home_left_pose), dtype=np.float64)
        cur_r = np.asarray(st.get("right_ee_pose", self.home_right_pose), dtype=np.float64)
        l_tcp = _ee_to_tcp(cur_l)
        r_tcp = _ee_to_tcp(cur_r)

        return {
            "status": "win" if is_win else ("gameover" if is_fail else "playing"),
            "is_actionable": not end_flag,
            "seed": int(self.current_layout_id),
            "score": round(float(effective_score), 4),
            "instruction": instruction,
            "game_state": {
                "score": round(float(effective_score), 4),
                "process_score": round(float(proc_score), 4),
                "max_process_score": round(float(self.max_proc_score), 4),
                "score_within_official_lim": round(float(self.score_within_official_lim), 4),
                "success": bool(is_win),
                "ended": bool(end_flag),
                "gamepad_steps": int(self.gamepad_steps),
                "sim_steps": int(self.sim_steps),
                "official_step_lim": int(self.official_step_lim),
                "step_lim": int(self.env.step_lim),
                "layout_id": int(self.current_layout_id),
                "stance_mode": self.stance_mode,
                "left_gripper_open": bool(self.left_grip >= 0.5),
                "right_gripper_open": bool(self.right_grip >= 0.5),
                "left_jaw_mm": round(float(self._read_jaw_mm_unlocked()[0]), 1),
                "right_jaw_mm": round(float(self._read_jaw_mm_unlocked()[1]), 1),
                "left_tcp_x_cm": round(float(l_tcp[0] * 100.0), 1),
                "left_tcp_y_cm": round(float(l_tcp[1] * 100.0), 1),
                "left_tcp_z_cm": round(float((l_tcp[2] - Z_TABLE) * 100.0), 1),
                "right_tcp_x_cm": round(float(r_tcp[0] * 100.0), 1),
                "right_tcp_y_cm": round(float(r_tcp[1] * 100.0), 1),
                "right_tcp_z_cm": round(float((r_tcp[2] - Z_TABLE) * 100.0), 1),
                "last_cmd": self.last_cmd_label,
            },
            "terminal": {
                "isTerminal": bool(end_flag),
                "outcome": "success" if is_win else ("fail" if is_fail else None),
            },
        }

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            return self._build_state_unlocked()

    def _run_ee_trajectory_unlocked(
        self,
        start_l_ee: np.ndarray,
        end_l_ee: np.ndarray,
        start_l_q: np.ndarray,
        end_l_q: np.ndarray,
        start_r_ee: np.ndarray,
        end_r_ee: np.ndarray,
        start_r_q: np.ndarray,
        end_r_q: np.ndarray,
        n_sub: int,
    ) -> None:
        for s_idx in range(1, n_sub + 1):
            if bool(self.env.end_flag[0]):
                break
            alpha = float(s_idx) / float(n_sub)
            p_l = start_l_ee + alpha * (end_l_ee - start_l_ee)
            p_r = start_r_ee + alpha * (end_r_ee - start_r_ee)
            q_l = _quat_slerp(start_l_q, end_l_q, alpha)
            q_r = _quat_slerp(start_r_q, end_r_q, alpha)
            action = {
                "left_ee_pose": np.concatenate([p_l, q_l]).tolist(),
                "left_ee_joint_state": [float(self.left_grip)],
                "right_ee_pose": np.concatenate([p_r, q_r]).tolist(),
                "right_ee_joint_state": [float(self.right_grip)],
            }
            self.env.take_action(action)
            self.sim_steps += 1

    def _resolve_reachable_wrist_quat_unlocked(
        self,
        arm_idx: int,
        tgt_tcp: np.ndarray,
        home_quat: np.ndarray,
        desired_pitch_deg: float,
        user_yaw: float,
        user_roll: float,
    ) -> np.ndarray:
        q_pref = _compose_wrist_quat(home_quat, desired_pitch_deg, user_yaw, user_roll)
        try:
            rm = self.env.robot_manager
            robot = rm.robot_list[arm_idx]
            ee_pref = _tcp_to_ee_pos(tgt_tcp, q_pref)
            res = rm.solve_ik(target_pose=np.concatenate([ee_pref, q_pref]).tolist(), env_idx=0, robot=robot)
            if isinstance(res, dict) and res.get("status") == "Success":
                return q_pref
            for p_deg in range(int(desired_pitch_deg) - 6, 8, -6):
                q_cand = _compose_wrist_quat(home_quat, float(p_deg), user_yaw, user_roll)
                ee_cand = _tcp_to_ee_pos(tgt_tcp, q_cand)
                res_c = rm.solve_ik(target_pose=np.concatenate([ee_cand, q_cand]).tolist(), env_idx=0, robot=robot)
                if isinstance(res_c, dict) and res_c.get("status") == "Success":
                    return q_cand
        except Exception:
            pass
        return q_pref

    def step_gamepad(self, keys: list[str], duration_s: float, hold_seq: int = 0) -> dict[str, Any]:
        with self.lock:
            if bool(self.env.end_flag[0]):
                return self._build_state_unlocked()

            norm_keys = [str(k).strip() for k in (keys or []) if str(k).strip()]
            if not norm_keys:
                return self._build_state_unlocked()

            dt_hold = float(np.clip(float(duration_s), 0.03, 1.50))
            st = self.obs.get("state", {})
            cur_l = np.asarray(st["left_ee_pose"], dtype=np.float64).copy()
            cur_r = np.asarray(st["right_ee_pose"], dtype=np.float64).copy()
            cur_l_tcp = _ee_to_tcp(cur_l)
            cur_r_tcp = _ee_to_tcp(cur_r)

            is_continuation = (hold_seq > 0) and (hold_seq == self._last_hold_seq)
            if not is_continuation:
                self._last_hold_seq = int(hold_seq)
                self._accum_hold_dur = dt_hold
                self._hold_start_l_tcp = cur_l_tcp.copy()
                self._hold_start_r_tcp = cur_r_tcp.copy()
                self._accum_dl = np.zeros(3, dtype=np.float64)
                self._accum_dr = np.zeros(3, dtype=np.float64)
                self.gamepad_steps += 1
            else:
                self._accum_hold_dur += dt_hold

            dl = np.zeros(3, dtype=np.float64)
            dr = np.zeros(3, dtype=np.float64)
            dyaw_l = 0.0
            dpitch_l = 0.0
            droll_l = 0.0
            dyaw_r = 0.0
            dpitch_r = 0.0
            droll_r = 0.0
            toggled_grip = False
            go_origin_home = False
            go_topdown = False

            for k in norm_keys:
                kl = k.lower()
                if kl == "w":
                    dl[1] += V_XY_MPS * dt_hold
                elif kl == "s":
                    dl[1] -= V_XY_MPS * dt_hold
                elif kl == "a":
                    dl[0] -= V_XY_MPS * dt_hold
                elif kl == "d":
                    dl[0] += V_XY_MPS * dt_hold
                elif kl == "q":
                    dl[2] += V_Z_MPS * dt_hold
                elif kl == "e":
                    dl[2] -= V_Z_MPS * dt_hold
                elif kl == "z":
                    dyaw_l += OMEGA_YAW_RPS * dt_hold
                elif kl == "c":
                    dyaw_l -= OMEGA_YAW_RPS * dt_hold
                elif kl == "t":
                    dpitch_l -= OMEGA_PITCH_RPS * dt_hold
                elif kl == "g":
                    dpitch_l += OMEGA_PITCH_RPS * dt_hold
                elif kl == "r":
                    droll_l += OMEGA_ROLL_RPS * dt_hold
                elif kl == "f":
                    droll_l -= OMEGA_ROLL_RPS * dt_hold
                elif kl == "space":
                    self.left_grip = 0.0 if self.left_grip >= 0.5 else 1.0
                    toggled_grip = True
                elif kl in ("1", "f1"):
                    self.left_grip = 0.0
                    toggled_grip = True
                elif kl in ("2", "f2"):
                    self.left_grip = 1.0
                    toggled_grip = True
                elif kl in ("arrowup", "up"):
                    dr[1] += V_XY_MPS * dt_hold
                elif kl in ("arrowdown", "down"):
                    dr[1] -= V_XY_MPS * dt_hold
                elif kl in ("arrowleft", "left"):
                    dr[0] -= V_XY_MPS * dt_hold
                elif kl in ("arrowright", "right"):
                    dr[0] += V_XY_MPS * dt_hold
                elif kl == "i":
                    dr[2] += V_Z_MPS * dt_hold
                elif kl == "k":
                    dr[2] -= V_Z_MPS * dt_hold
                elif kl == "n":
                    dyaw_r += OMEGA_YAW_RPS * dt_hold
                elif kl == "m":
                    dyaw_r -= OMEGA_YAW_RPS * dt_hold
                elif kl == "y":
                    dpitch_r -= OMEGA_PITCH_RPS * dt_hold
                elif kl == "b":
                    dpitch_r += OMEGA_PITCH_RPS * dt_hold
                elif kl == "u":
                    droll_r += OMEGA_ROLL_RPS * dt_hold
                elif kl == "j":
                    droll_r -= OMEGA_ROLL_RPS * dt_hold
                elif kl in ("enter", "return"):
                    self.right_grip = 0.0 if self.right_grip >= 0.5 else 1.0
                    toggled_grip = True
                elif kl in ("3", "f3"):
                    self.right_grip = 0.0
                    toggled_grip = True
                elif kl in ("4", "f4"):
                    self.right_grip = 1.0
                    toggled_grip = True
                elif kl == "v":
                    go_topdown = True
                elif kl in ("h", "backspace"):
                    go_origin_home = True
                elif kl == "x":
                    pass

            if go_origin_home:
                lift_l_ee = cur_l[:3].copy()
                lift_r_ee = cur_r[:3].copy()
                lift_l_ee[2] = max(lift_l_ee[2], 0.96)
                lift_r_ee[2] = max(lift_r_ee[2], 0.96)
                self._run_ee_trajectory_unlocked(
                    cur_l[:3], lift_l_ee, cur_l[3:7], cur_l[3:7],
                    cur_r[:3], lift_r_ee, cur_r[3:7], cur_r[3:7],
                    n_sub=4,
                )
                self.target_left_quat = self.home_left_pose[3:7].copy()
                self.target_right_quat = self.home_right_pose[3:7].copy()
                self._run_ee_trajectory_unlocked(
                    lift_l_ee, self.home_left_pose[:3], cur_l[3:7], self.target_left_quat,
                    lift_r_ee, self.home_right_pose[:3], cur_r[3:7], self.target_right_quat,
                    n_sub=7,
                )
                self._run_ee_trajectory_unlocked(
                    self.home_left_pose[:3], self.home_left_pose[:3], self.target_left_quat, self.target_left_quat,
                    self.home_right_pose[:3], self.home_right_pose[:3], self.target_right_quat, self.target_right_quat,
                    n_sub=2,
                )
                self.stance_mode = "ORIGIN"
            elif go_topdown:
                self.user_yaw_l = -0.5 * float(np.pi)
                self.user_pitch_l = self.user_roll_l = 0.0
                self.user_yaw_r = 0.5 * float(np.pi)
                self.user_pitch_r = self.user_roll_r = 0.0
                self.target_left_quat = self.topdown_left_quat.copy()
                self.target_right_quat = self.topdown_right_quat.copy()
                tgt_l_ee = _tcp_to_ee_pos(READY_LEFT_TCP, self.target_left_quat)
                tgt_r_ee = _tcp_to_ee_pos(READY_RIGHT_TCP, self.target_right_quat)
                lift_l_ee = cur_l[:3].copy()
                lift_r_ee = cur_r[:3].copy()
                lift_l_ee[2] = max(lift_l_ee[2], 0.92)
                lift_r_ee[2] = max(lift_r_ee[2], 0.92)
                self._run_ee_trajectory_unlocked(
                    cur_l[:3], lift_l_ee, cur_l[3:7], cur_l[3:7],
                    cur_r[:3], lift_r_ee, cur_r[3:7], cur_r[3:7],
                    n_sub=3,
                )
                self._run_ee_trajectory_unlocked(
                    lift_l_ee, tgt_l_ee, cur_l[3:7], self.target_left_quat,
                    lift_r_ee, tgt_r_ee, cur_r[3:7], self.target_right_quat,
                    n_sub=5,
                )
                self.stance_mode = "TOPDOWN"
            else:
                has_motion = (
                    float(np.linalg.norm(dl))
                    + float(np.linalg.norm(dr))
                    + abs(dyaw_l) + abs(dpitch_l) + abs(droll_l)
                    + abs(dyaw_r) + abs(dpitch_r) + abs(droll_r)
                ) > 1e-7
                if self.stance_mode == "ORIGIN" and has_motion:
                    tgt_l_ee0 = _tcp_to_ee_pos(READY_LEFT_TCP, self.topdown_left_quat)
                    tgt_r_ee0 = _tcp_to_ee_pos(READY_RIGHT_TCP, self.topdown_right_quat)
                    self._run_ee_trajectory_unlocked(
                        cur_l[:3], tgt_l_ee0, cur_l[3:7], self.topdown_left_quat,
                        cur_r[:3], tgt_r_ee0, cur_r[3:7], self.topdown_right_quat,
                        n_sub=5,
                    )
                    self.obs = self.env.get_obs()
                    st = self.obs.get("state", {})
                    cur_l = np.asarray(st["left_ee_pose"], dtype=np.float64).copy()
                    cur_r = np.asarray(st["right_ee_pose"], dtype=np.float64).copy()
                    cur_l_tcp = _ee_to_tcp(cur_l)
                    cur_r_tcp = _ee_to_tcp(cur_r)
                    self._hold_start_l_tcp = cur_l_tcp.copy()
                    self._hold_start_r_tcp = cur_r_tcp.copy()
                    self.target_left_quat = self.topdown_left_quat.copy()
                    self.target_right_quat = self.topdown_right_quat.copy()
                    self.stance_mode = "TOPDOWN"

                self._accum_dl += dl
                self._accum_dr += dr
                l_xy_active = (abs(dl[0]) + abs(dl[1])) > 1e-6
                r_xy_active = (abs(dr[0]) + abs(dr[1])) > 1e-6

                if l_xy_active:
                    base_lx = self._hold_start_l_tcp[0] + self._accum_dl[0] if is_continuation else cur_l_tcp[0] + dl[0]
                    base_ly = self._hold_start_l_tcp[1] + self._accum_dl[1] if is_continuation else cur_l_tcp[1] + dl[1]
                else:
                    base_lx = float(np.clip(self._latched_l_xy[0], cur_l_tcp[0] - 0.022, cur_l_tcp[0] + 0.022))
                    base_ly = float(np.clip(self._latched_l_xy[1], cur_l_tcp[1] - 0.022, cur_l_tcp[1] + 0.022))

                if r_xy_active:
                    base_rx = self._hold_start_r_tcp[0] + self._accum_dr[0] if is_continuation else cur_r_tcp[0] + dr[0]
                    base_ry = self._hold_start_r_tcp[1] + self._accum_dr[1] if is_continuation else cur_r_tcp[1] + dr[1]
                else:
                    base_rx = float(np.clip(self._latched_r_xy[0], cur_r_tcp[0] - 0.022, cur_r_tcp[0] + 0.022))
                    base_ry = float(np.clip(self._latched_r_xy[1], cur_r_tcp[1] - 0.022, cur_r_tcp[1] + 0.022))

                base_lz = (self._hold_start_l_tcp[2] + self._accum_dl[2]) if (is_continuation and abs(dl[2]) > 1e-6) else (cur_l_tcp[2] + dl[2])
                base_rz = (self._hold_start_r_tcp[2] + self._accum_dr[2]) if (is_continuation and abs(dr[2]) > 1e-6) else (cur_r_tcp[2] + dr[2])

                tgt_l_tcp = np.array(
                    [
                        np.clip(base_lx, *WS_X_TCP_LIM),
                        np.clip(base_ly, *WS_Y_TCP_LIM),
                        np.clip(base_lz, *WS_Z_TCP_LIM),
                    ],
                    dtype=np.float64,
                )
                tgt_r_tcp = np.array(
                    [
                        np.clip(base_rx, *WS_X_TCP_LIM),
                        np.clip(base_ry, *WS_Y_TCP_LIM),
                        np.clip(base_rz, *WS_Z_TCP_LIM),
                    ],
                    dtype=np.float64,
                )

                if (tgt_l_tcp[2] - Z_TABLE) > 0.065:
                    self._auto_pitch_rel_l = 0.0
                if (tgt_r_tcp[2] - Z_TABLE) > 0.065:
                    self._auto_pitch_rel_r = 0.0

                self.user_yaw_l += dyaw_l
                self.user_pitch_l = float(np.clip(self.user_pitch_l - dpitch_l, -np.deg2rad(35.0), np.deg2rad(25.0)))
                self.user_roll_l += droll_l
                self.user_yaw_r += dyaw_r
                self.user_pitch_r = float(np.clip(self.user_pitch_r - dpitch_r, -np.deg2rad(35.0), np.deg2rad(25.0)))
                self.user_roll_r += droll_r

                eff_pitch_l = float(np.clip(_natural_topdown_pitch_deg(tgt_l_tcp, False) + np.rad2deg(self.user_pitch_l), 18.0, 80.0))
                eff_pitch_r = float(np.clip(_natural_topdown_pitch_deg(tgt_r_tcp, True) + np.rad2deg(self.user_pitch_r), 18.0, 80.0))
                if (abs(dl[0]) + abs(dl[1]) + abs(dl[2]) + abs(dyaw_l) + abs(dpitch_l) + abs(droll_l)) > 1e-7:
                    self.target_left_quat = self._resolve_reachable_wrist_quat_unlocked(
                        0, tgt_l_tcp, self.home_left_pose[3:7], eff_pitch_l, self.user_yaw_l, self.user_roll_l
                    )
                if (abs(dr[0]) + abs(dr[1]) + abs(dr[2]) + abs(dyaw_r) + abs(dpitch_r) + abs(droll_r)) > 1e-7:
                    self.target_right_quat = self._resolve_reachable_wrist_quat_unlocked(
                        1, tgt_r_tcp, self.home_right_pose[3:7], eff_pitch_r, self.user_yaw_r, self.user_roll_r
                    )

                req_l_disp = float(np.linalg.norm(tgt_l_tcp - cur_l_tcp))
                req_r_disp = float(np.linalg.norm(tgt_r_tcp - cur_r_tcp))
                max_disp = max(req_l_disp, req_r_disp)
                max_rot = max(
                    abs(dyaw_l) + abs(dpitch_l) + abs(droll_l),
                    abs(dyaw_r) + abs(dpitch_r) + abs(droll_r),
                )
                carrying = (self.left_grip < 0.5) or (self.right_grip < 0.5)
                step_cap = 0.008 if carrying else 0.018
                n_sub = max(1, min(12, int(np.ceil(max(max_disp / step_cap, max_rot / 0.18)))))
                if toggled_grip and n_sub < 7:
                    n_sub = 7

                l_active = (abs(dl[0]) + abs(dl[1]) + abs(dl[2]) + abs(dyaw_l) + abs(dpitch_l) + abs(droll_l)) > 1e-7
                r_active = (abs(dr[0]) + abs(dr[1]) + abs(dr[2]) + abs(dyaw_r) + abs(dpitch_r) + abs(droll_r)) > 1e-7
                for s_idx in range(1, n_sub + 1):
                    if bool(self.env.end_flag[0]):
                        break
                    alpha = float(s_idx) / float(n_sub)
                    step_l_tcp = cur_l_tcp + alpha * (tgt_l_tcp - cur_l_tcp)
                    step_r_tcp = cur_r_tcp + alpha * (tgt_r_tcp - cur_r_tcp)
                    q_l = _quat_slerp(cur_l[3:7], self.target_left_quat, alpha)
                    q_r = _quat_slerp(cur_r[3:7], self.target_right_quat, alpha)
                    if l_active:
                        q_l = self._resolve_reachable_wrist_quat_unlocked(
                            0, step_l_tcp, self.home_left_pose[3:7], eff_pitch_l, self.user_yaw_l, self.user_roll_l
                        )
                    if r_active:
                        q_r = self._resolve_reachable_wrist_quat_unlocked(
                            1, step_r_tcp, self.home_right_pose[3:7], eff_pitch_r, self.user_yaw_r, self.user_roll_r
                        )
                    step_l_ee = _tcp_to_ee_pos(step_l_tcp, q_l)
                    step_r_ee = _tcp_to_ee_pos(step_r_tcp, q_r)
                    action = {
                        "left_ee_pose": np.concatenate([step_l_ee, q_l]).tolist(),
                        "left_ee_joint_state": [float(self.left_grip)],
                        "right_ee_pose": np.concatenate([step_r_ee, q_r]).tolist(),
                        "right_ee_joint_state": [float(self.right_grip)],
                    }
                    self.env.take_action(action)
                    self.sim_steps += 1

            self.obs = self.env.get_obs()
            post_st = self.obs.get("state", {})
            post_l_tcp = _ee_to_tcp(np.asarray(post_st.get("left_ee_pose", cur_l), dtype=np.float64))
            post_r_tcp = _ee_to_tcp(np.asarray(post_st.get("right_ee_pose", cur_r), dtype=np.float64))
            if go_origin_home or go_topdown or (abs(dl[0]) + abs(dl[1]) > 1e-6):
                self._latched_l_xy = post_l_tcp[:2].copy()
            if go_origin_home or go_topdown or (abs(dr[0]) + abs(dr[1]) > 1e-6):
                self._latched_r_xy = post_r_tcp[:2].copy()
            if go_topdown:
                self._auto_pitch_rel_l = 0.0
                self._auto_pitch_rel_r = 0.0
            if dl[2] < -0.006 and (post_l_tcp[2] - Z_TABLE) > 0.022 and (post_l_tcp[2] - cur_l_tcp[2]) > 0.35 * dl[2]:
                self._auto_pitch_rel_l = max(-22.0, self._auto_pitch_rel_l - 8.0)
            if dr[2] < -0.006 and (post_r_tcp[2] - Z_TABLE) > 0.022 and (post_r_tcp[2] - cur_r_tcp[2]) > 0.35 * dr[2]:
                self._auto_pitch_rel_r = max(-22.0, self._auto_pitch_rel_r - 8.0)
            d_l = post_l_tcp - self._hold_start_l_tcp
            d_r = post_r_tcp - self._hold_start_r_tcp
            cmd_r_mag = float(np.linalg.norm(dr)) + abs(dyaw_r) + abs(dpitch_r) + abs(droll_r)
            cmd_l_mag = float(np.linalg.norm(dl)) + abs(dyaw_l) + abs(dpitch_l) + abs(droll_l)
            use_r = cmd_r_mag >= cmd_l_mag
            d_vec = d_r if use_r else d_l
            dx_cm = float(d_vec[0]) * 100.0
            dy_cm = float(d_vec[1]) * 100.0
            dz_cm = float(d_vec[2]) * 100.0

            stall_tag = ""
            if not (toggled_grip or go_origin_home or go_topdown):
                cmd_trans = float(np.linalg.norm(dr if use_r else dl))
                tick_trans = float(np.linalg.norm((post_r_tcp - cur_r_tcp) if use_r else (post_l_tcp - cur_l_tcp)))
                if cmd_trans > 0.006 and tick_trans < 0.25 * cmd_trans:
                    if use_r and post_r_tcp[0] < -0.04:
                        stall_tag = " !IK_LIM(use L-ARM WASD)"
                    elif (not use_r) and post_l_tcp[0] > 0.04:
                        stall_tag = " !IK_LIM(use R-ARM Arrows)"
                    else:
                        stall_tag = " !IK_LIM(press V or other arm)"
                else:
                    act_xy_cmd = float(np.linalg.norm((dr[:2] if use_r else dl[:2])))
                    act_z_cm = float(((post_r_tcp[2] if use_r else post_l_tcp[2]) - Z_TABLE) * 100.0)
                    act_grip_open = (self.right_grip >= 0.5) if use_r else (self.left_grip >= 0.5)
                    if act_xy_cmd > 0.004 and act_z_cm < 3.8 and act_grip_open:
                        stall_tag = " !LOW_Z(lift>6cm before XY!)"

            key_str = "+".join(norm_keys)
            if toggled_grip or go_origin_home or go_topdown:
                self.last_cmd_label = f"{key_str}"
            else:
                self.last_cmd_label = (
                    f"{key_str} ({self._accum_hold_dur:.2f}s dX:{dx_cm:+.1f} dY:{dy_cm:+.1f} dZ:{dz_cm:+.1f}cm){stall_tag}"
                )
            self._latch_hold_joints_unlocked()
            self._cached_png = None
            return self._refresh_snapshots_unlocked()

    def _project_points_in_cam(self, cam_name: str, pts_world: np.ndarray) -> list[tuple[float, float, bool]]:
        try:
            names = self.env.camera_manager.camera_names[0]
            if cam_name not in names:
                return [(0.0, 0.0, False) for _ in range(len(pts_world))]
            cam_obj = self.env.camera_manager.cameras[0][names.index(cam_name)]
            K = cam_obj.get_intrinsics_matrix().detach().cpu().numpy().astype(np.float64)
            V = cam_obj.get_view_matrix_ros()[:3, :].detach().cpu().numpy().astype(np.float64)
            out: list[tuple[float, float, bool]] = []
            for pt in pts_world:
                hom = np.array([float(pt[0]), float(pt[1]), float(pt[2]), 1.0], dtype=np.float64)
                p_cam = V @ hom
                zc = float(p_cam[2])
                if zc <= 0.03:
                    out.append((0.0, 0.0, False))
                    continue
                uvw = K @ p_cam
                out.append((float(uvw[0] / zc), float(uvw[1] / zc), True))
            return out
        except Exception:
            return [(0.0, 0.0, False) for _ in range(len(pts_world))]

    def render_png(self) -> bytes:
        with self.lock:
            if self._cached_png is not None:
                return self._cached_png

            vis = self.obs.get("vision", {})

            def _get_img_and_shape(cam_key: str, fallback_wh: tuple[int, int]) -> tuple[Image.Image, int, int]:
                arr = vis.get(cam_key, {}).get("color")
                if arr is not None:
                    arr_np = np.asarray(arr, dtype=np.uint8)
                    if arr_np.ndim == 3 and arr_np.shape[2] >= 3:
                        h0, w0 = arr_np.shape[0], arr_np.shape[1]
                        return Image.fromarray(arr_np[:, :, :3]), w0, h0
                return Image.new("RGB", fallback_wh, (24, 24, 28)), fallback_wh[0], fallback_wh[1]

            head_raw, hw0, hh0 = _get_img_and_shape("cam_head", (640, 480))
            lw_raw, lw0, lh0 = _get_img_and_shape("cam_left_wrist", (640, 480))
            rw_raw, rw0, rh0 = _get_img_and_shape("cam_right_wrist", (640, 480))

            head_img = head_raw.resize((640, 420), Image.Resampling.BILINEAR)
            lw_img = lw_raw.resize((318, 226), Image.Resampling.BILINEAR)
            rw_img = rw_raw.resize((318, 226), Image.Resampling.BILINEAR)

            canvas = Image.new("RGB", (640, 720), (18, 20, 26))
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 11)
            except Exception:
                font = ImageFont.load_default()

            st = self.obs.get("state", {})
            cur_l = np.asarray(st.get("left_ee_pose", self.home_left_pose), dtype=np.float64)
            cur_r = np.asarray(st.get("right_ee_pose", self.home_right_pose), dtype=np.float64)
            l_tcp = _ee_to_tcp(cur_l)
            r_tcp = _ee_to_tcp(cur_r)
            l_jaw_mm, r_jaw_mm = self._read_jaw_mm_unlocked()
            l_ja, l_jb = _ee_to_jaws(cur_l, l_jaw_mm)
            r_ja, r_jb = _ee_to_jaws(cur_r, r_jaw_mm)
            l_is_held = (self.left_grip < 0.5) and (l_jaw_mm >= 10.0)
            r_is_held = (self.right_grip < 0.5) and (r_jaw_mm >= 10.0)
            l_is_empty_clsd = (self.left_grip < 0.5) and (l_jaw_mm < 8.0)
            r_is_empty_clsd = (self.right_grip < 0.5) and (r_jaw_mm < 8.0)
            l_plumb = np.array([l_tcp[0], l_tcp[1], min(float(l_tcp[2]), Z_PLUMB_REF)], dtype=np.float64)
            r_plumb = np.array([r_tcp[0], r_tcp[1], min(float(r_tcp[2]), Z_PLUMB_REF)], dtype=np.float64)
            l_x_cm, l_y_cm, l_z_cm = float(l_tcp[0]) * 100.0, float(l_tcp[1]) * 100.0, (float(l_tcp[2]) - Z_TABLE) * 100.0
            r_x_cm, r_y_cm, r_z_cm = float(r_tcp[0]) * 100.0, float(r_tcp[1]) * 100.0, (float(r_tcp[2]) - Z_TABLE) * 100.0

            instr = str(self.obs.get("instruction", "") or f"Task: {self.task_name}")
            draw.rectangle([0, 0, 639, 39], fill=(26, 30, 40))
            line1 = instr[:96]
            line2 = instr[96:192]
            draw.text((8, 4), f"GOAL: {line1}", fill=(240, 244, 255), font=font)
            if line2:
                draw.text((8, 20), f"      {line2}", fill=(210, 220, 240), font=font)

            # Draw calibrated Perimeter Ruler Ticks ONLY on outer workspace margins so interior RGB objects remain 100% unoccluded
            grid_rgba = Image.new("RGBA", (640, 420), (0, 0, 0, 0))
            gdraw = ImageDraw.Draw(grid_rgba)
            sx_h, sy_h = 640.0 / hw0, 420.0 / hh0
            seg_pts: list[list[float]] = []
            seg_meta: list[tuple[int, int, int, int]] = []
            for x_cm in range(-45, 46, 5):
                xm = x_cm * 0.01
                col = (255, 232, 120, 160) if x_cm == 0 else ((185, 230, 255, 130) if x_cm % 10 == 0 else (175, 220, 255, 75))
                # Front border tick only (Y=-35..-32.5cm)
                seg_pts.append([xm, -0.355, Z_TABLE])
                seg_pts.append([xm, -0.330, Z_TABLE])
                seg_meta.append(col)
            for y_cm in range(-35, 16, 5):
                ym = y_cm * 0.01
                col = (185, 230, 255, 130) if y_cm % 10 == 0 else (175, 220, 255, 75)
                # Left & Right margin ticks only (X=-47..-44.5cm and X=+44.5..+47cm)
                seg_pts.append([-0.470, ym, Z_TABLE])
                seg_pts.append([-0.445, ym, Z_TABLE])
                seg_meta.append(col)
                seg_pts.append([0.445, ym, Z_TABLE])
                seg_pts.append([0.470, ym, Z_TABLE])
                seg_meta.append(col)
            lbl_items: list[tuple[str, list[float]]] = []
            for x_cm in (-40, -30, -20, -10, 0, 10, 20, 30, 40):
                lbl_items.append((f"{x_cm:+d}" if x_cm != 0 else "X0", [x_cm * 0.01, -0.345, Z_TABLE]))
            for y_cm in (-30, -20, -10, -5, 0, 5, 10):
                ytxt = f"Y{y_cm:+d}" if y_cm != 0 else "Y0"
                lbl_items.append((ytxt, [-0.485, y_cm * 0.01, Z_TABLE]))
                lbl_items.append((ytxt, [0.455, y_cm * 0.01, Z_TABLE]))
            all_gpts = np.array(seg_pts + [it[1] for it in lbl_items], dtype=np.float64)
            gproj = self._project_points_in_cam("cam_head", all_gpts)
            n_seg = len(seg_meta)
            for idx_s, col in enumerate(seg_meta):
                u0, v0, ok0 = gproj[2 * idx_s]
                u1, v1, ok1 = gproj[2 * idx_s + 1]
                if ok0 and ok1:
                    gdraw.line(
                        [(int(round(u0 * sx_h)), int(round(v0 * sy_h))), (int(round(u1 * sx_h)), int(round(v1 * sy_h)))],
                        fill=col,
                        width=1,
                    )
            for idx_l, (txt_l, _) in enumerate(lbl_items):
                ul, vl, okl = gproj[2 * n_seg + idx_l]
                if okl:
                    px_l, py_l = int(round(ul * sx_h)) - 8, int(round(vl * sy_h)) - 5
                    if 4 <= px_l <= 614 and 20 <= py_l <= 406:
                        gdraw.rectangle([px_l - 1, py_l - 1, px_l + len(txt_l) * 6 + 1, py_l + 10], fill=(12, 16, 24, 130))
                        gdraw.text((px_l, py_l), txt_l, fill=(225, 240, 255, 190), font=font)
            head_img = Image.alpha_composite(head_img.convert("RGBA"), grid_rgba).convert("RGB")

            canvas.paste(head_img, (0, 40))
            head_proj = self._project_points_in_cam(
                "cam_head",
                np.array([l_tcp, l_plumb, l_ja, l_jb, r_tcp, r_plumb, r_ja, r_jb], dtype=np.float64),
            )

            def _draw_cockpit_sight(
                uv_tcp: tuple[float, float, bool],
                uv_plb: tuple[float, float, bool],
                uv_ja: tuple[float, float, bool],
                uv_jb: tuple[float, float, bool],
                ox: float,
                oy: float,
                sx: float,
                sy: float,
                xmin: int,
                ymin: int,
                xmax: int,
                ymax: int,
                ring_color: tuple[int, int, int],
                grip_state: str,
                is_head_cam: bool = False,
                z_cm: float = 0.0,
            ) -> None:
                tx, ty, tok = uv_tcp
                px, py, pok = uv_plb
                tx_s = int(round(ox + tx * sx))
                ty_s = int(round(oy + ty * sy))
                px_s = int(round(ox + px * sx))
                py_s = int(round(oy + py * sy))
                t_in = tok and (xmin <= tx_s <= xmax) and (ymin <= ty_s <= ymax)
                p_in = pok and (xmin <= px_s <= xmax) and (ymin <= py_s <= ymax)
                hover_head = is_head_cam and (z_cm > 3.5)
                eff_ring_col = (70, 255, 115) if (is_head_cam and grip_state == "HELD") else ring_color
                if is_head_cam and hover_head and t_in and p_in and abs(ty_s - py_s) > 16:
                    end_y = py_s - 13 if ty_s < py_s else py_s + 13
                    draw.line([(tx_s, ty_s), (px_s, end_y)], fill=eff_ring_col, width=1)
                if is_head_cam and p_in:
                    r_in, r_out = 13, 16
                    for sx_c in (-1, 1):
                        for sy_c in (-1, 1):
                            x_a, y_a = px_s + sx_c * r_in, py_s + sy_c * r_in
                            x_b, y_b = px_s + sx_c * r_out, py_s + sy_c * r_in
                            x_c, y_c = px_s + sx_c * r_in, py_s + sy_c * r_out
                            draw.line([(x_a, y_a), (x_b, y_b)], fill=eff_ring_col, width=2)
                            draw.line([(x_a, y_a), (x_c, y_c)], fill=eff_ring_col, width=2)
                    if grip_state == "HELD":
                        bx_h = min(xmax - 30, px_s + 16)
                        by_h = max(ymin + 2, py_s - 6)
                        draw.rectangle([bx_h - 1, by_h - 1, bx_h + 29, by_h + 10], fill=(12, 24, 16))
                        draw.text((bx_h, by_h), "HELD", fill=(70, 255, 115), font=font)
                if grip_state == "HELD":
                    c_cross = (70, 255, 115)
                elif grip_state == "EMPTY":
                    c_cross = (255, 85, 85)
                elif hover_head:
                    c_cross = (195, 195, 125)
                else:
                    c_cross = (255, 255, 95)
                if (not is_head_cam) or (grip_state == "OPEN"):
                    r_j = 1 if is_head_cam else 4
                    j_pts_ok: list[tuple[int, int]] = []
                    for uv_j in (uv_ja, uv_jb):
                        jx, jy, jok = uv_j
                        jx_s = int(round(ox + jx * sx))
                        jy_s = int(round(oy + jy * sy))
                        if jok and (xmin <= jx_s <= xmax) and (ymin <= jy_s <= ymax):
                            j_pts_ok.append((jx_s, jy_s))
                            if (not is_head_cam) or (abs(jx_s - px_s) >= 12 or abs(jy_s - py_s) >= 12):
                                draw.rectangle([jx_s - r_j, jy_s - r_j, jx_s + r_j, jy_s + r_j], fill=c_cross, outline=(15, 18, 24))
                    if (not is_head_cam) and len(j_pts_ok) == 2 and t_in:
                        draw.line([j_pts_ok[0], (tx_s, ty_s)], fill=c_cross, width=1)
                        draw.line([(tx_s, ty_s), j_pts_ok[1]], fill=c_cross, width=1)
                if t_in and (not is_head_cam or hover_head):
                    arm = 3 if is_head_cam else 9
                    draw.line([(tx_s - arm, ty_s), (tx_s + arm, ty_s)], fill=(15, 18, 24), width=2 if is_head_cam else 4)
                    draw.line([(tx_s, ty_s - arm), (tx_s, ty_s + arm)], fill=(15, 18, 24), width=2 if is_head_cam else 4)
                    draw.line([(tx_s - arm, ty_s), (tx_s + arm, ty_s)], fill=c_cross, width=1 if is_head_cam else 2)
                    draw.line([(tx_s, ty_s - arm), (tx_s, ty_s + arm)], fill=c_cross, width=1 if is_head_cam else 2)
                    if not is_head_cam:
                        lx_p = min(xmax - 34, max(xmin + 2, tx_s + 10))
                        ly_p = min(ymax - 12, max(ymin + 2, ty_s - 6))
                        draw.rectangle([lx_p - 1, ly_p - 1, lx_p + 33, ly_p + 9], fill=(12, 16, 22))
                        draw.text((lx_p, ly_p), "PINCH", fill=c_cross, font=font)

            _draw_cockpit_sight(
                head_proj[0], head_proj[1], head_proj[2], head_proj[3],
                0.0, 40.0, 640.0 / hw0, 420.0 / hh0,
                2, 62, 637, 458,
                (0, 225, 255), ("HELD" if l_is_held else ("EMPTY" if l_is_empty_clsd else "OPEN")),
                is_head_cam=True, z_cm=l_z_cm,
            )
            _draw_cockpit_sight(
                head_proj[4], head_proj[5], head_proj[6], head_proj[7],
                0.0, 40.0, 640.0 / hw0, 420.0 / hh0,
                2, 62, 637, 458,
                (255, 165, 40), ("HELD" if r_is_held else ("EMPTY" if r_is_empty_clsd else "OPEN")),
                is_head_cam=True, z_cm=r_z_cm,
            )

            draw.rectangle([4, 44, 575, 60], fill=(16, 20, 28))
            draw.text(
                (8, 46),
                "HEAD CAM (Border Ruler cm: X0=Center | Aim Ring 'o' on Table, '+'=3D Tip)",
                fill=(180, 230, 255),
                font=font,
            )

            draw.rectangle([0, 460, 319, 691], outline=(0, 200, 235), width=2)
            canvas.paste(lw_img, (1, 464))
            lw_proj = self._project_points_in_cam(
                "cam_left_wrist", np.array([l_tcp, l_plumb, l_ja, l_jb], dtype=np.float64)
            )
            _draw_cockpit_sight(
                lw_proj[0], lw_proj[1], lw_proj[2], lw_proj[3],
                1.0, 464.0, 318.0 / lw0, 226.0 / lh0,
                3, 482, 316, 688,
                (0, 230, 255), ("HELD" if l_is_held else ("EMPTY" if l_is_empty_clsd else "OPEN")),
            )
            if self.left_grip >= 0.5:
                lg_txt, lg_col = "OPEN", (165, 235, 255)
            elif l_is_held:
                lg_txt, lg_col = f"HELD({int(round(l_jaw_mm))}mm)", (70, 255, 115)
            else:
                lg_txt, lg_col = "EMPTY!(0mm)", (255, 95, 95)
            draw.rectangle([2, 464, 318, 480], fill=(12, 32, 42))
            draw.text(
                (5, 466),
                f"L:{lg_txt} X:{l_x_cm:+.0f} Y:{l_y_cm:+.0f} Z:{l_z_cm:+.0f} Yaw:{int(round(np.rad2deg(self.user_yaw_l))):+d} px:({int(round(head_proj[1][0]*640.0/hw0))},{int(round(40.0+head_proj[1][1]*420.0/hh0))})",
                fill=lg_col,
                font=font,
            )

            draw.rectangle([320, 460, 639, 691], outline=(245, 155, 50), width=2)
            canvas.paste(rw_img, (321, 464))
            rw_proj = self._project_points_in_cam(
                "cam_right_wrist", np.array([r_tcp, r_plumb, r_ja, r_jb], dtype=np.float64)
            )
            _draw_cockpit_sight(
                rw_proj[0], rw_proj[1], rw_proj[2], rw_proj[3],
                321.0, 464.0, 318.0 / rw0, 226.0 / rh0,
                323, 482, 636, 688,
                (255, 170, 45), ("HELD" if r_is_held else ("EMPTY" if r_is_empty_clsd else "OPEN")),
            )
            if self.right_grip >= 0.5:
                rg_txt, rg_col = "OPEN", (255, 225, 150)
            elif r_is_held:
                rg_txt, rg_col = f"HELD({int(round(r_jaw_mm))}mm)", (70, 255, 115)
            else:
                rg_txt, rg_col = "EMPTY!(0mm)", (255, 95, 95)
            draw.rectangle([322, 464, 638, 480], fill=(42, 26, 12))
            draw.text(
                (325, 466),
                f"R:{rg_txt} X:{r_x_cm:+.0f} Y:{r_y_cm:+.0f} Z:{r_z_cm:+.0f} Yaw:{int(round(np.rad2deg(self.user_yaw_r))):+d} px:({int(round(head_proj[5][0]*640.0/hw0))},{int(round(40.0+head_proj[5][1]*420.0/hh0))})",
                fill=rg_col,
                font=font,
            )
            def _wrist_edge_tags(cam_name: str, is_right: bool) -> tuple[str, str, str, str]:
                dirs = (
                    (("Up(+Y)", (0.0, 1.0)), ("Dn(-Y)", (0.0, -1.0)), ("Lt(-X)", (-1.0, 0.0)), ("Rt(+X)", (1.0, 0.0)))
                    if is_right else
                    (("W(+Y)", (0.0, 1.0)), ("S(-Y)", (0.0, -1.0)), ("A(-X)", (-1.0, 0.0)), ("D(+X)", (1.0, 0.0)))
                )
                try:
                    names = self.env.camera_manager.camera_names[0]
                    cam_obj = self.env.camera_manager.cameras[0][names.index(cam_name)]
                    R_cw = cam_obj.get_view_matrix_ros()[:2, :2].detach().cpu().numpy().astype(np.float64)
                    # Image screen vectors in ROS optical frame (+u right, +v down): top=(0,-1), bot=(0,+1), left=(-1,0), right=(+1,0)
                    out_tags = []
                    for pfx, sv in (("^", np.array([0.0, -1.0])), ("v", np.array([0.0, 1.0])), ("<", np.array([-1.0, 0.0])), (">", np.array([1.0, 0.0]))):
                        best_lbl = max(dirs, key=lambda item: float(np.dot(R_cw @ np.array(item[1]), sv)))[0]
                        out_tags.append(f"{best_lbl}>" if pfx == ">" else f"{pfx}{best_lbl}")
                    return (out_tags[0], out_tags[1], out_tags[2], out_tags[3])
                except Exception:
                    return ("^Up(+Y)", "vDn(-Y)", "<Lt(-X)", "Rt(+X)>") if is_right else ("^W(+Y)", "vS(-Y)", "<A(-X)", "D(+X)>")

            lu_t, ld_t, ll_t, lr_t = _wrist_edge_tags("cam_left_wrist", False)
            ru_t, rd_t, rl_t, rr_t = _wrist_edge_tags("cam_right_wrist", True)
            for (cx_m, xmin_p, xmax_p, up_t, dn_t, lt_t, rt_t, tcol) in (
                (160, 4, 315, lu_t, ld_t, ll_t, lr_t, (140, 235, 255)),
                (480, 324, 635, ru_t, rd_t, rl_t, rr_t, (255, 205, 115)),
            ):
                uw = max(24, len(up_t) * 3 + 4)
                dw = max(24, len(dn_t) * 3 + 4)
                lw = max(22, len(lt_t) * 6 + 4)
                rw = max(22, len(rt_t) * 6 + 4)
                draw.rectangle([cx_m - uw, 481, cx_m + uw, 493], fill=(14, 16, 22))
                draw.text((cx_m - uw + 3, 481), up_t, fill=tcol, font=font)
                draw.rectangle([cx_m - dw, 677, cx_m + dw, 689], fill=(14, 16, 22))
                draw.text((cx_m - dw + 3, 677), dn_t, fill=tcol, font=font)
                draw.rectangle([xmin_p, 578, xmin_p + lw, 590], fill=(14, 16, 22))
                draw.text((xmin_p + 2, 578), lt_t, fill=tcol, font=font)
                draw.rectangle([xmax_p - rw, 578, xmax_p, 590], fill=(14, 16, 22))
                draw.text((xmax_p - rw + 2, 578), rt_t, fill=tcol, font=font)

            st_dict = self._build_state_unlocked()
            gs = st_dict["game_state"]
            draw.rectangle([0, 692, 639, 719], fill=(24, 26, 34))
            bar_txt = (
                f"PAD:[{self.last_cmd_label}] | Stance:{self.stance_mode} [V=TopDown H=Origin] | "
                f"Step={gs['gamepad_steps']}"
            )
            draw.text((6, 700), bar_txt[:106], fill=(225, 230, 240), font=font)

            buf = io.BytesIO()
            canvas.save(buf, format="PNG")
            self._cached_png = buf.getvalue()
            return self._cached_png


    def _latch_hold_joints_unlocked(self) -> None:
        st = self.obs.get("state", {})
        lj = st.get("left_arm_joint_state")
        rj = st.get("right_arm_joint_state")
        self._hold_left_joints = np.asarray(lj, dtype=np.float64).reshape(-1).tolist() if lj is not None else None
        self._hold_right_joints = np.asarray(rj, dtype=np.float64).reshape(-1).tolist() if rj is not None else None

    def _render_png_unlocked(self) -> bytes:
        # Must be called with self.lock held on the main thread
        if self._cached_png is not None:
            return self._cached_png
        self.lock.release()
        try:
            return self.render_png()
        finally:
            self.lock.acquire()

    def _refresh_snapshots_unlocked(self) -> dict[str, Any]:
        png = self._render_png_unlocked()
        st = self._build_state_unlocked()
        self.latest_png = png
        self.latest_state = st
        return st

    def set_paused(self, paused: bool) -> dict[str, Any]:
        with self.lock:
            self.is_paused = bool(paused)
            self.latest_state = self._build_state_unlocked()
            return self.latest_state

    def step_realtime_tick(self) -> None:
        with self.lock:
            if self.is_paused or bool(self.env.end_flag[0]):
                return
            if self._hold_left_joints is not None and self._hold_right_joints is not None:
                action = {
                    "left_arm_joint_state": list(self._hold_left_joints),
                    "left_ee_joint_state": [float(self.left_grip)],
                    "right_arm_joint_state": list(self._hold_right_joints),
                    "right_ee_joint_state": [float(self.right_grip)],
                }
            else:
                st = self.obs.get("state", {})
                cur_l = np.asarray(st.get("left_ee_pose", self.home_left_pose), dtype=np.float64)
                cur_r = np.asarray(st.get("right_ee_pose", self.home_right_pose), dtype=np.float64)
                action = {
                    "left_ee_pose": cur_l.tolist(),
                    "left_ee_joint_state": [float(self.left_grip)],
                    "right_ee_pose": cur_r.tolist(),
                    "right_ee_joint_state": [float(self.right_grip)],
                }
            n_rt = max(1, int(os.environ.get("ROBODOJO_REALTIME_STEPS_PER_TICK", "2")))
            for _ in range(n_rt):
                if bool(self.env.end_flag[0]):
                    break
                self.env.take_action(action)
                self.sim_steps += 1
            self.obs = self.env.get_obs()
            self._cached_png = None
            self._refresh_snapshots_unlocked()

    def submit_main_cmd(self, fn_name: str, kwargs: dict[str, Any], timeout: float = 120.0) -> Any:
        done = threading.Event()
        box: dict[str, Any] = {}
        self.cmd_queue.put((fn_name, kwargs, box, done))
        if not done.wait(timeout=timeout):
            raise TimeoutError(f"Timed out waiting for main-thread {fn_name}")
        if "err" in box:
            raise box["err"]
        return box.get("res")


SESSION: RoboDojoGamepadSession | None = None


class SimRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        global SESSION
        assert SESSION is not None
        if self.path.startswith("/health"):
            self._send_json(200, {
                "ok": True,
                "task_name": SESSION.task_name,
                "paused": SESSION.is_paused,
                "sim_steps": SESSION.sim_steps,
            })
            return
        if self.path.startswith("/state"):
            self._send_json(200, SESSION.latest_state or SESSION.get_state())
            return
        if self.path.startswith("/screenshot"):
            SESSION._clock_started = True
            png = SESSION.latest_png
            if not png:
                png = SESSION.submit_main_cmd("render_png", {}, timeout=30.0)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        global SESSION
        assert SESSION is not None
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(body.decode("utf-8") or "{}")
        except Exception:
            data = {}

        try:
            if self.path.startswith("/pause"):
                st = SESSION.submit_main_cmd("set_paused", {"paused": True}, timeout=15.0)
                self._send_json(200, {"ok": True, "paused": True, "state": st})
                return
            if self.path.startswith("/resume"):
                st = SESSION.submit_main_cmd("set_paused", {"paused": False}, timeout=15.0)
                self._send_json(200, {"ok": True, "paused": False, "state": st})
                return
            if self.path.startswith("/reset"):
                st = SESSION.submit_main_cmd(
                    "reset_episode",
                    {
                        "advance_seed": bool(data.get("advance_seed", True)),
                        "explicit_seed": data.get("seed"),
                        "paused": data.get("paused"),
                    },
                    timeout=120.0,
                )
                self._send_json(200, {"ok": True, "state": st})
                return
            if self.path.startswith("/gamepad"):
                SESSION._clock_started = True
                keys = data.get("keys") or []
                dur = float(data.get("duration_s", 0.25))
                hold_seq = int(data.get("hold_seq", 0) or 0)
                st = SESSION.submit_main_cmd(
                    "step_gamepad",
                    {"keys": keys, "duration_s": dur, "hold_seq": hold_seq},
                    timeout=90.0,
                )
                self._send_json(200, {"ok": True, "state": st})
                return
            if self.path.startswith("/shutdown"):
                self._send_json(200, {"ok": True})
                threading.Thread(target=lambda: (time.sleep(0.1), os._exit(0)), daemon=True).start()
                return
            self._send_json(404, {"error": "not_found"})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    global SESSION
    realtime_tick_s = float(os.environ.get("ROBODOJO_REALTIME_TICK_S", "0.22"))
    realtime_enabled = os.environ.get("ROBODOJO_REALTIME_CLOCK", "1").strip().lower() not in ("0", "false", "no")
    print(
        f"[RoboDojoSimServer] Initializing 6-DoF RealTime task={args_cli.task_name} "
        f"on GPU {args_cli.device_id} (realtime={realtime_enabled}, tick={realtime_tick_s:.2f}s)...",
        flush=True,
    )
    t0 = time.time()
    SESSION = RoboDojoGamepadSession(
        task_name=args_cli.task_name,
        env_cfg_type=args_cli.env_cfg_type,
        device_id=args_cli.device_id,
        seed=args_cli.seed,
    )
    print(
        f"[RoboDojoSimServer] Ready in {time.time() - t0:.1f}s at "
        f"http://{args_cli.http_host}:{args_cli.http_port} (instruction={SESSION.obs.get('instruction')!r})",
        flush=True,
    )
    httpd = ThreadingHTTPServer((args_cli.http_host, args_cli.http_port), SimRequestHandler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    next_tick_t = time.monotonic() + realtime_tick_s
    try:
        while True:
            try:
                fn_name, kwargs, box, done = SESSION.cmd_queue.get(timeout=0.012)
                try:
                    box["res"] = getattr(SESSION, fn_name)(**kwargs)
                except Exception as exc:
                    traceback.print_exc()
                    box["err"] = exc
                finally:
                    done.set()
                continue
            except queue.Empty:
                pass

            if realtime_enabled and SESSION._clock_started and not SESSION.is_paused and not bool(SESSION.env.end_flag[0]):
                now = time.monotonic()
                if now >= next_tick_t:
                    try:
                        SESSION.step_realtime_tick()
                    except Exception:
                        traceback.print_exc()
                    next_tick_t = max(time.monotonic() + 0.04, next_tick_t + realtime_tick_s)
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
