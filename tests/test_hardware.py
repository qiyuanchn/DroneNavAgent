"""Hardware profile and fit-report contract (``vlm_nav/hardware.py``).

Everything here runs on a CPU-only host: the live-CUDA path is exercised with a
fake torch module, and the CLI's fit/no-fit exits use an in-process patched
snapshot. The unavailable path is tested against real ``snapshot("cpu")`` and
against an import failure, so a missing GPU can never silently become zeros.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_nav import hardware  # noqa: E402

MIB = 1024.0 * 1024.0
SCRIPT_PATH = ROOT / "scripts" / "check_hardware_profile.py"


def _load_script():
    name = "diffaero_script_check_hardware_profile"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        total_mib: float = 24576.0,
        free_mib: float = 24000.0,
        allocated_mib: float = 0.0,
        reserved_mib: float = 0.0,
        peak_allocated_mib: float = 0.0,
        peak_reserved_mib: float = 0.0,
        gpu_name: str = "Fake RTX 4090",
    ) -> None:
        self.available = available
        self.total_mib = total_mib
        self.free_mib = free_mib
        self.allocated_mib = allocated_mib
        self.reserved_mib = reserved_mib
        self.peak_allocated_mib = peak_allocated_mib
        self.peak_reserved_mib = peak_reserved_mib
        self.gpu_name = gpu_name
        self.reset_calls: list[str] = []
        self.sync_calls: list[str] = []
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise RuntimeError("CUDA driver error")

    def is_available(self) -> bool:
        return self.available

    def synchronize(self, device) -> None:
        self.sync_calls.append(str(device))

    def mem_get_info(self, device):
        self._check()
        return self.free_mib * MIB, self.total_mib * MIB

    def memory_allocated(self, device):
        self._check()
        return self.allocated_mib * MIB

    def memory_reserved(self, device):
        self._check()
        return self.reserved_mib * MIB

    def max_memory_allocated(self, device):
        self._check()
        return self.peak_allocated_mib * MIB

    def max_memory_reserved(self, device):
        self._check()
        return self.peak_reserved_mib * MIB

    def get_device_name(self, device) -> str:
        self._check()
        return self.gpu_name

    def reset_peak_memory_stats(self, device) -> None:
        self.reset_calls.append(str(device))


class FakeTorch:
    __version__ = "2.7.1+fake"

    class version:  # noqa: N801 - mirrors torch.version
        cuda = "12.8"

    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda


def fake_snapshot(
    *,
    total_mib: float = 24576.0,
    free_mib: float = 24000.0,
    allocated_mib: float = 0.0,
    reserved_mib: float = 0.0,
    peak_allocated_mib: float = 0.0,
    peak_reserved_mib: float = 0.0,
    gpu_name: str = "Fake RTX 4090",
    available: bool = True,
) -> dict:
    return {
        "available": available,
        "gpu_name": gpu_name if available else "",
        "device": "cuda",
        "total_mib": total_mib if available else None,
        "free_mib": free_mib if available else None,
        "allocated_mib": allocated_mib if available else None,
        "reserved_mib": reserved_mib if available else None,
        "peak_allocated_mib": peak_allocated_mib if available else None,
        "peak_reserved_mib": peak_reserved_mib if available else None,
        "driver_used_mib": (
            round(total_mib - free_mib, 1) if available else None
        ),
        "reason": "" if available else "CUDA is not available to torch",
    }


class ProfileTest(unittest.TestCase):
    def test_declared_totals(self):
        self.assertEqual(hardware.PROFILES["a100-40gb"].total_mib, 40960)
        self.assertEqual(hardware.PROFILES["4090-24gb"].total_mib, 24576)
        self.assertEqual(hardware.PROFILES["4090-local"].total_mib, 24576)

    def test_local_4090_alias_reserve_and_honest_peak(self):
        local = hardware.PROFILES["4090-local"]
        self.assertEqual(local.reserve_mib, 2048)
        self.assertEqual(
            local.expected_peak_mib,
            hardware.PROFILES["4090-24gb"].expected_peak_mib,
        )
        # ~21 GiB estimate, below the sum of isolated peaks (17.3+4.2+1.5 GiB)
        self.assertEqual(local.expected_peak_mib, 21504)
        self.assertLess(local.expected_peak_mib, 17300 + 4200 + 1500)
        self.assertLessEqual(local.expected_peak_mib + local.reserve_mib, local.total_mib)
        self.assertIn("alias", local.description.lower())

    def test_profile_validation(self):
        with self.assertRaises(ValueError):
            hardware.HardwareProfile(id="", total_mib=1, reserve_mib=0, expected_peak_mib=1)
        with self.assertRaises(ValueError):
            hardware.HardwareProfile(id="x", total_mib=0, reserve_mib=0, expected_peak_mib=1)
        with self.assertRaises(ValueError):
            hardware.HardwareProfile(id="x", total_mib=10, reserve_mib=-1, expected_peak_mib=1)

    def test_to_dict_is_json_serializable(self):
        data = hardware.PROFILES["4090-local"].to_dict()
        self.assertEqual(data["id"], "4090-local")
        self.assertEqual(data["reserve_mib"], 2048)
        json.dumps(data)


class SnapshotTest(unittest.TestCase):
    def assert_unavailable(self, snap: dict):
        for key in hardware.SNAPSHOT_KEYS:
            self.assertIn(key, snap)
        self.assertFalse(snap["available"])
        for key in (
            "total_mib", "free_mib", "allocated_mib", "reserved_mib",
            "peak_allocated_mib", "peak_reserved_mib", "driver_used_mib",
        ):
            self.assertIsNone(snap[key], key)
        self.assertTrue(snap["reason"])

    def test_cpu_device_is_explicitly_unavailable(self):
        snap = hardware.snapshot("cpu")
        self.assert_unavailable(snap)
        self.assertIn("CPU", snap["reason"])

    def test_missing_torch_is_explicitly_unavailable(self):
        with mock.patch.object(hardware, "_load_torch", side_effect=ImportError("no torch")):
            snap = hardware.snapshot("cuda")
        self.assert_unavailable(snap)
        self.assertIn("torch", snap["reason"].lower())

    def test_no_cuda_is_explicitly_unavailable(self):
        torch = FakeTorch(FakeCuda(available=False))
        with mock.patch.object(hardware, "_load_torch", return_value=torch):
            snap = hardware.snapshot("cuda")
        self.assert_unavailable(snap)
        self.assertIn("CUDA", snap["reason"])

    def test_cuda_read_failure_does_not_crash(self):
        cuda = FakeCuda()
        cuda.fail = True
        with mock.patch.object(hardware, "_load_torch", return_value=FakeTorch(cuda)):
            snap = hardware.snapshot("cuda")
        self.assert_unavailable(snap)
        self.assertIn("CUDA memory query failed", snap["reason"])

    def test_metrics_are_real_numbers_and_driver_used_is_derived(self):
        cuda = FakeCuda(
            total_mib=24576.0, free_mib=20000.0, allocated_mib=1000.0,
            reserved_mib=1500.0, peak_allocated_mib=3000.0,
            peak_reserved_mib=3500.0,
        )
        with mock.patch.object(hardware, "_load_torch", return_value=FakeTorch(cuda)):
            snap = hardware.snapshot("cuda:0")
        self.assertTrue(snap["available"])
        self.assertEqual(snap["device"], "cuda:0")
        self.assertEqual(snap["gpu_name"], "Fake RTX 4090")
        self.assertEqual(snap["total_mib"], 24576.0)
        self.assertEqual(snap["free_mib"], 20000.0)
        self.assertEqual(snap["allocated_mib"], 1000.0)
        self.assertEqual(snap["reserved_mib"], 1500.0)
        self.assertEqual(snap["peak_allocated_mib"], 3000.0)
        self.assertEqual(snap["peak_reserved_mib"], 3500.0)
        self.assertEqual(snap["driver_used_mib"], 4576.0)
        self.assertEqual(snap["reason"], "")
        self.assertEqual(cuda.sync_calls, ["cuda:0"])


class ResetPeakTest(unittest.TestCase):
    def test_cpu_is_a_noop_without_importing_torch(self):
        with mock.patch.object(hardware, "_load_torch", side_effect=AssertionError):
            self.assertIsNone(hardware.reset_peak("cpu"))

    def test_missing_torch_is_a_noop(self):
        with mock.patch.object(hardware, "_load_torch", side_effect=ImportError("no torch")):
            self.assertIsNone(hardware.reset_peak("cuda"))

    def test_reset_calls_torch_on_an_available_device(self):
        cuda = FakeCuda()
        with mock.patch.object(hardware, "_load_torch", return_value=FakeTorch(cuda)):
            hardware.reset_peak("cuda:1")
        self.assertEqual(cuda.reset_calls, ["cuda:1"])
        self.assertEqual(cuda.sync_calls, ["cuda:1"])

    def test_reset_is_a_noop_without_cuda(self):
        cuda = FakeCuda(available=False)
        with mock.patch.object(hardware, "_load_torch", return_value=FakeTorch(cuda)):
            hardware.reset_peak("cuda")
        self.assertEqual(cuda.reset_calls, [])


class FitReportTest(unittest.TestCase):
    def test_component_sum_and_reserve(self):
        snap = fake_snapshot()
        over = hardware.fit_report(
            hardware.PROFILES["4090-local"],
            snap,
            {"vlm": 17300, "detector": 4200, "renderer": 1500},
        )
        self.assertEqual(over["required_mib"], 23000.0)
        self.assertEqual(over["headroom_mib"], -472.0)
        self.assertFalse(over["fits"])
        self.assertTrue(any("exceeds" in reason for reason in over["reasons"]))

        under = hardware.fit_report(
            hardware.PROFILES["4090-local"],
            snap,
            {"vlm": 17000, "detector": 3000, "renderer": 1000},
        )
        self.assertEqual(under["required_mib"], 21000.0)
        self.assertEqual(under["headroom_mib"], 1528.0)
        self.assertTrue(under["fits"])
        self.assertEqual(under["reasons"], [])

    def test_reserve_boundary_is_inclusive(self):
        snap = fake_snapshot()
        exact = hardware.fit_report(
            "4090-local", snap, {"stack": 24576 - 2048}
        )
        self.assertEqual(exact["headroom_mib"], 0.0)
        self.assertTrue(exact["fits"])

        one_over = hardware.fit_report(
            "4090-local", snap, {"stack": 24576 - 2048 + 1}
        )
        self.assertEqual(one_over["headroom_mib"], -1.0)
        self.assertFalse(one_over["fits"])

    def test_expected_peak_is_used_without_components(self):
        report = hardware.fit_report("4090-local", fake_snapshot())
        self.assertEqual(report["required_mib"], 21504.0)
        self.assertEqual(report["headroom_mib"], 1024.0)
        self.assertEqual(report["source"], "expected_peak")
        self.assertEqual(report["components"], {})
        self.assertTrue(report["fits"])
        self.assertEqual(report["reasons"], [])

    def test_low_free_memory_blocks_an_otherwise_fitting_budget(self):
        snap = fake_snapshot(free_mib=1000.0)
        report = hardware.fit_report(
            "4090-local", snap, {"vlm": 17000, "detector": 3000}
        )
        self.assertGreater(report["headroom_mib"], 0)
        self.assertFalse(report["fits"])
        self.assertTrue(any("free/reusable" in reason for reason in report["reasons"]))

    def test_reserved_but_unallocated_memory_is_reusable(self):
        snap = fake_snapshot(
            free_mib=1000.0, allocated_mib=500.0, reserved_mib=20000.0
        )
        report = hardware.fit_report(
            "4090-local", snap, {"vlm": 17000, "detector": 3000}
        )
        # free 1000 + cached pool (20000 - 500) is enough for the 20000 MiB peak
        self.assertEqual(report["usable_free_mib"], 20500.0)
        self.assertTrue(report["fits"])

    def test_smaller_measured_total_caps_the_profile_budget(self):
        snap = fake_snapshot(total_mib=16384.0, free_mib=15000.0)
        report = hardware.fit_report("4090-local", snap)
        self.assertEqual(report["total_mib"], 16384.0)
        self.assertEqual(report["total_source"], "measured")
        self.assertFalse(report["fits"])
        self.assertTrue(any("below the 4090-local profile" in r for r in report["reasons"]))

    def test_larger_measured_total_does_not_inflate_the_profile(self):
        snap = fake_snapshot(total_mib=40960.0, free_mib=40000.0)
        report = hardware.fit_report("4090-local", snap)
        self.assertEqual(report["total_mib"], 24576.0)
        self.assertEqual(report["total_source"], "profile")
        self.assertTrue(report["fits"])

    def test_unavailable_snapshot_is_never_a_pretend_pass(self):
        snap = hardware.snapshot("cpu")
        report = hardware.fit_report("4090-local", snap, {"vlm": 1000})
        self.assertFalse(report["available"])
        self.assertTrue(report["fits"])  # static budget only
        self.assertTrue(any("unavailable" in reason for reason in report["reasons"]))
        self.assertTrue(any("static budget" in reason for reason in report["reasons"]))

    def test_no_snapshot_is_a_static_budget_check(self):
        report = hardware.fit_report(hardware.PROFILES["a100-40gb"], None)
        self.assertFalse(report["available"])
        self.assertEqual(report["total_mib"], 40960.0)
        self.assertTrue(report["fits"])
        self.assertTrue(any("no GPU snapshot" in reason for reason in report["reasons"]))

    def test_component_mapping_values_are_accepted(self):
        report = hardware.fit_report(
            "4090-local",
            fake_snapshot(),
            {"vlm": {"peak_allocated_mib": 17300}, "detector": {"mib": 4200}},
        )
        self.assertEqual(report["components"], {"vlm": 17300.0, "detector": 4200.0})
        self.assertEqual(report["required_mib"], 21500.0)

    def test_component_records_are_accepted(self):
        report = hardware.fit_report(
            "4090-local",
            fake_snapshot(),
            [
                {"name": "vlm", "peak_allocated_mib": 17000},
                ("detector", 3000),
            ],
        )
        self.assertEqual(report["components"], {"vlm": 17000.0, "detector": 3000.0})
        self.assertEqual(report["required_mib"], 20000.0)

    def test_invalid_components_and_profiles(self):
        with self.assertRaises(ValueError):
            hardware.fit_report("4090-local", fake_snapshot(), {"vlm": -1})
        with self.assertRaises(ValueError):
            hardware.fit_report("4090-local", fake_snapshot(), {"vlm": "lots"})
        with self.assertRaises(TypeError):
            hardware.fit_report("4090-local", fake_snapshot(), "vlm=1")
        with self.assertRaises(ValueError):
            hardware.fit_report("no-such-profile", fake_snapshot())
        with self.assertRaises(TypeError):
            hardware.fit_report("4090-local", "not-a-snapshot")

    def test_mapping_profile_is_accepted(self):
        report = hardware.fit_report(
            {
                "id": "tiny",
                "total_mib": 8192,
                "reserve_mib": 512,
                "expected_peak_mib": 8000,
            },
            fake_snapshot(total_mib=8192.0, free_mib=8000.0),
        )
        self.assertEqual(report["profile"], "tiny")
        self.assertEqual(report["headroom_mib"], -320.0)
        self.assertFalse(report["fits"])


class MarkdownTableTest(unittest.TestCase):
    def test_fit_table_contains_the_decision_and_components(self):
        report = hardware.fit_report(
            "4090-local",
            fake_snapshot(),
            {"vlm": 17000, "detector": 3000},
        )
        table = hardware.markdown_table(report)
        self.assertIn("| metric | value |", table)
        self.assertIn("| fits | yes |", table)
        self.assertIn("| headroom_mib | 2528.0 |", table)
        self.assertIn("| component:vlm | 17000.0 |", table)
        self.assertIn("4090-local", table)

    def test_no_fit_table_and_reasons(self):
        report = hardware.fit_report(
            "4090-local", fake_snapshot(), {"vlm": 24000}
        )
        table = hardware.markdown_table(report)
        self.assertIn("| fits | no |", table)
        self.assertIn("exceeds", table)

    def test_pipes_are_escaped(self):
        report = hardware.fit_report(
            "4090-local", fake_snapshot(), {"vlm|x": 1000}
        )
        table = hardware.markdown_table(report)
        self.assertIn("vlm\\|x", table)

    def test_non_mapping_is_rejected(self):
        with self.assertRaises(TypeError):
            hardware.markdown_table(["not", "a", "report"])


class CliTest(unittest.TestCase):
    def test_parser_switches(self):
        args = SCRIPT.build_parser().parse_args(
            ["--profile", "4090-local", "--device", "cpu",
             "--component", "vlm=17300", "--out", "x"]
        )
        self.assertEqual(args.profile, "4090-local")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.component, [("vlm", 17300.0)])

    def test_cpu_device_exits_2_and_writes_an_honest_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            completed = subprocess.run(
                [
                    sys.executable, str(SCRIPT_PATH),
                    "--profile", "4090-local",
                    "--device", "cpu",
                    "--out", str(out),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
            markdown = (out / "report.md").read_text(encoding="utf-8")
            stdout = json.loads(completed.stdout)
        self.assertEqual(payload["schema"], SCRIPT.SCHEMA)
        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(payload["exit_code"], 2)
        self.assertFalse(payload["available"])
        self.assertFalse(payload["snapshot"]["available"])
        self.assertIsNone(payload["snapshot"]["total_mib"])
        self.assertIsNone(payload["snapshot"]["peak_allocated_mib"])
        self.assertIn("unavailable", markdown.lower())
        self.assertIn("GPU available: false", markdown)
        self.assertEqual(stdout["exit_code"], 2)

    def test_fit_exits_0_in_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with mock.patch.object(
                SCRIPT, "snapshot", return_value=fake_snapshot()
            ):
                code = SCRIPT.main([
                    "--profile", "4090-local", "--device", "cuda",
                    "--out", str(out),
                ])
            payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
            markdown_exists = Path(payload["written"]["markdown"]).is_file()
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "fit")
        self.assertTrue(payload["available"])
        self.assertTrue(payload["fits"])
        self.assertEqual(payload["report"]["headroom_mib"], 1024.0)
        self.assertTrue(payload["report"]["fits"])
        self.assertTrue(markdown_exists)

    def test_cli_delegates_to_the_public_hardware_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with mock.patch.object(
                hardware, "snapshot", return_value=fake_snapshot()
            ):
                code = SCRIPT.main([
                    "--profile", "4090-local", "--device", "cuda",
                    "--out", str(out),
                ])
        self.assertEqual(code, 0)

    def test_no_fit_exits_1_in_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with mock.patch.object(
                SCRIPT, "snapshot", return_value=fake_snapshot()
            ):
                code = SCRIPT.main([
                    "--profile", "4090-local", "--device", "cuda",
                    "--component", "vlm=23000",
                    "--out", str(out),
                ])
            payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "no_fit")
        self.assertFalse(payload["report"]["fits"])
        self.assertEqual(payload["report"]["required_mib"], 23000.0)
        self.assertEqual(payload["report"]["headroom_mib"], -472.0)


if __name__ == "__main__":
    unittest.main()
