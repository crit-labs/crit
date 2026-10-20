"""Publication failures use synthetic artifacts, including abrupt coordinator exit."""
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from test_run_records import ROOT, RunFixture
from crit.run import store as run_store
from crit.run.records import digest, inventory, read_json


RECIPE = {"build": {"toolchain": {"identity": "host fixture"}, "dependencies": [], "command": ["fixture"]},
          "launch": ["engine.fixture"], "capabilities": ["prefill", "decode"]}


class PackagePublicationTests(RunFixture, unittest.TestCase):
    def prepare(self):
        candidate = self.candidate()
        source = self.root / "package-input"
        source.mkdir()
        (source / "engine.fixture").write_text("runnable fixture bytes", encoding="utf-8")
        return candidate["candidate_id"], source

    def publish(self, candidate_id, source, recipe=RECIPE):
        return self.store.package(self.run_id, candidate_id, source, recipe)

    def test_copy_failure_can_be_retried_without_partial_publication(self):
        candidate_id, source = self.prepare()

        def fail_copy(source, destination, *args):
            destination.mkdir()
            shutil.copy2(source / "engine.fixture", destination / "engine.fixture")
            raise OSError("injected copy failure")

        with patch.object(run_store, "copy_package", side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, "injected copy failure"):
                self.publish(candidate_id, source)
        view = self.store.recover(self.run_id)
        self.assertEqual(view["errors"], [])
        self.assertIsNone(view["candidates"][0]["identities"]["package_sha256"])
        package = self.publish(candidate_id, source)
        self.assertEqual(self.store.get(self.run_id)["candidates"][0]["identities"]["package_sha256"], digest(package))

    def test_rejected_recipe_does_not_strand_the_package_destination(self):
        candidate_id, source = self.prepare()
        with self.assertRaisesRegex(ValueError, "launch entry"):
            self.publish(candidate_id, source, RECIPE | {"launch": ["absent.fixture"]})
        self.publish(candidate_id, source)
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

    def test_entire_work_directory_can_be_packaged_without_copying_into_itself(self):
        candidate = self.candidate()
        source = self.directory / "work"
        (source / "engine.fixture").write_text("package from work", encoding="utf-8")
        copytree = shutil.copytree
        calls = []

        def bounded_copy(*args, **kwargs):
            calls.append(args)
            if len(calls) > 5:
                raise OSError("recursive copy of staging into itself")
            return copytree(*args, **kwargs)

        with patch.object(shutil, "copytree", side_effect=bounded_copy):
            package = self.publish(candidate["candidate_id"], source)
        self.assertEqual({row["path"] for row in package["files"]}, {"engine.fixture", "configuration.json"})
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

    def test_copy_must_match_the_input_inventory_before_adding_configuration(self):
        candidate_id, source = self.prepare()
        copy_package = run_store.copy_package

        def changed_copy(source, destination, *args):
            copy_package(source, destination, *args)
            (destination / "engine.fixture").write_text("different copied bytes", encoding="utf-8")

        with patch.object(run_store, "copy_package", side_effect=changed_copy):
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                self.publish(candidate_id, source)
        self.assertIsNone(self.store.get(self.run_id)["candidates"][0]["identities"]["package_sha256"])
        self.publish(candidate_id, source)

    def test_windows_reader_contention_retries_directory_publication(self):
        candidate_id, source = self.prepare()
        rename = Path.rename
        denied = PermissionError("reader holds staged package")
        denied.winerror = 32
        calls = []

        def transient(path, destination):
            calls.append(destination)
            if len(calls) == 1:
                raise denied
            return rename(path, destination)

        with patch.object(Path, "rename", transient):
            self.publish(candidate_id, source)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

    def test_abrupt_exit_during_copy_can_be_recovered_and_retried(self):
        candidate_id, source = self.prepare()
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--crash-copy",
             str(self.store.root), self.run_id, candidate_id, str(source)],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 73, result.stdout + result.stderr)
        self.assertTrue((self.directory / ".coordinator.lock").exists())
        self.assertEqual(self.store.recover(self.run_id)["errors"], [])
        package = self.publish(candidate_id, source)
        self.assertEqual(self.store.get(self.run_id)["candidates"][0]["identities"]["package_sha256"], digest(package))

    def test_committed_package_retry_preserves_bytes_and_rejects_changed_input(self):
        candidate_id, source = self.prepare()
        with patch.object(self.store, "_refresh", side_effect=OSError("injected projection failure")):
            with self.assertRaisesRegex(OSError, "injected projection failure"):
                self.publish(candidate_id, source)
        folder = self.directory / "candidates" / candidate_id / "package"
        retained = inventory(folder)
        self.assertEqual(self.store.recover(self.run_id)["errors"], [])
        package = self.publish(candidate_id, source)
        self.assertEqual(package, read_json(folder / "manifest.json"))
        self.assertEqual(inventory(folder), retained)
        with self.assertRaises(ValueError):
            self.publish(candidate_id, source, RECIPE | {"capabilities": ["decode"]})
        (source / "engine.fixture").write_text("changed bytes", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.publish(candidate_id, source)
        self.assertEqual(inventory(folder), retained)


class RecoveryOwnerTests(RunFixture, unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows retains exited process objects while handles remain open")
    def test_exited_owner_is_recoverable_while_its_process_handle_is_retained(self):
        with subprocess.Popen([sys.executable, "-B", "-c", "pass"]) as process:
            process.wait(timeout=20)
            owner = run_store.lease_owner() | {"pid": process.pid}
            (self.directory / ".coordinator.lock").write_text(json.dumps(owner), encoding="utf-8")
            self.assertEqual(self.store.recover(self.run_id)["errors"], [])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--crash-copy":
        def crash_copy(source, destination, *args):
            destination.mkdir()
            shutil.copy2(source / "engine.fixture", destination / "engine.fixture")
            os._exit(73)

        run_store.copy_package = crash_copy
        run_store.RunStore(sys.argv[2]).package(sys.argv[3], sys.argv[4], Path(sys.argv[5]), RECIPE)
    else:
        unittest.main()
