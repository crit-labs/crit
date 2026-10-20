"""Attempt contracts agree before publication, in projections and during recovery."""
from datetime import timedelta
import unittest

from test_run_records import RunFixture
from crit.run.records import candidate_projection, file_sha, read_json, record, timestamp, write_json


class AttemptFixture(RunFixture):
    def reservations(self):
        candidate = self.candidate()
        return [self.store.reserve(self.run_id, candidate["candidate_id"], "build", 10),
                self.store.reserve(self.run_id, None, "agent", 10),
                self.store.reserve(self.run_id, None, "discovery", 10)]

    def path(self, reservation, name="result.json"):
        owner = (f"candidates/{reservation['candidate_id']}" if reservation["candidate_id"]
                 else f"segments/{reservation['segment_id']}")
        return self.directory / owner / "attempts" / reservation["attempt_id"] / name

    def payload(self, reservation, **changes):
        return {"outcome": "failed", "ended_at": reservation["reserved_at"], "exit_code": 1,
                "cleanup_settled": True, "logs": [], "failure": "fixture failure", **changes}

    def result(self, reservation, data):
        return record("attempt_result", self.run_id,
                      **{key: reservation[key] for key in ("segment_id", "candidate_id", "attempt_id")}, **data)

    def assert_readers_reject(self, reservation, value):
        path = self.path(reservation)
        candidate_id = reservation["candidate_id"]
        if candidate_id:
            with self.assertRaises(ValueError):
                candidate_projection(self.directory, self.run, candidate_id,
                                     {path.relative_to(self.directory).as_posix(): value})
        # Only synthetic fixtures alter retained authority, to exercise the read boundary.
        write_json(path, value)
        try:
            with self.assertRaises(ValueError):
                self.store._attempts(self.directory)
            with self.assertRaises(ValueError):
                self.store.recover(self.run_id)
        finally:
            path.unlink()

    def assert_rejected_without_publication(self, reservation, data):
        path = self.path(reservation)
        state = (self.directory / "state.json").read_bytes()
        events = {p.name: p.read_bytes() for p in (self.directory / "events").glob("*.json")}
        reserved = self.path(reservation, "reservation.json").read_bytes()
        try:
            with self.assertRaises(ValueError):
                self.store.settle(self.run_id, reservation, data)
            self.assertFalse(path.exists(), "invalid result was published before validation finished")
            self.assertEqual((self.directory / "state.json").read_bytes(), state)
            self.assertEqual({p.name: p.read_bytes() for p in (self.directory / "events").glob("*.json")}, events)
            self.assertEqual(self.path(reservation, "reservation.json").read_bytes(), reserved)
        finally:
            if path.exists():
                path.unlink()
        self.assert_readers_reject(reservation, self.result(reservation, data))


class AttemptResultTests(AttemptFixture, unittest.TestCase):
    def test_malformed_result_fields_never_publish_and_all_readers_reject(self):
        for reservation in self.reservations():
            for changes in ({"logs": {}}, {"logs": ""}, {"logs": None}, {"failure": 12}, {"failure": ""},
                            {"cleanup_settled": 1}, {"outcome": "unknown"}, {"outcome": []}, {"exit_code": True},
                            {"exit_code": 1.5}, {"exit_code": -(2**31) - 1}, {"exit_code": 2**32 + 1}):
                with self.subTest(operation=reservation["operation"], changes=changes):
                    self.assert_rejected_without_publication(reservation, self.payload(reservation, **changes))

    def test_time_and_completion_rules_agree_before_publication_and_on_recovery(self):
        for reservation in self.reservations():
            before = (timestamp(reservation["reserved_at"], "start") - timedelta(microseconds=1)).isoformat()
            late = (timestamp(reservation["deadline_at"], "deadline") + timedelta(microseconds=1)).isoformat()
            for changes in ({"ended_at": before}, {"ended_at": "invalid"},
                            {"outcome": "completed", "exit_code": 0, "ended_at": late},
                            {"outcome": "completed", "exit_code": 1},
                            {"outcome": "completed", "exit_code": None},
                            {"outcome": "completed", "exit_code": 0, "cleanup_settled": False}):
                with self.subTest(operation=reservation["operation"], changes=changes):
                    self.assert_rejected_without_publication(reservation, self.payload(reservation, **changes))

    def test_result_identity_and_schema_are_checked_by_every_reader(self):
        for reservation in self.reservations():
            value = self.result(reservation, self.payload(reservation))
            for changes in ({"schema_version": True}, {"record_kind": "other"}, {"run_id": "other"},
                            {"segment_id": "segment-9999"}, {"attempt_id": "attempt-9999"},
                            {"candidate_id": "candidate-9999"}, {"unknown": 1}, {"extensions": []}):
                with self.subTest(operation=reservation["operation"], changes=changes):
                    self.assert_readers_reject(reservation, value | changes)

    def test_log_references_bind_retained_bytes_on_every_path(self):
        log = self.directory / "captured/attempt.log"
        log.write_text("retained fixture log", encoding="utf-8")
        for reservation in self.reservations():
            for ref in ({"path": "captured/absent.log", "sha256": file_sha(log)},
                        {"path": "captured/attempt.log", "sha256": "0" * 64},
                        {"path": "../attempt.log", "sha256": file_sha(log)}):
                with self.subTest(operation=reservation["operation"], ref=ref):
                    self.assert_rejected_without_publication(reservation, self.payload(reservation, logs=[ref]))

    def test_candidate_failure_requires_a_message(self):
        reservation = self.reservations()[0]
        self.assert_rejected_without_publication(reservation, self.payload(reservation, failure=None))

    def test_valid_results_preserve_bytes_and_projection_through_recovery(self):
        candidate = self.candidate()
        log = self.directory / "captured/attempt.log"
        log.write_text("retained fixture log", encoding="utf-8")
        logs = [{"path": "captured/attempt.log", "sha256": file_sha(log)}]
        for operation in ("build", "agent", "discovery"):
            candidate_id = candidate["candidate_id"] if operation == "build" else None
            for outcome in ("failed", "cancelled", "timed_out", "interrupted", "launch_failed", "completed"):
                with self.subTest(operation=operation, outcome=outcome):
                    reservation = self.store.reserve(self.run_id, candidate_id, operation, 10)
                    # Successful completion at the boundary is valid; failed cleanup can finish late.
                    ended = timestamp(reservation["deadline_at"], "deadline")
                    if outcome != "completed":
                        ended += timedelta(seconds=1)
                    data = self.payload(reservation, outcome=outcome, ended_at=ended.isoformat(), logs=logs,
                                        exit_code=0 if outcome == "completed" else None,
                                        cleanup_settled=outcome == "completed",
                                        failure="fixture failure" if candidate_id and outcome != "completed" else None,
                                        extensions={"fixture": True})
                    expected = self.result(reservation, data)
                    self.assertEqual(self.store.settle(self.run_id, reservation, data), expected)
                    self.assertEqual(read_json(self.path(reservation)), expected)
                    before = {p: p.read_bytes() for p in
                              (self.path(reservation), self.path(reservation, "reservation.json"), log)}
                    projection = candidate_projection(self.directory, self.run, candidate["candidate_id"])
                    recovered = self.store.recover(self.run_id)
                    self.assertEqual(recovered["errors"], [])
                    self.assertEqual(candidate_projection(self.directory, self.run, candidate["candidate_id"]), projection)
                    self.assertEqual({p: p.read_bytes() for p in before}, before)
                    self.assertIn(reservation, self.store._attempts(self.directory))


class AttemptReservationTests(AttemptFixture, unittest.TestCase):
    def mutations(self, reservation):
        start = timestamp(reservation["reserved_at"], "start")
        before = (timestamp(self.view["segment"]["started_at"], "start") - timedelta(microseconds=1)).isoformat()
        late = (timestamp(self.view["segment"]["deadline_at"], "deadline") + timedelta(microseconds=1)).isoformat()
        return ({"cost": {"unit": "attempt", "amount": 0}}, {"cost": {"unit": "attempt", "amount": True}},
                {"cost": {"unit": "attempt", "amount": 1.0}}, {"operation": "unknown"},
                {"operation": "agent" if reservation["candidate_id"] else "build"},
                {"reserved_at": before}, {"reserved_at": "invalid"}, {"deadline_at": late},
                {"deadline_at": (start - timedelta(microseconds=1)).isoformat()},
                {"schema_version": 2}, {"record_kind": "other"}, {"run_id": "other"})

    def assert_reservation_rejected(self, original, value, *, settlement=False):
        path = self.path(original, "reservation.json")
        retained = path.read_bytes()
        write_json(path, value)  # Corrupt synthetic authority to exercise its validation boundary.
        result_path = self.path(original)
        try:
            if settlement:
                state = (self.directory / "state.json").read_bytes()
                events = {p.name: p.read_bytes() for p in (self.directory / "events").glob("*.json")}
                with self.assertRaises(ValueError):
                    self.store.settle(self.run_id, value, self.payload(value))
                self.assertFalse(result_path.exists(), "invalid reservation allowed immutable result publication")
                self.assertEqual((self.directory / "state.json").read_bytes(), state)
                self.assertEqual({p.name: p.read_bytes() for p in (self.directory / "events").glob("*.json")}, events)
            else:
                if original["candidate_id"]:
                    with self.assertRaises(ValueError):
                        candidate_projection(self.directory, self.run, original["candidate_id"])
                with self.assertRaises(ValueError):
                    self.store._attempts(self.directory)
                with self.assertRaises(ValueError):
                    self.store.recover(self.run_id)
        finally:
            path.write_bytes(retained)
            if result_path.exists():
                result_path.unlink()

    def test_invalid_reservations_reject_settlement_before_publication(self):
        for reservation in self.reservations():
            for changes in self.mutations(reservation):
                with self.subTest(operation=reservation["operation"], changes=changes):
                    self.assert_reservation_rejected(reservation, reservation | changes, settlement=True)

    def test_reservation_rules_agree_during_projection_scanning_and_recovery(self):
        for reservation in self.reservations():
            for changes in (*self.mutations(reservation), {"attempt_id": "attempt-9999"},
                            {"segment_id": "segment-9999"}, {"candidate_id": "candidate-9999"},
                            {"unknown": 1}, {"extensions": []}):
                with self.subTest(operation=reservation["operation"], changes=changes):
                    self.assert_reservation_rejected(reservation, reservation | changes)

    def test_process_caps_apply_even_inside_a_longer_segment(self):
        self.view = self.store.create(self.data | {"budget_minutes": 180})
        self.run = self.view["run"]
        self.run_id = self.run["run_id"]
        self.directory = self.store.directory(self.run_id)
        for reservation in self.reservations():
            cap = self.view["segment"]["limits"]["process_seconds"][reservation["operation"]]
            # The agent cap equals this segment's duration; exercise shorter operation caps here.
            if reservation["operation"] == "agent":
                continue
            deadline = (timestamp(reservation["reserved_at"], "start") + timedelta(seconds=cap, microseconds=1)).isoformat()
            value = reservation | {"deadline_at": deadline}
            with self.subTest(operation=reservation["operation"]):
                self.assert_reservation_rejected(reservation, value)
                self.assert_reservation_rejected(reservation, value, settlement=True)

    def test_candidate_identity_checks_remain_in_projection(self):
        candidate = self.candidate()
        self.package(candidate)
        reservation = self.store.reserve(self.run_id, candidate["candidate_id"], "check", 10)
        before_creation = (timestamp(candidate["created_at"], "created") - timedelta(microseconds=1)).isoformat()
        for changes in ({"inputs_manifest_sha256": "0" * 64}, {"package_sha256": "0" * 64},
                        {"reserved_at": before_creation}):
            with self.subTest(changes=changes):
                path = self.path(reservation, "reservation.json")
                value = reservation | changes
                with self.assertRaises(ValueError):
                    candidate_projection(self.directory, self.run, candidate["candidate_id"],
                                         {path.relative_to(self.directory).as_posix(): value})
                self.assert_reservation_rejected(reservation, value, settlement=True)

    def test_all_operations_preserve_reservations_and_builds_may_precede_packages(self):
        candidate = self.candidate()
        candidate_id = candidate["candidate_id"]
        build = self.store.reserve(self.run_id, candidate_id, "build", 10)
        self.assertIsNone(build["package_sha256"])
        package = self.package(candidate)
        reservations = [build] + [self.store.reserve(self.run_id, candidate_id, operation, 10)
                                  for operation in ("check", "benchmark")]
        reservations += [self.store.reserve(self.run_id, None, operation, 10) for operation in ("agent", "discovery")]
        for reservation in reservations:
            self.store.settle(self.run_id, reservation, self.payload(
                reservation, outcome="completed", exit_code=0, failure=None))
        paths = [self.path(r, name) for r in reservations for name in ("reservation.json", "result.json")]
        retained = {p: p.read_bytes() for p in paths}
        projection = candidate_projection(self.directory, self.run, candidate_id)
        self.assertEqual(projection["attempt_count"], 3)
        self.assertFalse(projection["unresolved_cleanup"])
        self.assertIsNotNone(projection["identities"]["package_sha256"])
        self.assertEqual(read_json(self.directory / f"candidates/{candidate_id}/package/manifest.json"), package)
        self.assertEqual(self.store.recover(self.run_id)["state"]["budget"]["attempts"], 5)
        self.assertEqual(candidate_projection(self.directory, self.run, candidate_id), projection)
        self.assertEqual({p: p.read_bytes() for p in paths}, retained)
        self.assertCountEqual(self.store._attempts(self.directory), reservations)


if __name__ == "__main__":
    unittest.main()
