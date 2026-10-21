"""The resident host's channel: authentication, ordered delivery, routing and bounds."""
import os
from pathlib import Path
import socket
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "cli"):
    sys.path.insert(0, str(ROOT / _folder))
from crit_api import channel  # noqa: E402


class ChannelTests(unittest.TestCase):
    def setUp(self):
        self.supervisor = channel.Supervisor("owner-token")
        self.addCleanup(self.supervisor.close)

    def host(self, token="owner-token"):
        connection = channel.connect(self.supervisor.port, token)
        self.addCleanup(connection.close)
        return connection

    def wait(self, box):
        self.assertTrue(box.ready.wait(5), "nothing arrived")
        return self.supervisor.take(box)

    def test_commands_wait_for_the_authenticated_host_and_arrive_in_order(self):
        self.supervisor.send({"kind": "command", "command_id": "a"})
        intruder = channel.Connection(socket.create_connection(("127.0.0.1", self.supervisor.port)))
        self.addCleanup(intruder.close)
        intruder.send({"kind": "hello", "token": "guess", "pid": os.getpid()})
        with self.assertRaises((EOFError, OSError)):
            intruder.receive(channel.STATUS_BYTES)  # dropped without a command
        nameless = channel.Connection(socket.create_connection(("127.0.0.1", self.supervisor.port)))
        self.addCleanup(nameless.close)
        nameless.send({"kind": "hello", "token": "owner-token"})
        with self.assertRaises((EOFError, OSError)):
            nameless.receive(channel.STATUS_BYTES)  # the host names its process
        self.assertIsNone(self.supervisor.pid)
        host = self.host()
        self.supervisor.send({"kind": "cancel", "command_id": "a"})
        self.assertEqual([host.receive(channel.STATUS_BYTES) for _ in range(2)],
                         [{"kind": "command", "command_id": "a"}, {"kind": "cancel", "command_id": "a"}])
        self.assertIsNone(self.supervisor.failure)
        self.assertEqual(self.supervisor.pid, os.getpid())

    def test_pieces_and_results_reach_their_command_and_status_is_kept(self):
        host = self.host()
        first, second = self.supervisor.register("first"), self.supervisor.register("second")
        host.send({"kind": "status", "status": {"state": "generating"}})
        host.send({"kind": "progress", "command_id": "gone", "text": "late"})  # a command that already returned
        host.send({"kind": "progress", "command_id": "first", "text": "a"})
        host.send({"kind": "progress", "command_id": "second", "text": "x"})
        host.send({"kind": "progress", "command_id": "first", "text": "b"})
        host.send({"kind": "result", "command_id": "first", "status": "completed"})
        pieces, result = self.wait(first)
        while result is None:
            more, result = self.wait(first)
            pieces += more
        self.assertEqual([piece["text"] for piece in pieces], ["a", "b"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual([piece["text"] for piece in self.wait(second)[0]], ["x"])
        self.assertEqual(self.supervisor.status, {"state": "generating"})
        self.assertFalse(first.ready.is_set())

    def test_a_result_is_bounded_by_its_command_and_a_violation_fails_every_command(self):
        host = self.host()
        large = self.supervisor.register("large", 4 << 20)
        waiting = self.supervisor.register("waiting")
        host.send({"kind": "result", "command_id": "large", "response": "x" * (3 << 20)})
        self.assertEqual(len(self.wait(large)[1]["response"]), 3 << 20)
        self.supervisor.unregister("large")
        try:
            host.send({"kind": "result", "command_id": "waiting", "response": "x" * (2 << 20)})
        except OSError:
            pass  # crit serve may close the channel before the host has written it all
        self.assertTrue(waiting.ready.wait(5))
        self.assertIn("exceeds", self.supervisor.failure)
        with self.assertRaisesRegex(ValueError, "channel failed"):
            self.supervisor.send({"kind": "cancel", "command_id": "waiting"})

    def test_unknown_messages_fail_the_channel_and_close_ends_the_host(self):
        host = self.host()
        box = self.supervisor.register("command")
        host.send({"kind": "surprise"})
        self.assertTrue(box.ready.wait(5))
        self.assertIn("Unknown resident message", self.supervisor.failure)
        with self.assertRaises((EOFError, OSError)):
            host.receive(channel.STATUS_BYTES)

    def test_the_host_bounds_what_it_receives(self):
        host = self.host()
        self.supervisor.send({"kind": "command", "payload": "x" * 100})
        with self.assertRaisesRegex(ValueError, "exceeds 64"):
            host.receive(64)
        with self.assertRaisesRegex(ValueError, "command exceeds"):
            self.supervisor.send({"kind": "command", "payload": "x" * 100}, 64)


if __name__ == "__main__":
    unittest.main()
