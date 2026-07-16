import asyncio
import json
import unittest
from unittest import mock

from banana2_batch_node import (
    Banana2ExecutionError,
    GeminiBatchNode,
    _is_retryable_error,
)


def _result(success=False, retryable=False, message="error"):
    return {
        "group_id": 1,
        "success": success,
        "image": None,
        "url": "",
        "response_code": 1 if success else 2,
        "retryable": retryable,
        "info": json.dumps({"status": "error", "message": message}),
    }


class _FakeSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class RetryClassificationTests(unittest.TestCase):
    def test_retryable_network_and_http_errors(self):
        self.assertTrue(_is_retryable_error("SSLError: EOF occurred in violation of protocol"))
        self.assertTrue(_is_retryable_error("connection reset by peer"))
        self.assertTrue(_is_retryable_error(http_status=429))
        self.assertTrue(_is_retryable_error(http_status=503))

    def test_permanent_errors_are_not_retried(self):
        self.assertFalse(_is_retryable_error("invalid api key"))
        self.assertFalse(_is_retryable_error("参数错误"))
        self.assertFalse(_is_retryable_error(http_status=400))
        self.assertFalse(_is_retryable_error(http_status=403))


class RetryExecutionTests(unittest.TestCase):
    def setUp(self):
        self.node = GeminiBatchNode()
        self.task = {"group_id": 1}

    def test_transient_failure_retries_with_new_session(self):
        results = [
            _result(retryable=True, message="SSL EOF"),
            _result(success=True, message="ok"),
        ]
        sessions = []

        async def fake_execute(task, config, session):
            return results.pop(0)

        def new_session():
            session = _FakeSession()
            sessions.append(session)
            return session

        with (
            mock.patch.object(self.node, "_execute_single_task", side_effect=fake_execute),
            mock.patch("banana2_batch_node.requests.Session", side_effect=new_session),
            mock.patch("banana2_batch_node._retry_delay_seconds", return_value=0),
            mock.patch("banana2_batch_node.asyncio.sleep", new=mock.AsyncMock()),
        ):
            result = asyncio.run(
                self.node._execute_single_task_with_retry(
                    self.task,
                    {"retry_count": 2},
                )
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(len(sessions), 2)
        self.assertTrue(all(session.closed for session in sessions))

    def test_permanent_failure_stops_immediately(self):
        async def fake_execute(task, config, session):
            return _result(retryable=False, message="invalid api key")

        with (
            mock.patch.object(self.node, "_execute_single_task", side_effect=fake_execute) as execute,
            mock.patch("banana2_batch_node.requests.Session", return_value=_FakeSession()),
        ):
            result = asyncio.run(
                self.node._execute_single_task_with_retry(
                    self.task,
                    {"retry_count": 5},
                )
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(execute.await_count, 1)

    def test_all_failed_tasks_raise_from_gemini_batch_node(self):
        self.node._parse_tasks = mock.Mock(return_value=[{"group_id": 1, "is_valid": True}])

        async def fake_execute_tasks(tasks, config):
            return [_result(retryable=True, message="SSL EOF")]

        self.node._execute_tasks_async = fake_execute_tasks

        with self.assertRaises(Banana2ExecutionError) as raised:
            self.node.execute(node_enabled=True)

        self.assertIn("所有有效任务均失败", str(raised.exception))
        self.assertIn("SSL EOF", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
