import asyncio
import json
import re
import sys
import unittest
from unittest import mock

from PIL import Image

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = mock.MagicMock()

from banana2_batch_node import (
    Banana2ExecutionError,
    GeminiBatchNode,
    _is_retryable_error,
)
from doubao_batch_node import DoubaoBatchNode


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

    def test_transient_poll_failure_retries_submission(self):
        results = [
            _result(retryable=True, message="任务失败: Timeout awaiting 'connect' for 20000ms"),
            _result(success=True, message="ok"),
        ]

        async def fake_execute(task, config, session):
            result = results.pop(0)
            result["stage"] = "poll"
            return result

        with (
            mock.patch.object(self.node, "_execute_single_task", side_effect=fake_execute) as execute,
            mock.patch("banana2_batch_node.requests.Session", return_value=_FakeSession()),
            mock.patch("banana2_batch_node._retry_delay_seconds", return_value=0),
            mock.patch("banana2_batch_node.asyncio.sleep", new=mock.AsyncMock()),
        ):
            result = asyncio.run(
                self.node._execute_single_task_with_retry(
                    self.task,
                    {"retry_count": 1},
                )
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(execute.await_count, 2)

    def test_connect_timeout_is_retryable(self):
        self.assertTrue(_is_retryable_error("Timeout awaiting 'connect' for 20000ms"))

    def test_zhifou_poll_connect_timeout_is_marked_retryable(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {
            "data": {
                "status": "FAILURE",
                "fail_reason": "Timeout awaiting 'connect' for 20000ms",
            }
        }
        session = mock.Mock()
        session.get.return_value = response

        result = asyncio.run(
            self.node._poll_task_status(
                group_id=1,
                task_id="task-id",
                config={
                    "provider": "zhifou",
                    "base_url": "https://zhiai.art/api",
                    "api_key": "test-key",
                    "response_format": "url",
                    "timeout": 60,
                },
                session=session,
            )
        )

        self.assertFalse(result["success"])
        self.assertTrue(result["retryable"])
        self.assertEqual(result["stage"], "poll")
        self.assertIn("Timeout awaiting 'connect'", result["info"])


class MultipartFilenameTests(unittest.TestCase):
    def setUp(self):
        self.node = GeminiBatchNode()
        self.task = {
            "group_id": 3,
            "images": [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))],
            "prompt": "test",
        }
        self.config = {
            "provider": "zhifou",
            "base_url": "https://zhiai.art/api",
            "api_key": "test-key",
            "model": "Nano Banana 2",
            "mode": "Img2Img",
            "aspect_ratio": "1:1",
            "response_format": "url",
            "img_size": "2K",
            "img_n": 1,
        }

    def test_multipart_filenames_are_unique_per_request(self):
        _, _, first_payload = self.node._build_api_request(self.task, self.config)
        _, _, second_payload = self.node._build_api_request(self.task, self.config)

        first_names = [item[0] for item in first_payload["files"]["image"]]
        second_names = [item[0] for item in second_payload["files"]["image"]]

        self.assertEqual(len(set(first_names + second_names)), 4)
        self.assertRegex(
            first_names[0],
            re.compile(r"^banana_[0-9a-f]{32}_g03_i01\.png$"),
        )
        self.assertRegex(
            first_names[1],
            re.compile(r"^banana_[0-9a-f]{32}_g03_i02\.png$"),
        )


class DoubaoUploadFilenameTests(unittest.TestCase):
    def test_upload_filename_is_unique_and_requests_sets_boundary(self):
        node = DoubaoBatchNode()
        response = mock.Mock(status_code=200)
        response.json.return_value = {"data": {"url": "https://example.com/image.png"}}
        node.session = mock.Mock()
        node.session.post.return_value = response
        image = Image.new("RGB", (2, 2))

        asyncio.run(node._upload_image_to_doubao(image, {"api_key": "test-key"}))
        first_call = node.session.post.call_args
        first_name = first_call.kwargs["files"]["file"][0]

        asyncio.run(node._upload_image_to_doubao(image, {"api_key": "test-key"}))
        second_call = node.session.post.call_args
        second_name = second_call.kwargs["files"]["file"][0]

        self.assertNotEqual(first_name, second_name)
        self.assertRegex(first_name, re.compile(r"^doubao_[0-9a-f]{32}\.png$"))
        self.assertNotIn("Content-Type", first_call.kwargs["headers"])


if __name__ == "__main__":
    unittest.main()
