import asyncio
import json
import re
import sys
import threading
import time
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
from model_compare_node import ModelCompareNode


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


class TimeoutInputLimitTests(unittest.TestCase):
    def test_all_nodes_allow_one_hour_timeout(self):
        for node_class in (GeminiBatchNode, DoubaoBatchNode, ModelCompareNode):
            timeout_config = node_class.INPUT_TYPES()["required"]["timeout"][1]
            self.assertEqual(timeout_config["min"], 10)
            self.assertEqual(timeout_config["max"], 3600)


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


class EasyAIAsyncTests(unittest.TestCase):
    def setUp(self):
        self.node = GeminiBatchNode()
        self.task = {
            "group_id": 1,
            "images": [Image.new("RGB", (2, 2))],
            "prompt": "test prompt",
        }
        self.config = {
            "provider": "zhifou",
            "base_url": "https://zhiai.art/api",
            "api_key": "test-key",
            "model": "Nano Banana 2",
            "mode": "Img2Img",
            "aspect_ratio": "3:4",
            "response_format": "url",
            "img_size": "2K",
            "img_n": 1,
            "timeout": 10,
        }

    def test_zhifou_builds_easyai_async_multipart_request(self):
        url, headers, payload = self.node._build_api_request(self.task, self.config)

        self.assertEqual(url, "https://zhiai.art/api/v1/images/edits")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["x-async"], "true")
        self.assertEqual(payload["data"]["model"], "Nano Banana 2")
        self.assertEqual(len(payload["files"]["image"]), 1)

    def test_easyai_submit_response_extracts_task_id(self):
        response_data = {
            "task_id": "task-123",
            "status": "submitted",
        }

        async def fake_poll(group_id, task_id, config, session):
            self.assertEqual(group_id, 1)
            self.assertEqual(task_id, "task-123")
            return {"group_id": group_id, "success": True, "url": "ok"}

        with mock.patch.object(self.node, "_poll_task_status", side_effect=fake_poll):
            result = asyncio.run(
                self.node._handle_async_response(
                    1, response_data, self.config, mock.Mock()
                )
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["url"], "ok")

    def test_easyai_success_response_parses_output_url(self):
        result = self.node._parse_easyai_success_response(
            1,
            {
                "task_id": "task-123",
                "status": "success",
                "output": ["https://example.com/result.png"],
            },
            "url",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["url"], "https://example.com/result.png")
        self.assertEqual(result["response_code"], 1)

    def test_easyai_poll_uses_result_endpoint_until_success(self):
        processing = mock.Mock(status_code=200)
        processing.json.return_value = {
            "task_id": "task-123",
            "status": "process",
        }
        success = mock.Mock(status_code=200)
        success.json.return_value = {
            "task_id": "task-123",
            "status": "success",
            "output": ["https://example.com/result.png"],
        }
        session = mock.Mock()
        session.get.side_effect = [processing, success]

        with mock.patch("banana2_batch_node.asyncio.sleep", new=mock.AsyncMock()):
            result = asyncio.run(
                self.node._poll_task_status(1, "task-123", self.config, session)
            )

        self.assertTrue(result["success"])
        self.assertEqual(
            session.get.call_args_list[0].args[0],
            "https://zhiai.art/api/v1/ai/result/task-123",
        )

    def test_gemini_keeps_ten_way_batch_concurrency_for_easyai(self):
        active = 0
        max_active = 0

        async def fake_execute(task, config):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)
            active -= 1
            return {
                "group_id": task["group_id"],
                "success": True,
                "image": None,
                "url": "ok",
                "response_code": 1,
                "info": "ok",
            }

        tasks = [{"group_id": index} for index in range(1, 13)]
        with mock.patch.object(self.node, "_execute_single_task_with_retry", side_effect=fake_execute):
            results = asyncio.run(
                self.node._execute_tasks_async(
                    tasks,
                    {"provider": "zhifou"},
                )
            )

        self.assertEqual(len(results), 12)
        self.assertEqual(max_active, 10)

    def test_easyai_end_to_end_simulation_uses_async_submit_and_poll(self):
        active_posts = 0
        max_active_posts = 0
        state_lock = threading.Lock()
        sessions = []

        class FakeResponse:
            def __init__(self, body):
                self.status_code = 200
                self.headers = {"Content-Type": "application/json"}
                self._body = body
                self.text = json.dumps(body)

            def json(self):
                return self._body

        class FakeSession:
            def __init__(self, session_number):
                self.task_id = f"task-{session_number}"
                self.closed = False
                self.post_calls = []
                self.get_calls = []

            def post(self, url, **kwargs):
                nonlocal active_posts, max_active_posts
                self.post_calls.append((url, kwargs))
                with state_lock:
                    active_posts += 1
                    max_active_posts = max(max_active_posts, active_posts)
                time.sleep(0.02)
                with state_lock:
                    active_posts -= 1
                return FakeResponse({"task_id": self.task_id, "status": "submitted"})

            def get(self, url, **kwargs):
                self.get_calls.append((url, kwargs))
                return FakeResponse({
                    "task_id": self.task_id,
                    "status": "success",
                    "output": [f"https://example.com/{self.task_id}.png"],
                })

            def close(self):
                self.closed = True

        def new_session():
            session = FakeSession(len(sessions) + 1)
            sessions.append(session)
            return session

        tasks = [
            {"group_id": index, "images": [], "prompt": f"prompt-{index}", "is_valid": True}
            for index in range(1, 11)
        ]
        config = {
            "provider": "zhifou",
            "base_url": "https://zhiai.art/api",
            "api_key": "test-key",
            "model": "Nano Banana 2",
            "mode": "Text2Img",
            "aspect_ratio": "3:4",
            "response_format": "url",
            "img_size": "2K",
            "img_n": 1,
            "timeout": 10,
            "retry_count": 0,
        }

        with mock.patch("banana2_batch_node.requests.Session", side_effect=new_session):
            results = asyncio.run(self.node._execute_tasks_async(tasks, config))

        self.assertEqual(len(results), 10)
        self.assertTrue(all(result["success"] for result in results))
        self.assertEqual(max_active_posts, 10)
        self.assertTrue(all(session.closed for session in sessions))
        self.assertTrue(all(session.post_calls[0][0].endswith("/v1/images/generations") for session in sessions))
        self.assertTrue(all(session.post_calls[0][1]["headers"]["x-async"] == "true" for session in sessions))
        self.assertTrue(all(session.get_calls[0][0].startswith("https://zhiai.art/api/v1/ai/result/") for session in sessions))


class OtherNodeEasyAIAsyncTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "group_id": 1,
            "images": [],
            "prompt": "test prompt",
        }
        self.config = {
            "provider": "zhifou",
            "base_url": "https://zhiai.art/api",
            "api_key": "test-key",
            "model": "doubao-seedream-4-5-251128",
            "mode": "Text2Img",
            "aspect_ratio": "3:4",
            "response_format": "url",
            "img_size": "2K",
            "n": 1,
            "watermark": False,
            "stream": False,
            "timeout": 10,
            "retry_count": 0,
        }

    def test_doubao_builds_easyai_header_without_query_parameter(self):
        node = DoubaoBatchNode()
        url, headers, _ = node._build_api_request(self.task, self.config)

        self.assertEqual(url, "https://zhiai.art/api/v1/images/generations")
        self.assertEqual(headers["x-async"], "true")
        self.assertNotIn("async=true", url)

    def test_doubao_routes_easyai_polling_to_shared_protocol(self):
        node = DoubaoBatchNode()

        async def fake_poll(group_id, task_id, config):
            self.assertEqual(group_id, 1)
            self.assertEqual(task_id, "task-123")
            return {"group_id": group_id, "success": True, "url": "ok"}

        with mock.patch.object(node, "_poll_easyai_task_status", side_effect=fake_poll):
            result = asyncio.run(
                node._poll_doubao_task_status(1, "task-123", self.config)
            )

        self.assertTrue(result["success"])

    def test_doubao_non_easyai_polling_uses_configured_timeout(self):
        node = DoubaoBatchNode()
        response = mock.Mock(status_code=500, text="temporary failure")
        node.session = mock.Mock()
        node.session.get.return_value = response
        config = dict(self.config, provider="comfly", timeout=11)

        with mock.patch("doubao_batch_node.asyncio.sleep", new=mock.AsyncMock()):
            result = asyncio.run(
                node._poll_doubao_task_status(1, "task-123", config)
            )

        self.assertFalse(result["success"])
        self.assertEqual(node.session.get.call_count, 3)
        self.assertIn("11秒", result["info"])

    def test_doubao_easyai_shared_poller_runs_with_fake_result(self):
        node = DoubaoBatchNode()
        response = mock.Mock(status_code=200)
        response.json.return_value = {
            "task_id": "task-123",
            "status": "success",
            "output": ["https://example.com/doubao-result.png"],
        }
        node.session = mock.Mock()
        node.session.get.return_value = response

        result = asyncio.run(
            node._poll_doubao_task_status(1, "task-123", self.config)
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["url"], "https://example.com/doubao-result.png")
        self.assertEqual(
            node.session.get.call_args.args[0],
            "https://zhiai.art/api/v1/ai/result/task-123",
        )

    def test_model_compare_routes_zhifou_to_gemini_compatibility_path(self):
        node = ModelCompareNode()
        task = {"images": [], "prompt": "test prompt"}
        config = {
            "provider": "zhifou",
            "model": "Nano Banana 2",
            "api_enabled": True,
        }

        with mock.patch.object(
            node,
            "_execute_comfly_only_model",
            new=mock.AsyncMock(return_value={"success": True, "url": "ok"}),
        ) as execute:
            result = asyncio.run(node._execute_banana_model(task, config, {}))

        execute.assert_awaited_once_with(task, config)
        self.assertTrue(result["success"])


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
