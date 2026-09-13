"""Serial HTTP adapter; never starts a test in the simulator GUI."""

import json
import time
import uuid

import requests


class ProtocolError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, robot_id, base_url='http://127.0.0.1:2026', timeout=3.0, retries=2, session=None):
        self.robot_id, self.base_url = robot_id, base_url.rstrip('/')
        self.timeout, self.retries = timeout, retries
        self.session = session or requests.Session()
        self.deadline = float('inf')
        self.log = []

    def post(self, kind, action=None):
        request_id = uuid.uuid4().hex
        payload = {'arena_id': 'default', 'robot_id': self.robot_id, 'request_id': request_id}
        if action is not None:
            payload.update(action.payload())
        data = json.dumps(payload, allow_nan=False).encode('utf-8')
        for attempt in range(self.retries + 1):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Real deadline reached before request')
            try:
                response = self.session.post(
                    self.base_url + '/' + kind,
                    data=data,
                    headers={'Content-Type': 'application/json'},
                    timeout=min(self.timeout, remaining),
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                self.log.append({'path': kind, 'request': payload, 'attempt': attempt, 'error': str(exc)})
                if attempt == self.retries:
                    raise ProtocolError(
                        'Uncertain action outcome; stop, do not send a replacement action'
                    ) from exc
                continue  # Same bytes and same request_id on every retry.
            try:
                body = response.json()
            except ValueError as exc:
                raise ProtocolError('Malformed response; action outcome is uncertain') from exc
            self.log.append(
                {'path': kind, 'request': payload, 'http_status': response.status_code, 'response': body}
            )
            if response.status_code != 200 or body.get('accepted') is not True:
                raise ProtocolError(f'HTTP {response.status_code}, accepted={body.get("accepted")}')
            return request_id, body
        raise AssertionError('Unreachable')

    def enter(self):
        started = time.monotonic()
        request_id, body = self.post('enter')
        # Starting from pre-request time is deliberately conservative on retries.
        self.deadline = started + float(body['remaining_real_duration_s'])
        return request_id, body

    def execute(self, action):
        return self.post(action.kind, action)

    def exit(self):
        return self.post('exit')
