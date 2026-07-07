import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("heart_disease_api")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Tags every request with an ID and logs method/path/status/latency.

    A request ID makes it possible to correlate a single API call across
    logs -- genuinely useful once this runs anywhere with concurrent traffic,
    and something the original app had no way to do at all.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        request.state.request_id = request_id

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s (%sms) [request_id=%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response
