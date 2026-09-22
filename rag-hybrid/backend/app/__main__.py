"""`python -m backend.app` starts the API using host/port from the environment."""

import uvicorn

from backend.app.core.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("backend.app.main:create_app", factory=True, host=s.api_host, port=s.api_port, reload=False)
