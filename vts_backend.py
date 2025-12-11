import config

from vts_client import VTubeStudioClient, VTSPluginInfo


def build_vts_client():
    """
    VTS client factory.

    현재는 가벼운 직접 WebSocket 구현을 기본으로 사용합니다.
    추후 VTS 기능(핫키/이벤트/아이템 등)이 커지면 pyvts 백엔드로 전환할 수 있도록
    선택 스캐폴딩만 제공해 둡니다.
    """
    backend = getattr(config, "VTS_BACKEND", "ws")

    if backend == "pyvts":
        # Lazy import to keep dependency optional.
        from vts_pyvts import PyVTSClientAdapter  # noqa: F401

        return PyVTSClientAdapter(
            plugin_name=config.VTS_PLUGIN_NAME,
            plugin_developer=config.VTS_PLUGIN_DEVELOPER,
            ws_url=config.VTS_WS_URL,
            token_path=getattr(config, "VTS_AUTH_TOKEN_PATH", "./vts_token.txt"),
        )

    # default: "ws"
    return VTubeStudioClient(
        ws_url=config.VTS_WS_URL,
        plugin=VTSPluginInfo(
            plugin_name=config.VTS_PLUGIN_NAME,
            plugin_developer=config.VTS_PLUGIN_DEVELOPER,
            auth_token="",
        ),
    )


