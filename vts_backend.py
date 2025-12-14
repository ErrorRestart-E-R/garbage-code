import config

from vts_client import VTubeStudioClient, VTSPluginInfo


def build_vts_client():
    #VTS client factory.
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


