from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import config


@dataclass(frozen=True)
class EmotionExpression:
    """
    감정 식별자. (추후 Enum으로 바꿔도 됨)
    """

    key: str


def get_emotion_hotkey_map() -> Dict[str, str]:
    """
    config에서 감정→핫키 매핑을 가져옵니다.
    - 값이 빈 문자열/None이면 미지정으로 간주합니다.
    """
    raw = getattr(config, "VTS_EMOTION_HOTKEY_MAP", {}) or {}
    # normalize to {str: str}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if k is None:
            continue
        kk = str(k).strip()
        if not kk:
            continue
        vv = "" if v is None else str(v).strip()
        out[kk] = vv
    return out


def resolve_hotkey_for_emotion(emotion: str) -> Optional[str]:
    """
    emotion 문자열을 핫키 ID/이름으로 해석.
    """
    em = (emotion or "").strip()
    if not em:
        return None
    hotkey = get_emotion_hotkey_map().get(em, "")
    hotkey = (hotkey or "").strip()
    return hotkey or None


async def trigger_expression_for_emotion(vts_client: Any, emotion: str) -> bool:
    """
    감정에 해당하는 표정 핫키를 트리거합니다.

    반환:
    - True: 트리거 시도함(매핑 존재)
    - False: 매핑 없음(아무것도 하지 않음)
    """
    hotkey = resolve_hotkey_for_emotion(emotion)
    if not hotkey:
        return False

    # vts_client는 `vts_client.py` 또는 `vts_pyvts.py` 어댑터 모두 호환
    await vts_client.trigger_hotkey(hotkey)
    return True


