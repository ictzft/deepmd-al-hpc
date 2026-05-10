import json
from pathlib import Path
from typing import Any, Dict, Union


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    保存 JSON 文件。

    Parameters
    ----------
    data:
        需要保存的字典数据。

    path:
        输出文件路径，可以是字符串或 Path 对象。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取 JSON 文件。

    Parameters
    ----------
    path:
        JSON 文件路径。

    Returns
    -------
    data:
        读取后的字典数据。
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
