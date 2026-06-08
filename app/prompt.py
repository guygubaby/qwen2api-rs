from typing import Any


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = item.get("type")
                if kind in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text") or ""))
                elif kind in {"image", "input_image", "image_url"}:
                    parts.append("[Image attachment]")
                elif kind in {"file", "input_file"}:
                    name = item.get("filename") or item.get("name") or "file"
                    parts.append(f"[File attachment: {name}]")
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if content.get("type") in {"text", "input_text", "output_text"}:
            return str(content.get("text") or "")
        if "content" in content:
            return content_to_text(content["content"])
    return ""


def build_prompt(body: dict[str, Any]) -> str:
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    parts: list[str] = []

    system = body.get("system")
    if system is not None:
        text = content_to_text(system).strip()
        if text:
            parts.append(f"<system>\n{text}\n</system>")

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        if role == "system":
            text = content_to_text(message.get("content")).strip()
            if text:
                parts.append(f"<system>\n{text}\n</system>")
            continue
        if role == "assistant":
            label = "Assistant"
        else:
            label = "Human"
        text = content_to_text(message.get("content")).strip()
        if text:
            parts.append(f"{label}: {text}")

    parts.append("Assistant:")
    return "\n\n".join(parts)
