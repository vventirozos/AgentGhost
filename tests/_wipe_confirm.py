"""reset_all is two steps since §4KX r8: a PREVIEW with a token, then the
token in a LATER turn. The wipe-mechanics tests exercise the second step;
this wraps a test module's `tool_knowledge_base` so a reset_all call runs the
preview, then confirms it from another request id (the user's next turn)."""
import re

from ghost_agent.utils.logging import request_id_context


def confirming(kb):
    async def call(*a, **kw):
        action = kw.get("action", a[0] if a else None)
        if action != "reset_all" or kw.get("confirm"):
            return await kb(*a, **kw)
        t0 = request_id_context.set("req-the-user-asks")      # a preview the user sees
        try:
            prev = await kb(*a, **kw)
        finally:
            request_id_context.reset(t0)
        m = re.search(r"confirm='([0-9a-f]+)'", str(prev))
        assert m, f"no preview token in {prev!r}"
        tok = request_id_context.set("req-the-user-confirms")
        try:
            return await kb(*a, confirm=m.group(1), **kw)
        finally:
            request_id_context.reset(tok)
    return call
