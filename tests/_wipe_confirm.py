"""§4UK (operator 11-12: no confirmations): reset_all wipes in the call that
asks for it. The wipe-mechanics tests run it from a user request id (a
request with no id is not the user, so the wall would refuse it)."""
from ghost_agent.utils.logging import request_id_context


def confirming(kb):
    async def call(*a, **kw):
        action = kw.get("action", a[0] if a else None)
        if action != "reset_all":
            return await kb(*a, **kw)
        t0 = request_id_context.set("req-the-user-asks")
        try:
            return await kb(*a, **kw)
        finally:
            request_id_context.reset(t0)
    return call
