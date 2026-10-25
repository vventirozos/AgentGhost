"""Render the 27B chat template with jinja2 the way llama.cpp's minja does (checked byte-for-byte
against the server's /apply-template by fidelity())."""
import json, os, urllib.request
import jinja2

S = os.path.dirname(os.path.abspath(__file__))
SERVER = "http://127.0.0.1:8088"


def _env():
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=False,
                             extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda v, indent=None: json.dumps(v, ensure_ascii=False, indent=indent)
    def _raise(msg): raise jinja2.TemplateError(msg)
    env.globals["raise_exception"] = _raise
    return env


def render(tpl_path, messages, tools=None, **kw):
    t = _env().from_string(open(tpl_path).read())
    return t.render(messages=messages, tools=tools, add_generation_prompt=True,
                    bos_token="", eos_token="<|im_end|>", **kw)


def server_render(messages, tools=None, **kw):
    body = {"messages": messages}
    if tools:
        body["tools"] = tools
    if kw:
        body["chat_template_kwargs"] = kw
    req = urllib.request.Request(SERVER + "/apply-template", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))["prompt"]


def head():
    return json.load(open(os.path.join(S, "head.json")))


if __name__ == "__main__":
    h = head()
    convs = {
        "first": [{"role": "system", "content": h["system"]}, {"role": "user", "content": "write fizzbuzz in python"}],
        "tool_loop": [{"role": "system", "content": h["system"]}, {"role": "user", "content": "list files"},
                      {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                       "function": {"name": "file_system", "arguments": {"operation": "list", "path": "."}}}]},
                      {"role": "tool", "tool_call_id": "c1", "name": "file_system", "content": "a.py\nb.py"}],
    }
    bad = 0
    for name, msgs in convs.items():
        for lv in ("low", "medium", "xhigh"):
            a = render(os.path.join(S, "..", "tpl.jinja"), msgs, h["tools"], reasoning_effort=lv)
            b = server_render(msgs, h["tools"], reasoning_effort=lv)
            ok = a == b
            bad += not ok
            if not ok:
                i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                print(f"MISMATCH {name}/{lv} at {i}: ours={a[i-40:i+60]!r}\n   server={b[i-40:i+60]!r}")
            else:
                print(f"match {name}/{lv} ({len(a)} chars)")
    print("bad", bad)
