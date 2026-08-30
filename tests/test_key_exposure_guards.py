"""The master key must not be presentable-but-unmatchable, nor served back.

Both defects were live on 2026-08-29 and both had the same shape: a secret
handled correctly in one place and not in its sibling.

1. `~/.zshrc` exports `GHOST_API_KEY=" "`. HTTP strips leading/trailing OWS
   from header values, so a whitespace key is UNMATCHABLE over the wire —
   every request 403s while the operator believes auth is configured — and
   `enforce_api_key_policy` printed NOTHING and bound `0.0.0.0`.
   `interface/server.py:79` has an explicit guard for exactly this;
   `main.py` did not.
2. `_build_resolved_config` redacted the `arg.api_key` leg and copied every
   `GHOST_*` env var verbatim, so the 64-char master key was served by
   `/api/health` and written to `last_config.json` (0644) — while
   `~/Data/AI/.ghost_api_key` is correctly 0600.
"""

import pytest

from ghost_agent.main import _is_secret_env, enforce_api_key_policy


class TestAWhitespaceKeyIsNotAConfiguration:
    @pytest.mark.parametrize("key", [" ", "   ", "\t", "\n", " \t\n "])
    def test_it_refuses_to_start_on_a_public_bind(self, key, capsys):
        """⚠ THIS PIN USED TO ASSERT THE DEFECT.

        Its first version checked stdout for "whitespace-only" and "auth
        explicitly DISABLED" — a MESSAGE test wearing a behaviour test's
        docstring. The code printed "treating it as an explicit --api-key ''
        (auth disabled)" and assigned to a LOCAL; the function returns None
        and the caller passes `args.api_key` unchanged, so auth stayed ON
        with a credential no client can present. Both assertions held
        anyway, because both are about what was printed.

        Plumbing the empty value through would have been WORSE than the bug
        — it really would disable auth on a public bind. So the whitespace
        case is now refused outright, like an absent key, and what is
        asserted is the refusal.
        """
        with pytest.raises(SystemExit) as e:
            enforce_api_key_policy(key, "0.0.0.0")
        assert e.value.code == 2
        assert "whitespace-only" in capsys.readouterr().out

    @pytest.mark.parametrize("key", [" ", "\t\n"])
    def test_it_warns_but_runs_on_loopback(self, key, capsys):
        """The counterweight: unreachable from the network, so refusing
        would respawn-loop a KeepAlive daemon over a local typo."""
        enforce_api_key_policy(key, "127.0.0.1")
        assert "whitespace-only" in capsys.readouterr().out

    def test_the_explicit_opt_out_still_works(self, capsys):
        """`--api-key ''` is a deliberate operator choice on a trusted mesh
        and must keep booting; only WHITESPACE is the misconfiguration."""
        enforce_api_key_policy("", "0.0.0.0")
        assert "auth explicitly DISABLED" in capsys.readouterr().out

    def test_a_real_key_is_untouched(self, capsys):
        enforce_api_key_policy("a-real-64-char-secret", "0.0.0.0")
        out = capsys.readouterr().out
        assert "whitespace-only" not in out
        assert "auth explicitly DISABLED" not in out

    def test_an_absent_key_still_refuses_a_public_bind(self):
        with pytest.raises(SystemExit):
            enforce_api_key_policy(None, "0.0.0.0")


class TestNoSecretEnvVarReachesTheConfigDump:
    @pytest.mark.parametrize("name", [
        "GHOST_API_KEY", "GHOST_SLACK_TOKEN", "GHOST_DB_PASSWORD",
        "GHOST_OAUTH_SECRET", "GHOST_SERVICE_TOKEN", "GHOST_X_CREDENTIAL",
        "GHOST_BASIC_AUTH",
    ])
    def test_secret_shaped_names_are_recognised(self, name):
        assert _is_secret_env(name) is True, (
            f"{name} would be served in cleartext by /api/health and written "
            "to last_config.json (0644)")

    @pytest.mark.parametrize("name", [
        "GHOST_HOME", "GHOST_LLM_RECORD", "GHOST_SKILL_PRUNE",
        "GHOST_MAX_CONTEXT",
    ])
    def test_ordinary_names_are_still_visible(self, name):
        assert _is_secret_env(name) is False, (
            "redacting everything makes the config dump useless")

    def test_the_env_leg_is_redacted_in_the_source(self):
        """The `arg.` leg was redacted and the `env.` leg was not — the same
        secret, two sinks, one guarded."""
        import ast
        import inspect

        import ghost_agent.main as M

        fn = next(n for n in ast.walk(ast.parse(inspect.getsource(M)))
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_build_resolved_config")
        body = ast.unparse(fn)
        assert "_is_secret_env" in body, (
            "every GHOST_* env var is copied verbatim into a dict that "
            "/api/health serves and last_config.json stores")
