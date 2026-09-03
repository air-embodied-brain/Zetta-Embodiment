#!/usr/bin/env bash
# Copyright (c) 2026 Zetta Contributors
set -uo pipefail
umask 077

# Verify that the Codex stage runtime can actually run before a campaign spends
# a rollout budget reaching Diagnose and failing there.
#
# The campaign stages have no non-Codex path: zetta/evolution/stages.py builds
# `build_planner("codex", ...)` unconditionally, and `--role1-planner` only
# selects the rollout-side Role1 inside run_rollout.py.  Everything below is
# therefore a hard requirement for Diagnose and every stage after it.
#
# Checks run to completion and report together; the script does not stop at the
# first failure, because the usual case is several unrelated gaps at once.

usage() {
  local status=${1:-64}
  cat <<'EOF'
usage: preflight_codex_stage.sh [OPTIONS]

Options:
  --python PATH            Interpreter that will run the campaign (default: python3)
  --provider-env FILE      Shell file exporting the provider credentials.  Must be
                           mode 0600.  Never commit it; it is sourced, not parsed.
  --model ID               Model id the campaign will freeze (default: gpt-5.6-sol)
  --reasoning-effort LEVEL low|medium|high|xhigh (default: high)
  --output-root DIR        Where the live probe writes its report.  Must not exist.
                           (default: a fresh mktemp -d)
  --timeout-s N            Live-probe timeout in seconds (default: 600)
  --skip-probe             Run the static checks only; spend no API tokens.
  -h, --help               Show this message.

Exit codes: 0 all checks passed, 2 at least one check failed.
EOF
  exit "$status"
}

python_bin=${ZETTA_PREFLIGHT_PYTHON:-python3}
provider_file=${PROVIDER_ENV_FILE:-${ZETTA_PROVIDER_ENV_FILE:-}}
model=gpt-5.6-sol
reasoning_effort=high
output_root=
timeout_s=600
skip_probe=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) [[ $# -ge 2 ]] || usage; python_bin=$2; shift 2 ;;
    --provider-env) [[ $# -ge 2 ]] || usage; provider_file=$2; shift 2 ;;
    --model) [[ $# -ge 2 ]] || usage; model=$2; shift 2 ;;
    --reasoning-effort) [[ $# -ge 2 ]] || usage; reasoning_effort=$2; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || usage; output_root=$2; shift 2 ;;
    --timeout-s) [[ $# -ge 2 ]] || usage; timeout_s=$2; shift 2 ;;
    --skip-probe) skip_probe=1; shift ;;
    -h|--help) usage 0 ;;
    *) usage ;;
  esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)

failures=0
warnings=0

pass()  { printf '  [ ok ] %s\n' "$*"; }
fail()  { printf '  [FAIL] %s\n' "$*"; failures=$((failures + 1)); }
warn()  { printf '  [warn] %s\n' "$*"; warnings=$((warnings + 1)); }
info()  { printf '         %s\n' "$*"; }
section() { printf '\n== %s\n' "$*"; }
die()   { printf 'preflight_codex_stage: %s\n' "$*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# 0. Credentials file
# ---------------------------------------------------------------------------
# Sourced before anything else so every later check sees the same environment
# the campaign will see.  Left empty on purpose: fill in one of the four routes
# described in section 3 and point --provider-env at the file.

section "0. provider credential file"
if [[ -n "$provider_file" ]]; then
  if [[ ! -f "$provider_file" ]]; then
    fail "provider env file does not exist: $provider_file"
  else
    provider_file=$(realpath "$provider_file")
    mode=$(stat -c '%a' "$provider_file" 2>/dev/null || stat -f '%Lp' "$provider_file" 2>/dev/null || true)
    if [[ -n "$mode" ]] && (( (8#$mode & 077) != 0 )); then
      fail "provider env file must not be group/world accessible (chmod 600): $provider_file"
    else
      set -a
      # shellcheck disable=SC1090
      source "$provider_file"
      set +a
      pass "sourced $provider_file (mode ${mode:-unknown})"
    fi
  fi
else
  info "no --provider-env given; using the ambient environment"
fi

loopback_no_proxy=127.0.0.1,localhost,::1
export NO_PROXY="${loopback_no_proxy}${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${loopback_no_proxy}${no_proxy:+,${no_proxy}}"

# ---------------------------------------------------------------------------
# 1. Interpreter and Python packages
# ---------------------------------------------------------------------------

section "1. interpreter and packages"

if [[ "$python_bin" == */* ]]; then
  [[ -x "$python_bin" ]] || die "Python is not executable: $python_bin"
  resolved_python=$(cd "$(dirname "$python_bin")" && pwd)/$(basename "$python_bin")
else
  resolved_python=$(command -v "$python_bin") || die "Python is not available: $python_bin"
fi
# Deliberately not `realpath`: resolving a venv's bin/python to the base
# interpreter silently drops the venv, which is exactly the defect that killed
# a full rollout batch through prepare_*_campaign.py.
pass "interpreter $resolved_python"
info "$("$resolved_python" -V 2>&1)"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

# openai_codex is the planner SDK; the rest are what the campaign driver and the
# loopback MCP server import at stage time.
missing_modules=$(
  "$resolved_python" - <<'PY'
import importlib.util

required = [
    "zetta",
    "openai_codex",
    "mcp",
    "httpx",
    "pydantic_ai",
    "fastapi",
    "uvicorn",
    "starlette",
]
print(" ".join(n for n in required if importlib.util.find_spec(n) is None))
PY
)
if [[ -z "$missing_modules" ]]; then
  pass "all planner/driver modules import"
else
  fail "missing Python modules: $missing_modules"
  if [[ "$missing_modules" == *openai_codex* ]]; then
    info "openai-codex>=0.1.0b3 is a project dependency (pyproject.toml)."
    info "scripts/evolution/bootstrap_stage_runtime.sh only *validates* this"
    info "import, it does not install it -- the base interpreter must already"
    info "have it, e.g. via 'pip install -e .' or a targeted:"
    info "  $resolved_python -m pip install 'openai-codex>=0.1.0b3'"
  fi
fi

# ---------------------------------------------------------------------------
# 2. The codex CLI binary
# ---------------------------------------------------------------------------
# The wheel is a thin SDK that spawns a separate ~120MB CLI binary.  Installing
# the package and obtaining the binary are two independent failures, and
# CODEX_BIN (zetta/planner/codex.py) is the supported way to supply a binary
# fetched out of band when the host cannot download it.

section "2. codex CLI binary"

# Resolution order matches what the SDK does: an explicit CODEX_BIN wins,
# otherwise openai-codex-cli-bin ships its own binary inside site-packages, and
# a `codex` on PATH is only a last resort.
codex_bin=${CODEX_BIN:-}
codex_source=
if [[ -n "$codex_bin" ]]; then
  if [[ -x "$codex_bin" ]]; then
    codex_source="CODEX_BIN"
  else
    fail "CODEX_BIN is set but not executable: $codex_bin"
    codex_bin=
  fi
fi

if [[ -z "$codex_bin" ]]; then
  bundled=$(
    "$resolved_python" - <<'CODEXBIN' 2>/dev/null
import importlib.util
import pathlib

# The distribution is openai-codex-cli-bin; the module it installs is codex_cli_bin.
spec = importlib.util.find_spec("codex_cli_bin")
if spec is not None and spec.origin:
    candidate = pathlib.Path(spec.origin).parent / "bin" / "codex"
    if candidate.is_file():
        print(candidate)
CODEXBIN
  )
  if [[ -n "$bundled" && -x "$bundled" ]]; then
    codex_bin=$bundled
    codex_source="openai-codex-cli-bin"
  fi
fi

if [[ -z "$codex_bin" ]]; then
  if codex_bin=$(command -v codex 2>/dev/null); then
    codex_source="PATH"
  else
    codex_bin=
  fi
fi

if [[ -n "$codex_bin" ]]; then
  pass "codex binary via ${codex_source}: $codex_bin"
else
  fail "no codex CLI binary available"
  info "The wheel and the binary are separate packages: openai-codex depends on"
  info "openai-codex-cli-bin, a ~115MB platform wheel.  When the host cannot"
  info "download it, fetch it elsewhere and install from a local wheelhouse:"
  info "  pip install --find-links=/path/to/wheels openai-codex==<version>"
  info "Or point CODEX_BIN at a binary copied in out of band."
fi

if [[ -n "$codex_bin" ]]; then
  if version=$("$codex_bin" --version 2>&1); then
    pass "codex --version -> ${version//$'\n'/ }"
  else
    fail "codex binary is present but did not run: ${version//$'\n'/ }"
    info "A binary built for another libc/arch fails exactly like this."
  fi
fi

# ---------------------------------------------------------------------------
# 3. Credentials
# ---------------------------------------------------------------------------
# Four mutually exclusive routes, resolved in CodexPlanner._build_config with
# precedence D > C > B > A.  Presence only is reported; no value is printed.

section "3. credentials"

route_a=0; route_b=0; route_c=0; route_d=0
{ [[ -f "${HOME:-}/.codex/auth.json" ]] || [[ -f "${HOME:-}/.codex/config.toml" ]]; } && route_a=1
[[ -n "${CODEX_BASE_URL:-}" && -n "${CODEX_API_KEY:-}" ]] && route_b=1
[[ -n "${ZETTA_API_PROVIDERS:-}" ]] && route_c=1
[[ -n "${ZETTA_API_PROVIDER_BROKER_URL:-}" && -n "${ZETTA_API_PROVIDER_BROKER_API_KEY:-}" ]] && route_d=1

if { [[ -n "${ZETTA_API_PROVIDER_BROKER_URL:-}" ]] && [[ -z "${ZETTA_API_PROVIDER_BROKER_API_KEY:-}" ]]; } ||
   { [[ -z "${ZETTA_API_PROVIDER_BROKER_URL:-}" ]] && [[ -n "${ZETTA_API_PROVIDER_BROKER_API_KEY:-}" ]]; }; then
  fail "ZETTA_API_PROVIDER_BROKER_URL and ZETTA_API_PROVIDER_BROKER_API_KEY must be set together"
  info "zetta/planner/provider_proxy.py raises when only one is present."
fi
if [[ -n "${CODEX_BASE_URL:-}" && -z "${CODEX_API_KEY:-}" ]]; then
  warn "CODEX_BASE_URL is set but CODEX_API_KEY is empty"
fi

if (( route_a + route_b + route_c + route_d == 0 )); then
  fail "no provider credentials are configured"
  cat <<'EOF'
         Configure exactly one route (precedence D > C > B > A):
           A  ~/.codex/{config.toml,auth.json}      codex's own configuration
           B  CODEX_BASE_URL + CODEX_API_KEY        direct gateway
           C  ZETTA_API_PROVIDERS (JSON)            local failover pool
           D  ZETTA_API_PROVIDER_BROKER_URL
              + ZETTA_API_PROVIDER_BROKER_API_KEY   external broker
EOF
else
  (( route_a )) && info "A present: codex native config under ~/.codex"
  (( route_b )) && info "B present: CODEX_BASE_URL + CODEX_API_KEY"
  (( route_c )) && info "C present: ZETTA_API_PROVIDERS"
  (( route_d )) && info "D present: broker URL + broker key"
  if   (( route_d )); then effective=D
  elif (( route_c )); then effective=C
  elif (( route_b )); then effective=B
  else                     effective=A
  fi
  pass "route $effective will be used"
  if (( route_a + route_b + route_c + route_d > 1 )); then
    warn "more than one route is configured; the others are ignored"
  fi
fi

# When codex supplies the provider itself, the planner adds no model_provider
# override (only a set CODEX_BASE_URL triggers that), so config.toml decides
# where the request goes and which environment variable holds the key.
if [[ -f "${HOME:-}/.codex/config.toml" ]]; then
  native_report=$(
    "$resolved_python" - "${HOME}/.codex/config.toml" <<'NATIVE' 2>/dev/null
import os
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
try:
    import tomllib

    data = tomllib.loads(text)
    provider = data.get("model_provider")
    block = (data.get("model_providers") or {}).get(provider or "", {})
    env_key = block.get("env_key")
    base_url = block.get("base_url")
    wire_api = block.get("wire_api")
    query = block.get("query_params") or {}
except Exception:
    provider = (re.search(r'^model_provider\s*=\s*"([^"]+)"', text, re.M) or [None, None])[1]
    env_key = (re.search(r'^env_key\s*=\s*"([^"]+)"', text, re.M) or [None, None])[1]
    base_url = (re.search(r'^base_url\s*=\s*"([^"]+)"', text, re.M) or [None, None])[1]
    wire_api = (re.search(r'^wire_api\s*=\s*"([^"]+)"', text, re.M) or [None, None])[1]
    query = {}

if not provider:
    print("NONE")
else:
    have = "set" if (env_key and os.environ.get(env_key)) else "UNSET"
    print(
        f"PROVIDER\t{provider}\t{base_url}\t{wire_api}\t{env_key}\t{have}"
        f"\t{'api-version' in query}"
    )
NATIVE
  )
  if [[ "$native_report" == PROVIDER* ]]; then
    IFS=$'\t' read -r _ np nurl nwire nkey nhave nver <<<"$native_report"
    pass "codex config.toml provider '$np' -> $nurl (wire_api=$nwire)"
    if [[ "$nhave" == set ]]; then
      pass "its env_key $nkey is set in this environment"
    else
      fail "its env_key $nkey is NOT set -- codex will have no credential"
      info "Export it, or put it in the file passed to --provider-env."
    fi
    if [[ "$nwire" != responses ]]; then
      warn "wire_api is '$nwire'; the stage planner expects a Responses endpoint"
    fi
    if [[ "$np" == azure* || "$nurl" == *azure* ]] && [[ "$nver" != True ]]; then
      warn "an Azure provider usually needs query_params = { \"api-version\" = ... }"
    fi
    if [[ -n "${CODEX_BASE_URL:-}" ]]; then
      warn "CODEX_BASE_URL is set, so this config.toml provider is overridden"
      info "The override appends /v1 and carries no query_params; on Azure that breaks."
    fi
  fi
fi

# Validate ZETTA_API_PROVIDERS with the repo's own loader rather than a
# re-implementation, so this agrees with what the planner will accept.
if (( route_c )); then
  pool_report=$(
    "$resolved_python" - <<'PY'
import sys

try:
    from zetta.planner.provider_pool import load_provider_pool_config
except Exception as exc:  # pragma: no cover - import guard
    print(f"ERR could not import the provider pool loader ({exc}); fix section 1 first")
    sys.exit(0)

try:
    config = load_provider_pool_config(default_model="openai-responses:preflight")
except Exception as exc:
    # The message names the offending route but never echoes a key.
    print(f"ERR {type(exc).__name__}: {exc}")
    sys.exit(0)

if config is None:
    print("ERR ZETTA_API_PROVIDERS parsed to no configuration")
    sys.exit(0)

routes = getattr(config, "routes", ())
print(f"OK {len(routes)} route(s): " + ", ".join(str(r.route_id) for r in routes))
PY
  )
  if [[ "$pool_report" == OK* ]]; then
    pass "ZETTA_API_PROVIDERS ${pool_report#OK }"
  else
    fail "ZETTA_API_PROVIDERS is invalid -- ${pool_report#ERR }"
  fi
fi

# ---------------------------------------------------------------------------
# 4. Gateway wire API
# ---------------------------------------------------------------------------
# Once any base_url is in play, _codex_mcp_config_overrides pins
# wire_api = "responses".  A gateway that only implements /v1/chat/completions
# answers 404 on /responses and the stage dies mid-campaign.  Probed
# unauthenticated on purpose: no key leaves this script.

section "4. gateway wire API (responses)"

gateway_url=${ZETTA_API_PROVIDER_BROKER_URL:-${CODEX_BASE_URL:-}}
if [[ -z "$gateway_url" ]]; then
  info "no explicit base_url; codex will use its own default endpoint"
elif ! command -v curl >/dev/null 2>&1; then
  warn "curl is unavailable; cannot probe ${gateway_url%/}/responses"
else
  probe_url="${gateway_url%/}/responses"
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 15 -X POST \
           -H 'content-type: application/json' -d '{}' "$probe_url" 2>/dev/null) || code=
  [[ "$code" =~ ^[0-9]{3}$ ]] || code=000
  case "$code" in
    000) warn "could not reach $probe_url (network, proxy, or TLS)" ;;
    404) fail "$probe_url returned 404 -- gateway does not implement the Responses API"
         info "wire_api is pinned to \"responses\"; a chat-completions-only"
         info "gateway cannot serve the stage planner." ;;
    401|403) pass "$probe_url returned $code (route exists, auth required)" ;;
    400|422) pass "$probe_url returned $code (route exists, rejected the empty body)" ;;
    *)   warn "$probe_url returned $code; interpret manually" ;;
  esac
fi

# ---------------------------------------------------------------------------
# 5. Loopback MCP transport
# ---------------------------------------------------------------------------
# Codex reaches back into this process over an HTTP MCP server bound to
# 127.0.0.1 on a free port.  A proxy that swallows loopback, or a driver split
# across the container boundary, breaks that leg while the outbound leg looks
# healthy.

section "5. loopback MCP transport"

loopback_report=$(
  "$resolved_python" - <<'PY'
import http.server
import socket
import threading
import urllib.request

class Quiet(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"preflight"
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return

try:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
except OSError as exc:
    print(f"ERR cannot bind a loopback port: {exc}")
    raise SystemExit(0)

server = http.server.HTTPServer(("127.0.0.1", port), Quiet)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    # getproxies() honours http_proxy/no_proxy exactly as the real client does.
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
        payload = response.read()
    print("OK" if payload == b"preflight" else "ERR unexpected loopback payload")
except Exception as exc:
    print(f"ERR loopback round trip failed: {type(exc).__name__}: {exc}")
finally:
    server.shutdown()
PY
)
if [[ "$loopback_report" == OK ]]; then
  pass "bound 127.0.0.1 on a free port and round-tripped a request"
else
  fail "${loopback_report#ERR }"
  info "Check http_proxy/https_proxy: NO_PROXY must exempt 127.0.0.1."
  info "The campaign driver and codex must share one network namespace --"
  info "a driver outside the container cannot serve the in-container MCP port."
fi

# ---------------------------------------------------------------------------
# 6. Sandbox and working directory
# ---------------------------------------------------------------------------

section "6. sandbox and cwd"

sandbox=${ZETTA_CODEX_SANDBOX:-full-access}
sandbox_lc=$(printf '%s' "$sandbox" | tr '[:upper:]' '[:lower:]')
case "$sandbox_lc" in
  read-only|readonly|workspace-write|full-access|full)
    pass "ZETTA_CODEX_SANDBOX=$sandbox" ;;
  *)
    fail "ZETTA_CODEX_SANDBOX must be read-only, workspace-write, or full-access; got '$sandbox'" ;;
esac
info "codex runs with cwd=$repo_root"
if [[ "$sandbox_lc" == "full-access" || "$sandbox_lc" == "full" ]]; then
  info "full-access is the historical default and permits writes under that cwd"
fi

case "$reasoning_effort" in
  low|medium|high|xhigh) pass "reasoning effort $reasoning_effort" ;;
  *) fail "--reasoning-effort must be low|medium|high|xhigh; got '$reasoning_effort'" ;;
esac

# ---------------------------------------------------------------------------
# 7. Live single-turn probe
# ---------------------------------------------------------------------------
# One tool-free nonce turn.  Cheaper than discovering the same faults after a
# campaign has spent its rollout budget getting to Diagnose.

section "7. live probe"

if (( skip_probe )); then
  info "skipped (--skip-probe); no API tokens spent"
elif (( failures > 0 )); then
  info "skipped: fix the failures above first so the probe is meaningful"
else
  if [[ -z "$output_root" ]]; then
    output_root=$(mktemp -d)/probe
  fi
  info "report -> $output_root/report.json"
  if "$resolved_python" "$script_dir/probe_codex_stage_runtime.py" \
      --output-root "$output_root" \
      --model "$model" \
      --reasoning-effort "$reasoning_effort" \
      --timeout-s "$timeout_s"; then
    pass "probe passed for model $model"
  else
    fail "probe failed; see $output_root/failure_summary.json"
    info "It checks planner_error_absent, nonce_returned,"
    info "persistent_thread_id_present, raw_stream_parse_complete and"
    info "terminal_event_present -- the five things that break here."
  fi
fi

# ---------------------------------------------------------------------------

printf '\n== summary\n'
if (( failures == 0 )); then
  printf '  %d failure(s), %d warning(s) -- the Codex stage runtime is ready.\n' "$failures" "$warnings"
  exit 0
fi
printf '  %d failure(s), %d warning(s) -- Diagnose and every later stage will not run.\n' "$failures" "$warnings"
exit 2
