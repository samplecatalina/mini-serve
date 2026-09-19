#!/usr/bin/env bash
# Prints the Windows host's power state as one line of JSON, for the benchmark
# preflight: whether the laptop is on AC power, and whether the active power
# scheme is the high-performance one. Runs on the WSL2 host (not in the
# container), since both live on the Windows side.
#
# HIGH_PERF_SCHEME: GUID of the scheme that counts as high performance
# (vendor-specific on laptops; defaults to Windows' "High performance").
set -euo pipefail
want="${HIGH_PERF_SCHEME:-8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c}"
if ! command -v powershell.exe >/dev/null; then
    echo '{"ac_power": null, "power_scheme": null, "high_performance": null, "note": "not WSL2"}'
    exit 0
fi
ac=$(powershell.exe -NoProfile -Command '
  Add-Type -AssemblyName System.Windows.Forms
  [System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus' | tr -d '\r')
guid=$(powershell.exe -NoProfile -Command 'powercfg /getactivescheme' | grep -oE '[0-9a-f]{8}-[0-9a-f-]{27}' | head -1)
[[ "$ac" == "Online" ]] && acj=true || acj=false
[[ "$guid" == "$want" ]] && hp=true || hp=false
printf '{"ac_power": %s, "power_scheme": "%s", "high_performance": %s}\n' "$acj" "$guid" "$hp"
