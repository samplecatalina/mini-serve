#!/usr/bin/env bash
# Prints the Windows host's power state as one line of JSON, for the benchmark
# preflight: whether the laptop is on AC power, and whether the active power
# scheme is the high-performance one. Runs on the WSL2 host (not in the
# container), since both live on the Windows side.
#
# Also records the host CPU's state just before the run, which the preflight
# does not judge but the sidecar keeps: throughput has drifted by 3-6% across
# sessions on unchanged code with the GPU clock at its ceiling both times, and
# the CPU side had not been recorded at all. "% Processor Performance" is the
# effective clock as a percentage of the base clock (above 100 under boost);
# WMI's CurrentClockSpeed only reports the base clock on this machine.
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
cpu=$(powershell.exe -NoProfile -Command '
  $c = (Get-Counter "\Processor Information(_Total)\% Processor Performance","\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 1).CounterSamples
  "{0:F1} {1:F1}" -f $c[0].CookedValue, $c[1].CookedValue' | tr -d '\r')
read -r perf util <<<"$cpu"
[[ "$perf" =~ ^[0-9.]+$ ]] || perf=null
[[ "$util" =~ ^[0-9.]+$ ]] || util=null
[[ "$ac" == "Online" ]] && acj=true || acj=false
[[ "$guid" == "$want" ]] && hp=true || hp=false
printf '{"ac_power": %s, "power_scheme": "%s", "high_performance": %s, "cpu_performance_pct": %s, "cpu_util_pct": %s}\n' \
    "$acj" "$guid" "$hp" "$perf" "$util"
