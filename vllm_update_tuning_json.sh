#!/bin/bash
set -ex

Avg_Next_token_latency=${1:-0}
Avg_First_token_latency=${2:-0}
throughput=${3:-0}
Batch_size=${4:-0}
hardware=${5:-0}
backend=${6:-0}
model=${7:-0}

dtype=${8:-0}
TP=${9:-0}
length_config=${10:-0}
INPUT_FILE=${11:-0}
tpot_sla=${12:-}
ttft_sla=${13:-}
pipeline_parallel=${14:-1}
dp_mode=${15:-none}
dp_size=${16:-1}
launch_fingerprint=${17:-}
updated_at=${18:-}
tp_socket=${19:-}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"
update_file="bak_${INPUT_FILE}"
if [[ -f "$INPUT_FILE" ]]; then
    cp "$INPUT_FILE" "$update_file"
else
    echo '{}' > "$update_file"
fi

jq --arg backend "$backend" \
   --arg hardware "$hardware" \
   --arg model "$model" \
   --arg dtype "$dtype" \
   --arg TP "$TP" \
   --arg length_config "$length_config" \
   --arg tpot_sla "$tpot_sla" \
   --arg ttft_sla "$ttft_sla" \
   --arg pipeline_parallel "$pipeline_parallel" \
   --arg dp_mode "$dp_mode" \
   --arg dp_size "$dp_size" \
   --arg launch_fingerprint "$launch_fingerprint" \
   --arg updated_at "$updated_at" \
   --arg tp_socket "$tp_socket" \
   --argjson Batch_size "$Batch_size" \
   --argjson throughput "$throughput" \
   --argjson Avg_First_token_latency "$Avg_First_token_latency" \
   --argjson Avg_Next_token_latency "$Avg_Next_token_latency" \
   '
   .[$backend] |= (. // {}) |
   .[$backend][$hardware] |= (. // {}) |
   .[$backend][$hardware][$model] |= (. // {}) |
   .[$backend][$hardware][$model][$dtype] |= (. // {}) |
   .[$backend][$hardware][$model][$dtype][$TP] |= (. // {}) |
   .[$backend][$hardware][$model][$dtype][$TP][$length_config] =
     {
       "batch_size": $Batch_size,
       "baseline_thp": $throughput,
       "baseline_first_token": $Avg_First_token_latency,
       "baseline_next_token": $Avg_Next_token_latency,
       "tpot_sla": $tpot_sla,
       "ttft_sla": $ttft_sla,
       "pipeline_parallel": $pipeline_parallel,
       "dp_mode": $dp_mode,
       "dp_size": $dp_size,
       "backend": $backend,
       "launch_fingerprint": $launch_fingerprint,
       "updated_at": $updated_at,
       "tp_socket": $tp_socket
     }
   ' "$update_file" > output.json

mv output.json "$INPUT_FILE"

echo "Json updated"
