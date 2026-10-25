#!/usr/bin/env bash
# Starts three CPU RPC servers on loopback and runs test-rpc-comm-mesh against them.
set -euo pipefail

server=$1
client=$2
port_a=$((40000 + $$ % 9000))
ports=("$port_a" $((port_a + 1)) $((port_a + 2)))
test_dir=$(mktemp -d)
pids=()

cleanup() {
    kill "${pids[@]}" 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

wait_for_port() {
    local port=$1
    for _ in {1..600}; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3>&-
            exec 3<&-
            return 0
        fi
        sleep 0.05
    done
    return 1
}

endpoints=()
for port in "${ports[@]}"; do
    "$server" --device CPU --host 127.0.0.1 --port "$port" >"$test_dir/server-$port.log" 2>&1 &
    pids+=($!)
    endpoints+=("127.0.0.1:$port")
done
for port in "${ports[@]}"; do
    wait_for_port "$port"
done

if ! "$client" "${endpoints[@]}"; then
    for port in "${ports[@]}"; do
        echo "--- server $port"
        cat "$test_dir/server-$port.log"
    done
    exit 1
fi
