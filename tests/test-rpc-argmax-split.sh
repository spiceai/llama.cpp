#!/usr/bin/env bash
# Starts two RPC servers on loopback, both on one device (RPC_TEST_DEVICE, default CPU), and runs test-rpc-argmax-split
# against them. RPC_TEST_DEVICE=MTL0 runs them on Metal, which has neither ARGMAX_PART nor ARGMAX_MERGE.
set -euo pipefail

server=$1
client=$2
device=${RPC_TEST_DEVICE:-CPU}
port_a=$((40000 + $$ % 9000))
port_b=$((port_a + 1))
test_dir=$(mktemp -d)

cleanup() {
    kill "${pid_a:-}" "${pid_b:-}" 2>/dev/null || true
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

"$server" --device "$device" --host 127.0.0.1 --port "$port_a" >"$test_dir/server-a.log" 2>&1 &
pid_a=$!
"$server" --device "$device" --host 127.0.0.1 --port "$port_b" >"$test_dir/server-b.log" 2>&1 &
pid_b=$!
wait_for_port "$port_a"
wait_for_port "$port_b"

if ! "$client" "127.0.0.1:$port_a" "127.0.0.1:$port_b" "$device"; then
    for log in "$test_dir"/server-a.log "$test_dir"/server-b.log; do
        echo "--- $(basename "$log")"
        cat "$log"
    done
    exit 1
fi
