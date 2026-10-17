#!/usr/bin/env bash
# Install the official Brev CLI Linux binary into ~/.local/bin (no sudo), checksum-verified.
# Windows has no native build: run this inside WSL Ubuntu. macOS: brew install brevdev/homebrew-brev/brev
#   wsl -d Ubuntu -- bash asr_eval/scripts/brev/install_cli_wsl.sh [version]
set -euo pipefail
V=${1:-0.6.337}
BASE="https://github.com/brevdev/brev-cli/releases/download/v${V}"
TMP=$(mktemp -d)
cd "$TMP"
curl -fsSL -o "brev-cli_${V}_linux_amd64.tar.gz" "${BASE}/brev-cli_${V}_linux_amd64.tar.gz"
curl -fsSL -o sums.txt "${BASE}/brev-cli_${V}_checksums.txt"
grep "brev-cli_${V}_linux_amd64.tar.gz" sums.txt | sha256sum -c -
tar xzf "brev-cli_${V}_linux_amd64.tar.gz"
mkdir -p "$HOME/.local/bin"
install -m 755 brev "$HOME/.local/bin/brev"
grep -q 'HOME/.local/bin' "$HOME/.bashrc" || echo 'export PATH=$HOME/.local/bin:$PATH' >> "$HOME/.bashrc"
"$HOME/.local/bin/brev" --version
