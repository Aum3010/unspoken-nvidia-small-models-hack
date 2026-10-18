#!/usr/bin/env bash
# Laptop -> Brev: ship the committed backend (services, shared, scripts, requirements) to ~/unspoken.
# Instance-only state (third_party/, data/, .run/) is left in place. Usage: sync_app.sh <instance>
set -euo pipefail
INST=${1:?usage: sync_app.sh <instance>}
cd "$(dirname "$0")/../.."
tmp=$(mktemp -d)
git archive --format=tar.gz -o "$tmp/unspoken.tgz" HEAD services shared scripts evaluations requirements-cuda.txt requirements.txt pytest.ini
brev copy "$tmp/unspoken.tgz" "$INST:/home/shadeform/unspoken.tgz"
brev exec "$INST" 'mkdir -p ~/unspoken && tar -xzf ~/unspoken.tgz -C ~/unspoken && rm ~/unspoken.tgz && echo synced $(ls ~/unspoken | tr "\n" " ")'
rm -rf "$tmp"
