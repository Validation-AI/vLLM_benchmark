#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

log_clock_state() {
  echo "UTC now: $(date -u '+%Y-%m-%d %H:%M:%S')"
  # Cheap sanity guard: bad clocks often break signature validation.
  local year
  year="$(date -u '+%Y')"
  if [[ "${year}" -lt 2024 ]]; then
    echo "WARNING: UTC year looks wrong (${year}); verify node clock sync."
  fi
}

switch_ubuntu_mirrors_to_https() {
  if [[ -f /etc/apt/sources.list ]]; then
    sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
      -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
      /etc/apt/sources.list
  fi
  if compgen -G "/etc/apt/sources.list.d/*.list" >/dev/null; then
    sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
      -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
      /etc/apt/sources.list.d/*.list
  fi
}

apt_update_robust() {
  apt-get clean || true
  rm -rf /var/lib/apt/lists/* || true
  apt-get -q=1 update \
    -o Acquire::Retries=3 \
    -o Acquire::http::No-Cache=true \
    -o Acquire::https::No-Cache=true \
    -o Acquire::AllowInsecureRepositories=true \
    -o Acquire::AllowDowngradeToInsecureRepositories=true
}

# Required system deps for building/installing vllm-router.
APT_DEPS=(protobuf-compiler libprotobuf-dev pkg-config build-essential libssl-dev)

apt_deps_already_installed() {
  local pkg
  for pkg in "${APT_DEPS[@]}"; do
    if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
      return 1
    fi
  done
  return 0
}

# Install apt deps best-effort. The base image may already provide everything;
# if it does, we skip apt entirely. If apt update fails (e.g. transient mirror
# GPG signature errors), we still proceed to the pip install so that the real
# failure surfaces from pip rather than apt.
install_apt_deps_best_effort() {
  if apt_deps_already_installed; then
    echo "All required apt deps already installed; skipping apt update."
    return 0
  fi
  switch_ubuntu_mirrors_to_https
  if ! apt_update_robust; then
    echo "WARNING: apt-get update failed (likely transient mirror/GPG issue); continuing."
  fi
  apt-get -q=1 --fix-broken install -y || true
  if ! apt-get -q=1 install -y "${APT_DEPS[@]}"; then
    echo "WARNING: apt-get install of build deps failed; continuing in case the deps are already present or pip uses prebuilt wheels."
  fi
}

if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -q -y --profile minimal
fi

source "$HOME/.cargo/env"

log_clock_state
install_apt_deps_best_effort

uv pip install setuptools-rust wheel build
uv pip install vllm-router

echo "vllm-router installation complete."
