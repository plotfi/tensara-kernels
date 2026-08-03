#!/usr/bin/env bash
#
# egpu-fix.sh — mitigate NVIDIA Thunderbolt/USB4 eGPU drop-offs (Xid 79 via GSP
# RPC hang) on Linux. Applies:
#   B) GRUB: pci=realloc=off -> pci=realloc
#   A2) modprobe: NVreg_EnableGpuFirmware=0 (disable GSP-RM)
#   C) systemd oneshot: cap power/clocks to soften the boost transient
#
# It does NOT switch the driver from the open to the proprietary module (step A1)
# — that is distro/install-specific and risky to automate — but it detects the
# open module and warns, because NVreg_EnableGpuFirmware=0 only works on the
# proprietary module.
#
# Usage:
#   sudo ./egpu-fix.sh apply       # apply B + A2 + C   (reboot afterwards)
#   sudo ./egpu-fix.sh status      # show current state, no changes
#   sudo ./egpu-fix.sh rollback    # undo everything this script installed
#
set -euo pipefail

# ---- tunables --------------------------------------------------------------
POWER_LIMIT_W=150        # valid 100–314 for the RTX 4070 Ti (default 285)
GPU_CLOCK_MIN=210        # MHz
GPU_CLOCK_MAX=1500       # MHz

# Dedicated file so we never clobber an existing /etc/modprobe.d/nvidia-egpu.conf
# (modprobe merges options from all *.conf in the dir).
MODPROBE_CONF=/etc/modprobe.d/nvidia-gsp-disable.conf
SERVICE_UNIT=/etc/systemd/system/nvidia-egpu-powercap.service
GRUB_FILE=/etc/default/grub
SMI=/usr/bin/nvidia-smi

log()  { printf '\033[1;34m[egpu-fix]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

require_root() {
    if [[ ${EUID} -ne 0 ]]; then
        err "must run as root:  sudo $0 ${1:-apply}"; exit 1
    fi
}

module_kind() {   # prints "open", "proprietary", or "unknown"
    if [[ -r /proc/driver/nvidia/version ]]; then
        if grep -qi "Open Kernel Module" /proc/driver/nvidia/version; then echo open
        else echo proprietary; fi
    else echo unknown; fi
}

# ---- apply -----------------------------------------------------------------
apply() {
    require_root apply

    # B) GRUB: pci=realloc=off -> pci=realloc
    if [[ -f "$GRUB_FILE" ]]; then
        cp -n "$GRUB_FILE" "${GRUB_FILE}.egpu-fix.bak" || true
        if grep -q 'pci=realloc=off' "$GRUB_FILE"; then
            sed -i 's/pci=realloc=off/pci=realloc/g' "$GRUB_FILE"
            log "GRUB: pci=realloc=off -> pci=realloc"
        elif grep -q 'pci=realloc\b' "$GRUB_FILE"; then
            log "GRUB: pci=realloc already set"
        else
            sed -i 's/\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"/\1 pci=realloc"/' "$GRUB_FILE"
            log "GRUB: appended pci=realloc"
        fi
        if command -v update-grub >/dev/null 2>&1; then update-grub
        else grub2-mkconfig -o /boot/grub2/grub.cfg 2>/dev/null || warn "regenerate GRUB config manually"; fi
    else
        warn "$GRUB_FILE not found — skipping GRUB step"
    fi

    # A2) modprobe: disable GSP-RM
    cat > "$MODPROBE_CONF" <<'CONF'
# Managed by egpu-fix.sh — disable GSP-RM to avoid the GSP RPC hang that drops
# this Thunderbolt eGPU. Only effective on the PROPRIETARY nvidia module.
# In its own file so existing /etc/modprobe.d/nvidia*.conf options are preserved.
options nvidia NVreg_EnableGpuFirmware=0
CONF
    log "wrote $MODPROBE_CONF"
    for f in /etc/modprobe.d/nvidia*.conf; do
        [[ "$f" == "$MODPROBE_CONF" || ! -e "$f" ]] && continue
        log "preserved existing $f ($(grep -c '^options' "$f" 2>/dev/null || echo 0) option lines)"
    done
    if command -v update-initramfs >/dev/null 2>&1; then update-initramfs -u
    elif command -v dracut >/dev/null 2>&1; then dracut -f
    else warn "rebuild your initramfs manually"; fi

    if [[ "$(module_kind)" == open ]]; then
        warn "the OPEN nvidia module is loaded — it REQUIRES GSP and ignores"
        warn "NVreg_EnableGpuFirmware=0. Switch to the proprietary module (step A1)"
        warn "for the GSP-disable to take effect."
    fi

    # C) systemd power/clock cap
    cat > "$SERVICE_UNIT" <<UNIT
# Managed by egpu-fix.sh
[Unit]
Description=Cap NVIDIA eGPU power/clocks (reduce Thunderbolt drop-offs)
After=multi-user.target
Wants=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=-${SMI} -pm 1
ExecStart=-${SMI} -pl ${POWER_LIMIT_W}
ExecStart=-${SMI} -lgc ${GPU_CLOCK_MIN},${GPU_CLOCK_MAX}

[Install]
WantedBy=multi-user.target
UNIT
    log "wrote $SERVICE_UNIT"
    systemctl daemon-reload
    systemctl enable --now nvidia-egpu-powercap.service || warn "service enable failed (eGPU present?)"

    log "done. REBOOT to apply the GRUB + GSP changes."
    log "after reboot verify:  nvidia-smi -q | grep -i 'GSP Firmware'   (want N/A)"
}

# ---- status ----------------------------------------------------------------
status() {
    echo "== module ==";        module_kind
    echo "== GRUB cmdline ==";  grep -m1 GRUB_CMDLINE_LINUX_DEFAULT "$GRUB_FILE" 2>/dev/null || echo "(no $GRUB_FILE)"
    echo "== modprobe (all nvidia*.conf) =="; grep -H '^options' /etc/modprobe.d/nvidia*.conf 2>/dev/null || echo "(none)"
    echo "== service ==";       systemctl is-enabled nvidia-egpu-powercap.service 2>/dev/null || echo "(not installed)"
    if [[ -x "$SMI" ]]; then
        echo "== live GPU =="
        "$SMI" -q -d POWER,CLOCK 2>/dev/null | grep -iE 'GSP|Power Limit|Max Clocks|SM ' | head || true
        "$SMI" -q 2>/dev/null | grep -i 'GSP Firmware' || true
    fi
}

# ---- rollback --------------------------------------------------------------
rollback() {
    require_root rollback
    systemctl disable --now nvidia-egpu-powercap.service 2>/dev/null || true
    rm -f "$SERVICE_UNIT"; systemctl daemon-reload || true
    [[ -x "$SMI" ]] && { "$SMI" -rgc 2>/dev/null || true; "$SMI" -pl 285 2>/dev/null || true; }
    rm -f "$MODPROBE_CONF"
    command -v update-initramfs >/dev/null 2>&1 && update-initramfs -u || true
    if [[ -f "${GRUB_FILE}.egpu-fix.bak" ]]; then
        mv "${GRUB_FILE}.egpu-fix.bak" "$GRUB_FILE"
        command -v update-grub >/dev/null 2>&1 && update-grub || true
        log "restored $GRUB_FILE from backup"
    fi
    log "rolled back. reboot to fully restore GRUB/GSP state."
}

case "${1:-apply}" in
    apply)    apply ;;
    status)   status ;;
    rollback) rollback ;;
    *) err "usage: sudo $0 {apply|status|rollback}"; exit 1 ;;
esac
