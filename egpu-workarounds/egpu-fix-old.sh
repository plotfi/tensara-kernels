#!/usr/bin/env bash
#
# egpu-fix-old.sh — the ORIGINAL Reddit mitigation (SUPERSEDED, kept for the record).
#
# Source: the r/eGPU post "Razer Core X V2 on Kubuntu 26.04 having PCIe drop"
#   (RTX 4070 Ti, Razer Core X V2, Thunderbolt, nvidia 395 driver).
#
# This set encodes ONE hypothesis: "the PCIe/Thunderbolt link (or the GPU) drops
# into a low-power state and doesn't come back, so keep everything awake." It is a
# LINK-POWER-MANAGEMENT fix.
#
# It does NOT work for this box's actual failure, which dmesg shows is a GSP
# firmware RPC hang (_kgspLogRpcSanityCheckFailure ... GSP_RM_CONTROL -> Xid 79),
# a different layer entirely. Nothing here touches GSP. See README.md and use
# egpu-fix.sh instead. This script exists only to reproduce / roll back the old
# state.
#
# Usage:
#   sudo ./egpu-fix-old.sh apply       # apply the old GRUB + modprobe set (reboot after)
#   sudo ./egpu-fix-old.sh status      # show current state, no changes
#   sudo ./egpu-fix-old.sh rollback    # undo what this script wrote
#
set -euo pipefail

# The old GRUB cmdline verbatim from the Reddit post.
OLD_CMDLINE='quiet splash pcie_aspm=off pci=realloc=off pcie_ports=native pcie_port_pm=off thunderbolt.clx=0 thunderbolt.host_reset=0 iommu=pt'

# The old post wrote these into nvidia-egpu.conf. We manage our OWN file so we
# never clobber a hand-edited /etc/modprobe.d/nvidia-egpu.conf.
MODPROBE_CONF=/etc/modprobe.d/nvidia-egpu-old-reddit.conf
GRUB_FILE=/etc/default/grub

log()  { printf '\033[1;34m[egpu-fix-old]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

require_root() {
    if [[ ${EUID} -ne 0 ]]; then
        err "must run as root:  sudo $0 ${1:-apply}"; exit 1
    fi
}

# ---- apply -----------------------------------------------------------------
apply() {
    require_root apply
    warn "This is the SUPERSEDED link-power mitigation. It does NOT disable GSP"
    warn "and will NOT stop the GSP_RM_CONTROL / Xid 79 hang this box actually"
    warn "hits. Use egpu-fix.sh instead. Continuing anyway..."

    # GRUB: replace the whole GRUB_CMDLINE_LINUX_DEFAULT with the old line.
    if [[ -f "$GRUB_FILE" ]]; then
        cp -n "$GRUB_FILE" "${GRUB_FILE}.egpu-fix-old.bak" || true
        if grep -q '^GRUB_CMDLINE_LINUX_DEFAULT=' "$GRUB_FILE"; then
            sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT='${OLD_CMDLINE}'|" "$GRUB_FILE"
        else
            printf "GRUB_CMDLINE_LINUX_DEFAULT='%s'\n" "$OLD_CMDLINE" >> "$GRUB_FILE"
        fi
        log "GRUB cmdline set to the old Reddit line (note: pci=realloc=OFF)"
        if command -v update-grub >/dev/null 2>&1; then update-grub
        else grub2-mkconfig -o /boot/grub2/grub.cfg 2>/dev/null || warn "regenerate GRUB config manually"; fi
    else
        warn "$GRUB_FILE not found — skipping GRUB step"
    fi

    # modprobe: NVIDIA runtime power management off + don't preserve VRAM.
    cat > "$MODPROBE_CONF" <<'CONF'
# Managed by egpu-fix-old.sh — the old Reddit r/eGPU mitigation.
# Disables NVIDIA dynamic (runtime/D3) power management and VRAM preservation.
# NOTE: this does NOT touch GSP and does not fix the GSP RPC hang. Superseded by
# egpu-fix.sh (NVreg_EnableGpuFirmware=0). In its own file to avoid clobbering an
# existing /etc/modprobe.d/nvidia-egpu.conf.
options nvidia NVreg_DynamicPowerManagement=0x00
options nvidia NVreg_PreserveVideoMemoryAllocations=0
CONF
    log "wrote $MODPROBE_CONF"
    if command -v update-initramfs >/dev/null 2>&1; then update-initramfs -u
    elif command -v dracut >/dev/null 2>&1; then dracut -f
    else warn "rebuild your initramfs manually"; fi

    warn "NVreg_PreserveVideoMemoryAllocations=0 here conflicts with the =1 that"
    warn "nvidia-graphics-drivers-kms.conf sets — last-loaded wins. Reconcile if"
    warn "you rely on suspend/resume VRAM preservation."

    log "done. REBOOT to apply. (Reminder: prefer egpu-fix.sh — this won't stop GSP hangs.)"
}

# ---- status ----------------------------------------------------------------
status() {
    echo "== GRUB cmdline ==";  grep -m1 GRUB_CMDLINE_LINUX_DEFAULT "$GRUB_FILE" 2>/dev/null || echo "(no $GRUB_FILE)"
    echo "== this script's modprobe file =="; cat "$MODPROBE_CONF" 2>/dev/null || echo "(not present)"
    echo "== all nvidia modprobe options =="; grep -H '^options' /etc/modprobe.d/nvidia*.conf 2>/dev/null || echo "(none)"
    echo "== GSP firmware (this fix does NOT change it) =="
    nvidia-smi -q 2>/dev/null | grep -i 'GSP Firmware' || echo "(nvidia-smi unavailable)"
}

# ---- rollback --------------------------------------------------------------
rollback() {
    require_root rollback
    rm -f "$MODPROBE_CONF"
    command -v update-initramfs >/dev/null 2>&1 && update-initramfs -u || true
    if [[ -f "${GRUB_FILE}.egpu-fix-old.bak" ]]; then
        mv "${GRUB_FILE}.egpu-fix-old.bak" "$GRUB_FILE"
        command -v update-grub >/dev/null 2>&1 && update-grub || true
        log "restored $GRUB_FILE from backup"
    fi
    log "rolled back. reboot to fully restore GRUB state."
}

case "${1:-status}" in
    apply)    apply ;;
    status)   status ;;
    rollback) rollback ;;
    *) err "usage: sudo $0 {apply|status|rollback}"; exit 1 ;;
esac
