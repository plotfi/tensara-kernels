# eGPU drop-off workarounds (NVIDIA over Thunderbolt/USB4)

This directory documents why a **Thunderbolt/USB4-connected RTX 4070 Ti** (Razer
Core X V2) periodically falls off the PCIe bus on Linux, and the mitigations that
do — and don't — fix it.

- **[`egpu-fix.sh`](egpu-fix.sh)** — the working mitigation. **Use this.**
- **[`egpu-fix-old.sh`](egpu-fix-old.sh)** — the original Reddit mitigation, kept
  for the record. **Superseded** — it treats the wrong layer (see below).

---

## The symptom

Mid- or post-session the GPU vanishes:

- `nvidia-smi` → "No devices were found"
- CUDA programs segfault inside `libcuda`
- benchmark harnesses silently emit all-zero output (an unchecked `cudaMemcpy`
  onto a dead device)
- recovery requires a **full OS reboot** — `nvidia-smi -r` cannot bring it back
  once the device is off the bus

Notably it often survives a heavy **compute** run and then drops at **idle** just
afterward — the tell that this is a power/link *transition* failure, not a
compute failure.

## The actual root cause (from dmesg)

```
_kgspLogRpcSanityCheckFailure ... waiting for RPC response from GSP (GSP_RM_CONTROL)
NVRM ... Bad TimeLo value ffffffff          <- all-Fs MMIO = device already off the bus
Xid 79  "GPU has fallen off the bus"
Xid 154 "Node Reboot Required"
uvm ... fatal error 0x60, requiring os reboot
```

This is a **GSP firmware RPC hang**, not a link-sleep drop.

**GSP** (GPU System Processor) is an on-die RISC-V core that runs the NVIDIA
"resman" resource-manager logic that used to live in the host kernel driver. The
host driver talks to GSP over an **RPC channel**. Here a `GSP_RM_CONTROL` RPC is
sent and GSP never answers; MMIO then reads all-Fs and the device is declared off
the bus.

Two facts about *this* box make it happen:

1. **It's an external GPU over a Thunderbolt/USB4 PCIe tunnel**, not a card in a
   slot. dmesg shows `USB4 _OSC` and the GPU behind Intel PCIe root port
   `00:07.0 [8086:7ec4]` as a "PCIe Legacy Endpoint" on a Gen1 x4 link — that
   link is the TB tunnel. Power-state transitions over a TB tunnel are far more
   fragile than over a physical slot, so a GSP RPC that has to complete across it
   during an idle/power transition can time out and take the whole device down.

2. **It was on the NVIDIA _open_ kernel module**, which **mandates GSP**. On the
   open module GSP cannot be disabled and `NVreg_EnableGpuFirmware=0` is silently
   ignored — so the hang was unavoidable until switching to the **proprietary**
   module.

---

## The working mitigation — `egpu-fix.sh`

Layered fix; the first item is the one that matches the dmesg signature, the rest
harden the Thunderbolt link around it.

| # | Change | What it addresses | Why it works |
|---|--------|-------------------|--------------|
| A1 | Switch **open → proprietary** nvidia module | prerequisite | Only the proprietary module lets GSP be disabled at all. (Done by the 610 driver install; `egpu-fix.sh` detects + warns, doesn't automate the swap.) |
| A2 | `NVreg_EnableGpuFirmware=0` (in `nvidia-gsp-disable.conf`) | **the GSP RPC hang** | No GSP ⇒ no GSP-RM RPC channel ⇒ the `GSP_RM_CONTROL` RPC that hung no longer exists; resman runs on the host CPU. Verify: `nvidia-smi -q \| grep -i 'GSP Firmware'` → **N/A**. |
| B | GRUB `pci=realloc=off` → **`pci=realloc`** | TB BAR/window sizing | Lets the kernel reassign PCI BAR/bridge windows for the hot-plugged TB device so its resource windows fit behind the TB bridge — fewer enumeration/link-setup failures. |
| C | Power cap **150 W** + clock lock **210–1500 MHz** (systemd oneshot) | boost transient | Sudden power/voltage swings on a bandwidth-limited TB link are a known drop trigger; capping the boost keeps those transitions gentle. Belt-and-suspenders. |

Run:

```bash
sudo ./egpu-fix.sh apply     # writes A2 + B + C (detects/warns on the open module)
sudo reboot                  # required for GRUB + GSP to take effect
# verify:
nvidia-smi -q | grep -i 'GSP Firmware'    # want: N/A
```

`./egpu-fix.sh status` prints current state (read-only); `sudo ./egpu-fix.sh
rollback` undoes everything it installed.

**Confirmed in effect on this box:** GSP Firmware `N/A`, proprietary module,
`pci=realloc`, power limit `150 W`, powercap service enabled.

### Orthogonal but real: the cable
Use the **bundled** TB4/TB5 cable, not a random one — a marginal cable causes
TB4/TB5 negotiation issues that look like drops. This helps link negotiation; it
does **not** address GSP, so it's a complement to A2, not a substitute.

---

## The old mitigation — `egpu-fix-old.sh` (why it does NOT work here)

From the r/eGPU post *"Razer Core X V2 on Kubuntu 26.04 having PCIe drop"* (same
4070 Ti + Razer Core X V2, on the **395** driver):

```
# /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT='quiet splash pcie_aspm=off pci=realloc=off \
  pcie_ports=native pcie_port_pm=off thunderbolt.clx=0 thunderbolt.host_reset=0 iommu=pt'

# /etc/modprobe.d/nvidia-egpu.conf
options nvidia NVreg_DynamicPowerManagement=0x00
options nvidia NVreg_PreserveVideoMemoryAllocations=0
```

Every knob here is **PCIe / Thunderbolt link power management** or **NVIDIA
runtime power management**:

| Old setting | Does | Layer |
|---|---|---|
| `pcie_aspm=off` | disable PCIe Active State Power Mgmt (link L-states) | PCIe link |
| `pcie_port_pm=off` | disable PCIe port power management | PCIe link |
| `pcie_ports=native` | OS (not firmware) owns the ports | PCIe link |
| `thunderbolt.clx=0` | disable TB CL power states | TB link |
| `thunderbolt.host_reset=0` | disable the TB host-controller reset path | TB link |
| `NVreg_DynamicPowerManagement=0x00` | disable GPU runtime/D3 power-down | NVIDIA PM |
| `NVreg_PreserveVideoMemoryAllocations=0` | don't save/restore VRAM on suspend | NVIDIA PM |

The whole set encodes **one hypothesis**: *"the link or the GPU sleeps and won't
wake — so keep everything awake."* That's a coherent fix for a **link-sleep**
drop, and it held for the poster on the 395 driver.

**It cannot fix this box because:**

1. **It never disables GSP.** The only knob that does is `NVreg_EnableGpuFirmware=0`,
   which is absent from the old config. Your fault is a GSP RPC hang — a
   firmware-communication failure *above* the link layer. You can pin every link
   state wide awake and the GSP channel can still hang.
2. **You were on the open module**, which ignores `NVreg_EnableGpuFirmware=0`
   even if you add it — so the GSP hang was unavoidable until the proprietary
   swap. "Days of stability" on the poster's setup didn't transfer.
3. **`pci=realloc=off` is backwards for a TB eGPU.** A hot-plugged TB device
   needs its BAR windows *reallocated* behind the TB bridge; you want
   `pci=realloc` (on). The working fix flips exactly this.

The old link/PM knobs aren't *harmful* — they're cheap insurance against a
second, different drop mode — they just were never going to stop the GSP hang
that was actually killing the card.

`egpu-fix-old.sh` writes its settings into dedicated files
(`nvidia-egpu-old-reddit.conf`, a GRUB `.bak`) so it can be applied and rolled
back cleanly without clobbering hand-edited configs. It prints a warning that it
does not disable GSP.

---

## TL;DR

| | Old (`egpu-fix-old.sh`) | Working (`egpu-fix.sh`) |
|---|---|---|
| Theory | link/GPU sleeps and won't wake | GSP firmware RPC hangs |
| Headline change | keep PCIe/TB link + GPU awake | **disable GSP** (`NVreg_EnableGpuFirmware=0`) |
| Needs proprietary module | no | **yes** (open module ignores the GSP knob) |
| `pci=realloc` | `off` | **on** |
| Matches this box's dmesg | ✗ | ✓ |
