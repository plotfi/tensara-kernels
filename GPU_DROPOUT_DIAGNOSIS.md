# GPU Drop-Out Diagnosis — RTX 4070 Ti falls off the PCIe bus

> Root-cause writeup from analyzing `dmesg.log` (kernel log captured 2026-07-31).
> TL;DR: the GPU **fell off the PCIe bus (Xid 79)** mid-workload. This is a
> hardware/link fault, **not** a bug in the CUDA kernels. Recovery requires an
> **OS reboot**.

## Machine / GPU

- Host: "Micro Computer (HK) Tech Limited AI Series/ARBSC", BIOS 1.01 06/19/2025
  (a small-form-factor / mini "AI" box).
- GPU: NVIDIA RTX 4070 Ti (`[10de:2782]`) at PCI `0000:03:00.0`, audio fn `.1`.
- Driver: `libcuda.so.595.71.05` (NVIDIA 595.x).
- Attached under `pcieport 0000:00:07.0`, enumerated as a **PCIe *Legacy*
  Endpoint**.

## The event chain (in order, from dmesg)

1. **`Xid 79 ... GPU has fallen off the bus.`**  ← ROOT EVENT (first error, line ~1367)
   - `pid=50451, name=avg-pool-1d.exe` — this is just the process holding the GPU
     at the moment it dropped; it did **not** cause the fault.
   - `krcRcAndNotifyAllChannels ... critical error 79`,
     `API_GPU_ATTACHED_SANITY_CHECK failed!`
2. The driver was mid-`GSP_RM_CONTROL` RPC — specifically **`kperfBoostSet_IMPL`**
   (bumping GPU clocks/power as the kernel launched). `_kgspRpcRecvPoll` never got
   a reply from the GSP (GPU System Processor firmware); RPC history shows an
   incomplete `GSP_LOAD_EXEC_HS_BINA`.
3. **`tmrGetTimeEx_GM107: Consistently Bad TimeLo value ffffffff`** — reading a GPU
   hardware register returns **all-Fs**. Classic signature of a device no longer on
   the bus: MMIO read → no response → PCIe root complex returns `0xffffffff`.
4. **`Xid 154 ... GPU recovery action ... 0x2 (Node Reboot Required)`**.
5. **`uvm encountered global fatal error 0x60, requiring os reboot to recover`**,
   then a storm of `NV_ERR_GPU_IN_FULLCHIP_RESET` assertion failures.
6. **`general protection fault ... in libcuda.so`** — the userspace segfault
   (`avg-pool-1d.exe`, and later a trivial CUDA test) is downstream fallout: libcuda
   crashed because the driver/GPU state died underneath it.

## Why this is NOT the kernel code

- Xid 79 is a bus-level disappearance of the whole GPU, not a kernel fault
  (that would be Xid 13/31/43 etc. with a specific bad-instruction/MMU signature).
- A **trivial one-line CUDA kernel** crashed identically right after avg-pool-1d,
  confirming the device — not the workload — is gone.
- The failure fired inside `kperfBoostSet` (clock/power boost at launch), i.e. on
  the **power/clock ramp**, not during actual compute.
- Symptoms on a dead device that look like "bugs" but aren't:
  - `nvidia-smi` → "No devices were found" (though `/dev/nvidia*` nodes persist).
  - Harness output all-zero — the harness's `cudaMemcpy` isn't error-checked, so a
    failed launch just leaves the pre-zeroed buffer.
  - Managed-memory (`cudaMallocManaged`) test **segfaults** on host access, because
    the dead driver can't back the pages.

## Most likely physical cause — marginal PCIe link / power

Strongest evidence, from boot (line ~513):

```
0000:03:00.0: 8.000 Gb/s available PCIe bandwidth, limited by 2.5 GT/s PCIe x4 link
             at 0000:00:07.0 (capable of 252.048 Gb/s with 16.0 GT/s PCIe x16 link)
```

- The 4070 Ti is training at **PCIe Gen1 x4** — ~1/32 of its Gen4 x16 capability —
  and shows up as a "Legacy Endpoint". On an SFF/mini box this almost always means
  the card is on an **OCuLink / M.2-to-PCIe riser or eGPU-style adapter**.
- A link that falls back to **Gen1** is a hallmark of a **marginal cable/connector
  or signal-integrity / power problem**. Under load the GPU boosts power draw and
  clocks, the link or power rail glitches, and the card drops off the bus.
- No thermal Xids and no ECC (double-bit) errors were logged, so overheating and
  VRAM faults are unlikely to be the primary trigger here — it points at the
  **connection/power**, not heat or memory.

(Secondary, less likely: a GSP-firmware/driver hang. Xid 79 is rarely driver-caused,
and the all-Fs register reads argue for a genuine off-the-bus condition.)

## Recovery + mitigation

1. **Reboot.** Once Xid 154 "Node Reboot Required" / uvm fatal `0x60` fire, an OS
   reboot is the only recovery. `nvidia-smi -r` (GPU reset) generally can't recover a
   card that's already off the bus.
2. After reboot, **verify the link retrained** and reseat the physical connection —
   chase that Gen1 x4:
   ```sh
   nvidia-smi -q | grep -iA3 'PCIe\|Link'
   sudo lspci -vv -s 03:00.0 | grep -i 'LnkSta\|LnkCap'
   ```
   Reseat the OCuLink/riser cable (both ends) and the GPU power connectors.
3. If it keeps dropping under load, soften the boost transient as a workaround while
   you sort the hardware:
   ```sh
   sudo nvidia-smi -pl <watts>      # cap power limit
   sudo nvidia-smi -lgc <min,max>   # lock graphics clocks
   ```
4. Watch whether drops correlate with heavy load / clock boost. If reseating +
   power-cap stops it, it was the link/power; if it persists at Gen1 even idle, the
   riser/adapter or slot is suspect.

## What was blocked by this

The `generalize-pooling-scalar-op` branch (op-generalized 1-D pooling +
`max-pool-1d` implementation) is written and compiles cleanly but is **unvalidated**
— it needs a healthy GPU to run `tests/run_tests.sh avg-pool-1d max-pool-1d`. See
`POOLING_GENERALIZATION_HANDOFF.md`.
