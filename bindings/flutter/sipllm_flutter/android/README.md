# sipllm_flutter — Android plugin & Wear OS model-transfer bridge

This module contributes two things to the `sipllm_flutter` plugin's Android
build:

1. **Native build glue** — `build.gradle` drives the shared
   `../src/CMakeLists.txt`, compiling `libsipllm_ffi.so` once per ABI
   (`arm64-v8a`, `armeabi-v7a`, `x86_64`) with `-DANDROID_STL=c++_shared`, and
   packages it into the host APK. The Dart FFI layer loads this `.so` directly;
   the Kotlin here does **not** touch it.
2. **Platform-channel bridge** (`SipllmFlutterPlugin.kt`) — device profiling
   (`sipllm/device`) and the phone ⇆ Wear OS model transfer (`sipllm/wear` +
   `sipllm/wear/events`).

---

## Transport decision: Wearable Data Layer `ChannelClient`

Copying a GGUF model (hundreds of MB) from the phone to the watch uses the
Wearable **Data Layer**, specifically
`com.google.android.gms.wearable.ChannelClient`.

**Why not Bluetooth RFCOMM / SPP?** Wear OS does not expose classic Bluetooth
RFCOMM sockets to apps. Attempting a raw SPP socket between a phone app and a
watch app is not a supported path — the sanctioned mechanism for paired-device
I/O is the Data Layer.

**Why `ChannelClient` specifically?** The Data Layer has three clients:

| Client          | Shape                              | Fit for a large model |
| --------------- | ---------------------------------- | --------------------- |
| `DataClient`    | small synced key/value items       | ✗ size-capped         |
| `MessageClient` | small one-shot payloads (tens KB)  | ✗ size-capped         |
| `ChannelClient` | **bidirectional byte stream**      | ✓ built for bulk      |

`ChannelClient` gives both an `OutputStream` and an `InputStream` over a single
channel, and the platform **auto-negotiates the physical link**: Bluetooth for
the control handshake, and the **Wi-Fi High-Bandwidth** path for the bulk stream
when both devices are on the same Wi-Fi network. The bidirectional stream is
what makes the in-band resume handshake below possible without a second channel.

**Prerequisite:** the two devices must already be paired via the Wear OS
companion app, and both must run a build embedding this plugin under the **same
application id**, so the Data Layer can route the `/sipllm/model-transfer`
channel. With nothing paired, `connectedNodes` returns `[]` and `sendModel`
emits a `failed` progress event.

---

## Resumable protocol (channel path `/sipllm/model-transfer`)

1. **Sender** (phone) opens the channel and writes a newline-terminated JSON
   header: `{"filename","totalSize","sha256","resumeOffset"}`.
2. **Receiver** (watch) reads the header, resolves the destination
   (`filesDir/models/<filename>`), and replies with a newline-terminated JSON
   ack `{"ackOffset"}` — how many valid bytes it already holds (0 when fresh or
   when resume is impossible, e.g. a longer-than-total stale file, which it
   discards).
3. **Sender** seeks the source file to `ackOffset` and streams the remainder,
   emitting throttled progress (~200 ms).
4. **Receiver** appends (`append=true` when resuming) until it holds `totalSize`
   bytes, then verifies `sha256` over the whole file. Mismatch ⇒ delete +
   `failed`.

Control operations:

- **pause** — sender stops the loop and closes the channel; both sides retain
  their offsets and partial files. Emits `paused`.
- **resume** — sender reopens the channel and re-runs the ack handshake, so it
  seeks to wherever the receiver actually got to (not merely its own last
  offset). Robust to bytes lost in a half-closed channel.
- **cancel** — closes the channel; the receiver discards its partial file on an
  explicit cancel. Emits `canceled`.

Integrity is end-to-end: the sha256 in the header is verified against the fully
assembled file on the watch before `completed` is emitted.

---

## Dart surface

```dart
final wear = WearTransfer();
final nodes = await wear.connectedNodes();          // phone: paired watches
wear.events().listen((p) => print(p));              // broadcast progress
await wear.sendModel('/path/model.gguf', resume: true);
await wear.pause(); await wear.resume(); await wear.cancel();
// on the watch:
await wear.listenIncoming();
```

Progress maps decode to `TransferProgress`
(`{direction,nodeId,filename,sent,total,bytesPerSecond,state}`); `state` strings
match `TransferState.name`, `direction` matches `TransferDirection.name`.

---

## Device-pairing test runbook — POCO X6 Pro + OnePlus Watch 2

> None of the steps below have been executed in the current session (no paired
> hardware attached — only an emulator). This is the procedure to run once the
> real devices are on hand.

**Prep**
1. Pair the OnePlus Watch 2 to the POCO X6 Pro through the OnePlus/Wear OS
   companion app. Confirm the watch shows as *Connected*.
2. Put **both** devices on the **same Wi-Fi network** (enables the Wi-Fi
   High-Bandwidth bulk path; without it the transfer still works over Bluetooth,
   just far slower).
3. Enable ADB on both (watch: Developer options → *ADB debugging* +
   *Debug over Wi-Fi*; note its IP).
4. Install the host app (embedding this plugin, same application id) on **both**
   the phone and the watch.
   - `adb -s <phone> install app-phone.apk`
   - `adb connect <watch-ip>:5555 && adb -s <watch> install app-watch.apk`

**Exercise**
5. Watch: call `listenIncoming()`, keep the receiver screen foregrounded.
6. Phone: `connectedNodes()` — assert the watch node appears with
   `nearby == true`.
7. Phone: `sendModel('<path-to>.gguf', resume: true)`. Observe `events()`
   moving through `connecting → transferring` with rising `sent` and a nonzero
   `bytesPerSecond`.
8. Watch: confirm `filesDir/models/<filename>` grows
   (`adb -s <watch> shell run-as <appId> ls -l files/models`).

**Resume path**
9. Mid-transfer, phone: `pause()` — assert both sides emit `paused` and the
   watch's partial file size is retained.
10. Phone: `resume()` — assert the sender seeks to the watch's current length
    (log `ackOffset`) and `sent` continues from there, not from 0.
11. Harsher test: kill the phone app mid-transfer, relaunch, `sendModel(...)`
    again with `resume: true` — the ack handshake should skip the already-sent
    prefix.

**Completion / integrity**
12. Let it finish: assert `completed` on both sides and that the watch file's
    sha256 equals the source's
    (`adb ... shell run-as <appId> sha256sum files/models/<filename>` vs the
    phone's `sha256sum`).
13. Corruption test: truncate the watch's partial file by one byte before the
    final chunk; on completion the receiver must emit `failed` and delete the
    file (sha mismatch).

**Cancel path**
14. Start a fresh send, `cancel()` mid-transfer — assert `canceled` and that the
    watch's partial file is removed.

**Link negotiation sanity**
15. Repeat step 7 once on Wi-Fi and once with the phone's Wi-Fi off (Bluetooth
    only) — both must complete; expect materially higher `bytesPerSecond` on
    Wi-Fi.

---

## NOT yet runtime-verified (explicit)

Everything below compiles and is idiomatic, but has **not** been run against
paired hardware in this session. Each corresponding site in
`WearModelTransfer.kt` is tagged `// RUNTIME-UNVERIFIED: requires paired
POCO X6 Pro + OnePlus Watch 2`:

- `NodeClient.getConnectedNodes()` actually returning the paired watch, and the
  `nearby` flag's real-world accuracy.
- `ChannelClient.openChannel(...)` succeeding and the platform selecting the
  Wi-Fi High-Bandwidth vs Bluetooth link (and the throughput difference).
- The receiver's `ChannelCallback.onChannelOpened` firing on the watch for the
  `/sipllm/model-transfer` path.
- The full send→receive stream, throughput numbers, and progress cadence under
  a real BT/Wi-Fi link (buffering, back-pressure, mid-stream stalls).
- The pause/resume ack handshake seeking correctly across a real channel
  teardown, and app-kill resume.
- sha256 verification over a genuinely transferred (not local-loopback) file.
- The Gradle/CMake/NDK build itself: no `./gradlew`/device build was run here.
  Native `.so` + Kotlin compilation is asserted **by inspection only**.

What *was* verified this session: `dart analyze` on the Dart facade
(`lib/src/wear/model_transfer.dart`, `lib/src/wear/wear_transfer.dart`) reports
no issues.
