// arch.dart — device/ABI profile + Wear OS detection.
//
// The GGUF model itself is architecture-independent; the only arch-specific
// artifact is libsipllm_ffi.so, which the OS already resolves per-ABI at load
// time. This profile is what the companion phone app and the watch app use to:
//   * gate SIMD / Vulkan expectations (arm64 dotprod vs armv7 scalar),
//   * pick a sane default --ram-budget for a constrained watch,
//   * confirm the peer is actually a Wear OS device before a transfer.
//
// The values come from the Android side (Build.SUPPORTED_ABIS, PackageManager
// FEATURE_WATCH) over a MethodChannel; on non-Android hosts they are inferred
// from dart:io so the same API works in tests and on desktop.
import 'dart:io';

import 'package:flutter/services.dart';

enum CpuArch {
  arm64('arm64-v8a', true),
  arm32('armeabi-v7a', false),
  x64('x86_64', true),
  x86('x86', false),
  unknown('unknown', false);

  const CpuArch(this.androidAbi, this.is64Bit);
  final String androidAbi;
  final bool is64Bit;

  static CpuArch fromAbi(String abi) {
    for (final a in values) {
      if (a.androidAbi == abi) return a;
    }
    return CpuArch.unknown;
  }
}

/// Resolved hardware profile of the running device.
class DeviceProfile {
  const DeviceProfile({
    required this.primaryArch,
    required this.supportedAbis,
    required this.isWearOs,
    required this.cores,
    required this.model,
    required this.androidSdkInt,
  });

  final CpuArch primaryArch;
  final List<CpuArch> supportedAbis;
  final bool isWearOs;
  final int cores;
  final String model;
  final int androidSdkInt;

  bool get is64Bit => primaryArch.is64Bit;

  /// A conservative default RAM budget (bytes) for this device class: watches
  /// are tiny (2 GB class), so stream hard; phones can pin more.
  int get suggestedRamBudgetBytes {
    if (isWearOs) return 220 * 1024 * 1024; // ~220 MB: streams an 8B Q4 model
    return 0; // phones: unlimited streaming by default, user raises the dial
  }

  Map<String, Object> toMap() => {
        'primaryArch': primaryArch.androidAbi,
        'supportedAbis': supportedAbis.map((a) => a.androidAbi).toList(),
        'isWearOs': isWearOs,
        'cores': cores,
        'model': model,
        'androidSdkInt': androidSdkInt,
      };

  @override
  String toString() =>
      'DeviceProfile($model, ${primaryArch.androidAbi}, wearOs=$isWearOs, '
      'cores=$cores, sdk=$androidSdkInt)';
}

/// Detects the running device's arch/Wear profile.
class ArchDetector {
  ArchDetector([MethodChannel? channel])
      : _channel = channel ?? const MethodChannel('sipllm/device');

  final MethodChannel _channel;

  Future<DeviceProfile> detect() async {
    if (Platform.isAndroid) {
      final map = await _channel
          .invokeMapMethod<String, Object?>('getDeviceProfile');
      if (map != null) return _fromNativeMap(map);
    }
    return _inferFromDartIo();
  }

  DeviceProfile _fromNativeMap(Map<String, Object?> m) {
    final abis = (m['supportedAbis'] as List?)
            ?.map((e) => CpuArch.fromAbi(e as String))
            .toList() ??
        <CpuArch>[CpuArch.unknown];
    return DeviceProfile(
      primaryArch: abis.isNotEmpty ? abis.first : CpuArch.unknown,
      supportedAbis: abis,
      isWearOs: (m['isWearOs'] as bool?) ?? false,
      cores: (m['cores'] as int?) ?? Platform.numberOfProcessors,
      model: (m['model'] as String?) ?? 'unknown',
      androidSdkInt: (m['androidSdkInt'] as int?) ?? 0,
    );
  }

  DeviceProfile _inferFromDartIo() {
    // Desktop/test fallback: infer arch from the process, no Wear OS.
    final osArch = _hostArch();
    return DeviceProfile(
      primaryArch: osArch,
      supportedAbis: [osArch],
      isWearOs: false,
      cores: Platform.numberOfProcessors,
      model: Platform.operatingSystemVersion,
      androidSdkInt: 0,
    );
  }

  CpuArch _hostArch() {
    final v = Platform.version.toLowerCase();
    if (v.contains('arm64') || v.contains('aarch64')) return CpuArch.arm64;
    if (v.contains('x64') || v.contains('x86_64')) return CpuArch.x64;
    if (v.contains('arm')) return CpuArch.arm32;
    if (v.contains('ia32') || v.contains('x86')) return CpuArch.x86;
    return CpuArch.unknown;
  }
}
