// sipllm_device.dart — engine/build/device capabilities that need no model.
import '../ffi/native_library.dart';
import '../ffi/sipllm_bindings.dart';
import 'package:ffi/ffi.dart';

/// Static engine + accelerator capabilities. Cheap to construct.
class SipllmDevice {
  SipllmDevice._(this._b);

  factory SipllmDevice.load({String? libraryPath}) =>
      SipllmDevice._(loadSipllmBindings(libraryPath: libraryPath));

  final SipllmBindings _b;

  String get engineVersion => _b.version().toDartString();

  /// Logical CPU count reported by the native runtime.
  int get hardwareConcurrency => _b.hardwareConcurrency();

  /// Whether the .so was built with the Vulkan backend compiled in.
  bool get vulkanCompiled => _b.vulkanCompiled() != 0;

  /// Whether a usable Vulkan device was found at runtime.
  bool get vulkanAvailable => _b.vulkanAvailable() != 0;

  /// Human-readable accelerator status (device name / fallback reason).
  String get vulkanInfo => _b.vulkanInfo().toDartString();

  /// Benchmark 1..hw threads for [modelPath] and cache the fastest to the
  /// device profile. Expensive (first call) — call off the UI path.
  int optimalThreads(String modelPath, {int ramBudgetBytes = 0}) {
    final p = modelPath.toNativeUtf8();
    try {
      return _b.optimalThreads(p, ramBudgetBytes);
    } finally {
      malloc.free(p);
    }
  }

  void setLogLevel(int level) => _b.setLogLevel(level);
}
