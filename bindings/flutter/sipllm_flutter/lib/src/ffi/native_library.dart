// native_library.dart — locate and open libsipllm_ffi across Flutter runtimes
// and the plain Dart VM (`dart test`).
//
// Resolution order:
//   1. explicit path argument
//   2. SIPLLM_FFI_LIBRARY environment variable (used by tests / CI)
//   3. the plugin's own build/desktop artifact (walking up from CWD)
//   4. the platform default (Flutter bundles the lib next to the app)
//   5. DynamicLibrary.process() (iOS/macOS static linking)
import 'dart:ffi';
import 'dart:io';

import 'sipllm_bindings.dart';

const String _libStem = 'sipllm_ffi';

String get _platformFileName {
  if (Platform.isWindows) return '$_libStem.dll';
  if (Platform.isMacOS || Platform.isIOS) return 'lib$_libStem.dylib';
  return 'lib$_libStem.so'; // android, linux
}

DynamicLibrary _openFrom(String path) => DynamicLibrary.open(path);

/// Opens the SipLLM native library and returns resolved [SipllmBindings].
SipllmBindings loadSipllmBindings({String? libraryPath}) {
  return SipllmBindings(openSipllmDynamicLibrary(libraryPath: libraryPath));
}

DynamicLibrary openSipllmDynamicLibrary({String? libraryPath}) {
  final explicit = libraryPath ?? Platform.environment['SIPLLM_FFI_LIBRARY'];
  if (explicit != null && explicit.isNotEmpty) return _openFrom(explicit);

  // Desktop / test discovery: search a few likely artifact locations so
  // `dart test` works right after `ffi/build_desktop.sh`.
  for (final candidate in _desktopCandidates()) {
    if (File(candidate).existsSync()) return _openFrom(candidate);
  }

  // Flutter runtime default: the embedder places the lib on the loader path.
  try {
    return _openFrom(_platformFileName);
  } on ArgumentError {
    // iOS/macOS static linking: symbols live in the running process image.
    if (Platform.isMacOS || Platform.isIOS) return DynamicLibrary.process();
    rethrow;
  }
}

Iterable<String> _desktopCandidates() sync* {
  final fname = _platformFileName;
  // Walk up from the current directory looking for the plugin build output.
  var dir = Directory.current;
  for (var i = 0; i < 6; i++) {
    yield '${dir.path}/build/desktop/$fname';
    yield '${dir.path}/bindings/flutter/sipllm_flutter/build/desktop/$fname';
    final parent = dir.parent;
    if (parent.path == dir.path) break;
    dir = parent;
  }
}
