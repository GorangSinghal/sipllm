// sipllm_runtime.dart — the public, isolate-backed inference runtime.
//
// One [SipllmRuntime] owns one worker isolate holding one native context. It
// exposes streaming generation, cancellation, embeddings, and per-run stats.
import 'dart:async';
import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import '../ffi/native_library.dart';
import '../ffi/sipllm_bindings.dart';
import 'sipllm_isolate.dart';
import 'sipllm_types.dart';

class SipllmException implements Exception {
  SipllmException(this.message);
  final String message;
  @override
  String toString() => 'SipllmException: $message';
}

/// A loaded SipLLM model, driven from a background isolate.
///
/// ```dart
/// final rt = await SipllmRuntime.open(
///   '/path/model.gguf',
///   params: const SipllmParams(ramBudgetBytes: 512 * 1024 * 1024, fastQuant: true),
/// );
/// await for (final tok in rt.generate('The capital of France is')) {
///   stdout.write(tok.piece);
/// }
/// print(rt.lastStats?.decodeTokensPerSecond);
/// await rt.close();
/// ```
class SipllmRuntime {
  SipllmRuntime._(
    this._isolate,
    this._toWorker,
    this._ctxAddress,
    this._cancelFn,
    this.modelInfo,
    this.threads,
  );

  final Isolate _isolate;
  final SendPort _toWorker;
  final int _ctxAddress;
  final void Function(Pointer<SipllmCtx>) _cancelFn;

  /// Static description of the loaded model.
  final SipllmModelInfo modelInfo;

  /// Worker thread count the engine actually spun up.
  final int threads;

  SipllmStats? _lastStats;
  bool _busy = false;
  bool _closed = false;

  /// Metrics from the most recent completed generation.
  SipllmStats? get lastStats => _lastStats;

  bool get isBusy => _busy;

  /// Open [modelPath] (GGUF or .llmw) on a worker isolate.
  static Future<SipllmRuntime> open(
    String modelPath, {
    SipllmParams params = const SipllmParams(),
    String? libraryPath,
    int logLevel = 1,
  }) async {
    // The main isolate loads its own handle so it can call the thread-safe
    // cancel() on the shared context while the worker is blocked in generate().
    final mainBindings = loadSipllmBindings(libraryPath: libraryPath);

    final init = ReceivePort();
    final isolate = await Isolate.spawn(
      sipllmWorkerMain,
      SipllmWorkerInit(
          init.sendPort, libraryPath, modelPath, params.toMap(), logLevel),
      errorsAreFatal: true,
      debugName: 'sipllm-worker',
    );

    final first = await init.first as Map;
    init.close();
    if (first['error'] != null) {
      isolate.kill(priority: Isolate.immediate);
      throw SipllmException(first['error'] as String);
    }
    return SipllmRuntime._(
      isolate,
      first['toWorker'] as SendPort,
      first['ctxAddress'] as int,
      mainBindings.cancel,
      SipllmModelInfo.fromMap(first['modelInfo'] as Map<String, Object?>),
      first['threads'] as int,
    );
  }

  /// Stream generated pieces for [prompt]. Populates [lastStats] on completion.
  /// Throws if a generation is already in flight (a runtime is single-context).
  Stream<SipllmToken> generate(
    String prompt, {
    int maxNew = 256,
    SipllmSampler sampler = const SipllmSampler(),
  }) {
    if (_closed) {
      return Stream.error(SipllmException('runtime is closed'));
    }
    if (_busy) {
      return Stream.error(
          SipllmException('a generation is already in progress'));
    }
    _busy = true;
    final controller = StreamController<SipllmToken>();
    final port = ReceivePort();

    port.listen((message) {
      final m = message as Map;
      if (m['token'] != null) {
        controller.add(SipllmToken(m['token'] as String, m['id'] as int));
      } else if (m['done'] == true) {
        _lastStats = SipllmStats.fromMap(m['stats'] as Map<String, Object?>);
        _finish(port, controller);
      } else if (m['error'] != null) {
        controller.addError(SipllmException(m['error'] as String));
        _finish(port, controller);
      }
    });

    controller.onCancel = () {
      // Consumer stopped listening: ask the engine to stop too.
      if (_busy) cancel();
    };

    _toWorker.send({
      'cmd': 'generate',
      'prompt': prompt,
      'maxNew': maxNew,
      'sampler': sampler.toMap(),
      'port': port.sendPort,
    });
    return controller.stream;
  }

  void _finish(ReceivePort port, StreamController<SipllmToken> controller) {
    _busy = false;
    port.close();
    controller.close();
  }

  /// Convenience: collect the full completion as a single string.
  Future<String> complete(
    String prompt, {
    int maxNew = 256,
    SipllmSampler sampler = const SipllmSampler(),
  }) async {
    final buf = StringBuffer();
    await for (final tok in generate(prompt, maxNew: maxNew, sampler: sampler)) {
      buf.write(tok.piece);
    }
    return buf.toString();
  }

  /// Ask the in-flight generation to stop at the next token boundary. Safe to
  /// call from the UI isolate while the worker is mid-generate (thread-safe
  /// atomic in native code).
  void cancel() {
    if (_closed) return;
    _cancelFn(Pointer<SipllmCtx>.fromAddress(_ctxAddress));
  }

  /// Compute an L2-normalized embedding for [text].
  Future<Float32List> embed(
    String text, {
    Pooling pooling = Pooling.last,
  }) async {
    if (_closed) throw SipllmException('runtime is closed');
    if (_busy) throw SipllmException('runtime is busy');
    _busy = true;
    final port = ReceivePort();
    final completer = Completer<Float32List>();
    port.listen((message) {
      final m = message as Map;
      if (m['embedding'] != null) {
        completer.complete(m['embedding'] as Float32List);
      } else {
        completer.completeError(
            SipllmException((m['error'] ?? 'embed failed') as String));
      }
      port.close();
    });
    _toWorker.send({
      'cmd': 'embed',
      'text': text,
      'pooling': pooling.code,
      'port': port.sendPort,
    });
    try {
      return await completer.future;
    } finally {
      _busy = false;
    }
  }

  /// Clear KV cache / conversation state.
  void reset() {
    if (_closed) return;
    _toWorker.send({'cmd': 'reset'});
  }

  /// Free the native context and shut down the worker isolate.
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    if (_busy) cancel();
    _toWorker.send({'cmd': 'close'});
    // Give the worker a beat to free the context, then hard-stop the isolate.
    await Future<void>.delayed(const Duration(milliseconds: 50));
    _isolate.kill(priority: Isolate.beforeNextEvent);
  }
}
