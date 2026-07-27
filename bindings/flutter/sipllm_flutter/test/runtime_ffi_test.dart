// runtime_ffi_test.dart — exercises the full Dart -> isolate -> FFI -> engine
// path against a real (toy) GGUF model. Requires:
//   * the desktop dylib built by ffi/build_desktop.sh (auto-discovered), and
//   * a model at $SIPLLM_TEST_MODEL (defaults to /tmp/sip_toy.gguf).
// Skips gracefully if the model is absent so CI without a model stays green.
import 'dart:io';
import 'dart:math';

import 'package:test/test.dart';

import 'package:sipllm_flutter/src/runtime/sipllm_runtime.dart';
import 'package:sipllm_flutter/src/runtime/sipllm_types.dart';
import 'package:sipllm_flutter/src/device/sipllm_device.dart';

void main() {
  final modelPath =
      Platform.environment['SIPLLM_TEST_MODEL'] ?? '/tmp/sip_toy.gguf';
  final hasModel = File(modelPath).existsSync();

  test('device info loads from native library', () {
    final dev = SipllmDevice.load();
    expect(dev.engineVersion, isNotEmpty);
    expect(dev.hardwareConcurrency, greaterThan(0));
    // vulkanInfo must be callable even when not compiled in.
    expect(dev.vulkanInfo, isNotNull);
  });

  test('open + streaming generate + stats', () async {
    if (!hasModel) {
      markTestSkipped('no model at $modelPath');
      return;
    }
    final rt = await SipllmRuntime.open(
      modelPath,
      params: const SipllmParams(threads: 2),
    );
    addTearDown(rt.close);

    expect(rt.modelInfo.dim, greaterThan(0));
    expect(rt.modelInfo.nLayers, greaterThan(0));
    expect(rt.threads, greaterThan(0));

    final pieces = <String>[];
    await for (final tok in rt.generate('hello',
        maxNew: 8, sampler: const SipllmSampler.greedy())) {
      pieces.add(tok.piece);
    }
    expect(pieces, isNotEmpty);
    expect(rt.lastStats, isNotNull);
    expect(rt.lastStats!.genTokens, greaterThan(0));
    expect(rt.lastStats!.decodeTokensPerSecond, greaterThan(0));
  }, timeout: const Timeout(Duration(seconds: 30)));

  test('cancel stops generation early', () async {
    if (!hasModel) {
      markTestSkipped('no model at $modelPath');
      return;
    }
    final rt = await SipllmRuntime.open(modelPath,
        params: const SipllmParams(threads: 2));
    addTearDown(rt.close);

    var count = 0;
    final stream = rt.generate('hello',
        maxNew: 1000, sampler: const SipllmSampler.greedy());
    await for (final _ in stream) {
      count++;
      if (count >= 3) rt.cancel();
    }
    // Cancel is best-effort at token boundaries; must be far below the cap.
    expect(count, lessThan(1000));
    expect(rt.lastStats!.genTokens, lessThan(1000));
  }, timeout: const Timeout(Duration(seconds: 30)));

  test('embed returns an L2-normalized vector of model dim', () async {
    if (!hasModel) {
      markTestSkipped('no model at $modelPath');
      return;
    }
    final rt = await SipllmRuntime.open(modelPath,
        params: const SipllmParams(threads: 2));
    addTearDown(rt.close);

    final v = await rt.embed('the capital of france is');
    expect(v.length, rt.modelInfo.dim);
    final norm = sqrt(v.fold<double>(0, (a, b) => a + b * b));
    expect(norm, closeTo(1.0, 1e-3));
  }, timeout: const Timeout(Duration(seconds: 30)));
}
