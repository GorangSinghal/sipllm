// wear_transfer.dart — Dart facade over the Android Wearable Data Layer bridge
// that copies a model file between a paired phone and watch.
import 'package:flutter/services.dart';

import 'model_transfer.dart';

/// Phone <-> Wear OS model-transfer client.
///
/// ## Transport
/// Bulk model files are moved over the Wearable **Data Layer**
/// (`com.google.android.gms.wearable.ChannelClient`), *not* over a raw
/// Bluetooth RFCOMM/SPP socket. RFCOMM is deliberately not exposed to Wear OS
/// apps; the Data Layer is the sanctioned channel for paired-device I/O and it
/// auto-negotiates the physical link: Bluetooth for control/handshake, and the
/// **Wi-Fi High-Bandwidth** path for the bulk stream when both devices are on
/// the same network. A `ChannelClient` channel is a bidirectional byte stream —
/// exactly what a large GGUF copy needs — with far higher throughput than
/// `MessageClient`/`DataClient` (which cap payloads in the tens/hundreds of KB).
///
/// ## Prerequisite
/// The two devices **must already be paired** through the Wear OS companion app
/// and both must have this plugin's package installed with the same
/// application id, so the Data Layer can route the `/sipllm/model-transfer`
/// channel. With no paired peer, [connectedNodes] returns an empty list and
/// [sendModel] reports a [TransferState.failed] progress event.
///
/// ## Resumability
/// The protocol is resumable: the sender writes a small JSON header
/// (`filename`, `totalSize`, `sha256`, `resumeOffset`); the receiver appends to
/// the destination file and, when it already holds bytes, replies with its
/// current length so the sender seeks forward. [pause] tears the channel down
/// while keeping offsets; [resume] reopens it and re-negotiates the offset;
/// [cancel] discards the partial file.
///
/// A single [WearTransfer] instance drives one transfer at a time (the Android
/// bridge is single-slot); construct per-transfer or serialize calls.
class WearTransfer {
  /// Creates a client bound to the default plugin channels. Custom channels may
  /// be injected for testing.
  WearTransfer({MethodChannel? methodChannel, EventChannel? eventChannel})
      : _method = methodChannel ?? const MethodChannel(_methodName),
        _events = eventChannel ?? const EventChannel(_eventName);

  static const String _methodName = 'sipllm/wear';
  static const String _eventName = 'sipllm/wear/events';

  final MethodChannel _method;
  final EventChannel _events;

  Stream<TransferProgress>? _broadcast;

  /// Lists Data Layer nodes reachable from this device: the paired watches when
  /// called on a phone, or the paired phone when called on a watch. Empty when
  /// nothing is paired/reachable.
  Future<List<WearNode>> connectedNodes() async {
    final raw = await _method.invokeListMethod<Object?>('connectedNodes');
    if (raw == null) return const <WearNode>[];
    return raw
        .whereType<Map<Object?, Object?>>()
        .map(WearNode.fromMap)
        .toList(growable: false);
  }

  /// A broadcast stream of [TransferProgress] snapshots for both send and
  /// receive activity. Backed by the `sipllm/wear/events` EventChannel; the
  /// underlying platform stream is subscribed lazily and shared across
  /// listeners, so callers can freely attach/detach.
  Stream<TransferProgress> events() {
    return _broadcast ??= _events
        .receiveBroadcastStream()
        .map<TransferProgress>((dynamic e) {
          final map = (e as Map).cast<Object?, Object?>();
          return TransferProgress.fromMap(map);
        })
        .asBroadcastStream();
  }

  /// Sends the model file at [path] to a paired watch.
  ///
  /// If [nodeId] is omitted the bridge picks the first connected node. When
  /// [resume] is true (the default) and the receiver already holds a partial
  /// file, the sender seeks to the receiver's current length instead of
  /// restarting from zero. Progress is reported via [events]; this future
  /// completes once the transfer has been *started*, not finished.
  Future<void> sendModel(String path, {String? nodeId, bool resume = true}) {
    return _method.invokeMethod<void>('sendModel', <String, Object?>{
      'path': path,
      'nodeId': nodeId,
      'resume': resume,
    });
  }

  /// Watch side: register the incoming-channel callback so this device accepts
  /// a `/sipllm/model-transfer` channel opened by the paired phone. Idempotent.
  Future<void> listenIncoming() {
    return _method.invokeMethod<void>('listenIncoming');
  }

  /// Pauses the active transfer, closing the channel but retaining offsets so a
  /// later [resume] continues where it stopped.
  Future<void> pause() => _method.invokeMethod<void>('pause');

  /// Resumes a previously [pause]d transfer, reopening the channel and
  /// re-negotiating the resume offset with the peer.
  Future<void> resume() => _method.invokeMethod<void>('resume');

  /// Cancels the active transfer and discards any partial destination file.
  Future<void> cancel() => _method.invokeMethod<void>('cancel');
}
