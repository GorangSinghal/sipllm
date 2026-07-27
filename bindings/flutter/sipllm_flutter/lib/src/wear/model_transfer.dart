// model_transfer.dart — plain-value types for the phone <-> Wear OS model
// transfer bridge. These cross the MethodChannel/EventChannel boundary as
// StandardMessageCodec maps, so they hold no platform handles and are safe to
// pass around the Dart isolate freely.
//
// A "model" here is a GGUF file (hundreds of MB) copied from the paired phone
// to the watch over the Wearable Data Layer `ChannelClient`. Progress for both
// the sending (phone) and receiving (watch) side is reported as a stream of
// [TransferProgress] snapshots.

/// Which end of the link a progress event describes.
enum TransferDirection {
  /// This device is the sender (the phone pushing a model to the watch).
  send,

  /// This device is the receiver (the watch accepting an incoming model).
  receive;

  /// Parses the wire string emitted by the Android bridge, defaulting to
  /// [send] for unknown values rather than throwing on a malformed event.
  static TransferDirection fromWire(String? s) {
    switch (s) {
      case 'receive':
        return TransferDirection.receive;
      case 'send':
      default:
        return TransferDirection.send;
    }
  }
}

/// Lifecycle of a single transfer, mirrored 1:1 by the Kotlin side.
enum TransferState {
  /// No transfer in progress.
  idle,

  /// Resolving nodes / opening the Data Layer channel.
  connecting,

  /// Bytes are actively moving.
  transferring,

  /// User-requested pause; the channel is closed but progress is retained so a
  /// later [resume] can seek to the last acknowledged offset.
  paused,

  /// All bytes transferred and the sha256 checksum verified.
  completed,

  /// Terminated by an error (I/O, checksum mismatch, node lost, ...).
  failed,

  /// User-requested cancel; any partial destination file is discarded.
  canceled;

  /// Parses the wire string emitted by the Android bridge, defaulting to
  /// [idle] for unknown values.
  static TransferState fromWire(String? s) {
    for (final v in TransferState.values) {
      if (v.name == s) return v;
    }
    return TransferState.idle;
  }
}

/// A peer discovered on the Wearable Data Layer.
///
/// On the phone these are the paired watches; on the watch this is the paired
/// phone. [nearby] reflects the Data Layer's own reachability signal (a Bluetooth
/// or Wi-Fi hop currently exists) — it does not guarantee the high-bandwidth
/// Wi-Fi path is up, only that the node is addressable.
class WearNode {
  const WearNode({
    required this.id,
    required this.displayName,
    required this.nearby,
  });

  /// Opaque Data Layer node id (stable per paired device).
  final String id;

  /// Human-readable name reported by the peer (e.g. "OnePlus Watch 2").
  final String displayName;

  /// Whether the Data Layer currently considers the node directly reachable.
  final bool nearby;

  /// Builds a node from a StandardMessageCodec map sent by the Kotlin bridge.
  factory WearNode.fromMap(Map<Object?, Object?> m) => WearNode(
        id: (m['id'] as String?) ?? '',
        displayName: (m['displayName'] as String?) ?? '',
        nearby: (m['nearby'] as bool?) ?? false,
      );

  @override
  String toString() =>
      'WearNode($displayName, id=$id, nearby=$nearby)';
}

/// A single progress snapshot for an in-flight (or just-finished) transfer.
class TransferProgress {
  const TransferProgress({
    required this.direction,
    required this.nodeId,
    required this.filename,
    required this.sent,
    required this.total,
    required this.bytesPerSecond,
    required this.state,
  });

  /// Whether this snapshot describes a send (phone) or receive (watch).
  final TransferDirection direction;

  /// The peer node id, or null before a node has been resolved.
  final String? nodeId;

  /// The model file's basename (e.g. "qwen2.5-0.5b-q4_0.gguf").
  final String filename;

  /// Bytes transferred so far. On the sender this is the current stream offset
  /// (already including any [resume] seek); on the receiver it is the
  /// destination file length.
  final int sent;

  /// Total byte count from the header, or null if not yet known.
  final int? total;

  /// Instantaneous throughput estimate in bytes/second (0 while connecting).
  final double bytesPerSecond;

  /// The current lifecycle state.
  final TransferState state;

  /// Completion in `[0, 1]`, or null when [total] is unknown/zero so callers
  /// can render an indeterminate indicator.
  double? get fraction {
    final t = total;
    if (t == null || t <= 0) return null;
    final f = sent / t;
    if (f < 0) return 0;
    if (f > 1) return 1;
    return f;
  }

  /// Whether this is a terminal snapshot (no further events will follow for the
  /// current transfer).
  bool get isTerminal =>
      state == TransferState.completed ||
      state == TransferState.failed ||
      state == TransferState.canceled;

  /// Decodes a progress map delivered over the `sipllm/wear/events`
  /// EventChannel. Tolerant of missing/loosely-typed fields (the platform codec
  /// may deliver ints as num, etc.).
  factory TransferProgress.fromMap(Map<Object?, Object?> m) {
    final rawTotal = m['total'];
    return TransferProgress(
      direction: TransferDirection.fromWire(m['direction'] as String?),
      nodeId: m['nodeId'] as String?,
      filename: (m['filename'] as String?) ?? '',
      sent: (m['sent'] as num?)?.toInt() ?? 0,
      total: rawTotal == null ? null : (rawTotal as num).toInt(),
      bytesPerSecond: (m['bytesPerSecond'] as num?)?.toDouble() ?? 0.0,
      state: TransferState.fromWire(m['state'] as String?),
    );
  }

  @override
  String toString() =>
      'TransferProgress(${direction.name}, ${state.name}, $filename, '
      '$sent/${total ?? '?'}, ${bytesPerSecond.toStringAsFixed(0)} B/s)';
}
