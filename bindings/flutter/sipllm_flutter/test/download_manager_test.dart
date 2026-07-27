// Exercises the resumable HF download manager against a local HttpServer that
// serves a deterministic ~2MB blob two ways: a range-capable endpoint and a
// range-oblivious one. Runs on the host VM under `flutter test`.
import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:sipllm_flutter/src/download/hf_download_manager.dart';
import 'package:test/test.dart';

const int _blobSize = 2 * 1024 * 1024;

/// Deterministic pseudo-random payload via a simple LCG so both the server and
/// the sha256 expectation agree byte-for-byte.
Uint8List _buildBlob() {
  final bytes = Uint8List(_blobSize);
  var state = 0x12345678;
  for (var i = 0; i < bytes.length; i++) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    bytes[i] = (state >> 16) & 0xff;
  }
  return bytes;
}

/// Writes [data] in slow 64KB slices so tests have a window to pause/cancel
/// mid-flight before the fast localhost socket drains.
Future<void> _writeSlow(HttpResponse response, Uint8List data) async {
  const slice = 64 * 1024;
  for (var offset = 0; offset < data.length; offset += slice) {
    final end = (offset + slice < data.length) ? offset + slice : data.length;
    response.add(data.sublist(offset, end));
    await response.flush();
    await Future<void>.delayed(const Duration(milliseconds: 40));
  }
}

void main() {
  final blob = _buildBlob();
  final blobSha = sha256.convert(blob).toString();

  late HttpServer server;
  late String base;
  var rangeBytesServed = 0; // Body bytes served by the /range endpoint.
  late Directory tmp;

  setUpAll(() async {
    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    base = 'http://127.0.0.1:${server.port}';
    unawaited(_serve(server, blob, (n) {
      rangeBytesServed += n;
    }));
  });

  tearDownAll(() async {
    await server.close(force: true);
  });

  setUp(() {
    rangeBytesServed = 0;
    tmp = Directory.systemTemp.createTempSync('sipdl_test_');
  });

  tearDown(() {
    if (tmp.existsSync()) tmp.deleteSync(recursive: true);
  });

  Future<String> fileSha(String path) async {
    return sha256.convert(await File(path).readAsBytes()).toString();
  }

  List<String> leftovers(String dest) {
    final dir = Directory(dest).parent;
    return dir
        .listSync()
        .map((e) => e.path.split(Platform.pathSeparator).last)
        .where((name) => name.contains('.part') || name.contains('.sipdl'))
        .toList();
  }

  test('multi-connection download assembles a byte-exact file', () async {
    final manager = DownloadManager(maxConnections: 4);
    final dest = '${tmp.path}/model.bin';
    final task = manager.enqueue(
      url: '$base/range',
      destPath: dest,
      connections: 4,
      sha256: blobSha,
    );

    await task.done;

    expect(task.state, DownloadState.completed);
    expect(await File(dest).length(), _blobSize);
    expect(await fileSha(dest), blobSha);
    expect(leftovers(dest), isEmpty);
    await manager.disposeAll();
  });

  test('pause then resume issues Range requests without re-downloading',
      () async {
    final manager = DownloadManager(maxConnections: 4);
    final dest = '${tmp.path}/model.bin';
    final task = manager.enqueue(
      url: '$base/range',
      destPath: dest,
      connections: 4,
      sha256: blobSha,
    );

    final paused = Completer<int>();
    final sub = task.progress.listen((p) {
      final total = p.total;
      if (!paused.isCompleted &&
          total != null &&
          p.received > 64 * 1024 &&
          p.received < total) {
        task.pause();
        paused.complete(p.received);
      }
    });

    final receivedAtPause = await paused.future;
    await task.settled;
    await sub.cancel();

    expect(task.state, DownloadState.paused);
    expect(receivedAtPause, greaterThan(0));
    expect(receivedAtPause, lessThan(_blobSize));
    // Sidecar + parts must survive a pause for a later resume.
    expect(File('$dest.sipdl.json').existsSync(), isTrue);

    // Reset the server meter: everything served from here is the resume.
    rangeBytesServed = 0;
    await task.resume();
    await task.done;

    expect(task.state, DownloadState.completed);
    expect(await fileSha(dest), blobSha);
    // Proof of true resumption: the resume served strictly less than a full
    // blob (it only fetched the missing tail of each segment).
    expect(rangeBytesServed, lessThan(_blobSize));
    expect(leftovers(dest), isEmpty);
    await manager.disposeAll();
  });

  test('a fresh manager resumes from an existing sidecar (process restart)',
      () async {
    final managerA = DownloadManager(maxConnections: 4);
    final dest = '${tmp.path}/model.bin';
    final taskA = managerA.enqueue(
      url: '$base/range',
      destPath: dest,
      connections: 4,
      sha256: blobSha,
    );

    final paused = Completer<void>();
    final sub = taskA.progress.listen((p) {
      final total = p.total;
      if (!paused.isCompleted &&
          total != null &&
          p.received > 64 * 1024 &&
          p.received < total) {
        taskA.pause();
        paused.complete();
      }
    });
    await paused.future;
    await taskA.settled;
    await sub.cancel();
    expect(taskA.state, DownloadState.paused);

    // Simulate the process dying: drop managerA entirely (client closed),
    // leaving only on-disk parts + sidecar.
    await managerA.disposeAll();
    expect(File('$dest.sipdl.json').existsSync(), isTrue);

    // A brand-new manager picks up where the old one left off.
    final managerB = DownloadManager(maxConnections: 4);
    final taskB = managerB.enqueue(
      url: '$base/range',
      destPath: dest,
      connections: 4,
      sha256: blobSha,
    );
    await taskB.done;

    expect(taskB.state, DownloadState.completed);
    expect(await fileSha(dest), blobSha);
    expect(leftovers(dest), isEmpty);
    await managerB.disposeAll();
  });

  test('range-unsupported endpoint falls back to a single stream', () async {
    final manager = DownloadManager(maxConnections: 4);
    final dest = '${tmp.path}/model.bin';
    final task = manager.enqueue(
      url: '$base/norange',
      destPath: dest,
      connections: 4,
      sha256: blobSha,
    );

    await task.done;

    expect(task.state, DownloadState.completed);
    expect(await File(dest).length(), _blobSize);
    expect(await fileSha(dest), blobSha);
    expect(leftovers(dest), isEmpty);
    await manager.disposeAll();
  });

  test('cancel leaves no part or sidecar files', () async {
    final manager = DownloadManager(maxConnections: 4);
    final dest = '${tmp.path}/model.bin';
    final task = manager.enqueue(
      url: '$base/range',
      destPath: dest,
      connections: 4,
      sha256: blobSha,
    );

    final started = Completer<void>();
    final sub = task.progress.listen((p) {
      if (!started.isCompleted && p.received > 0) started.complete();
    });
    await started.future;
    await sub.cancel();

    task.cancel();
    await task.done;

    expect(task.state, DownloadState.canceled);
    expect(leftovers(dest), isEmpty);
    expect(File(dest).existsSync(), isFalse);
    await manager.disposeAll();
  });
}

/// Serves [blob] on `/range` (with `Accept-Ranges`/`Range` support, metered via
/// [addServed]) and on `/norange` (ignores `Range`, always full 200).
Future<void> _serve(
  HttpServer server,
  Uint8List blob,
  void Function(int) addServed,
) async {
  // Handle each connection concurrently: multi-connection downloads open
  // several sockets at once, and a paused/canceled one must never stall the
  // others (or later tests sharing this server).
  server.listen((request) {
    unawaited(_handle(request, blob, addServed));
  });
}

Future<void> _handle(
  HttpRequest request,
  Uint8List blob,
  void Function(int) addServed,
) async {
  final response = request.response;
  response.headers.set(HttpHeaders.contentTypeHeader, 'application/octet-stream');
  final path = request.uri.path;
  try {
    if (path == '/range') {
      if (request.method == 'HEAD') {
        response.headers.set(HttpHeaders.acceptRangesHeader, 'bytes');
        response.headers.contentLength = blob.length;
        await response.close();
        return;
      }
      final range = request.headers.value(HttpHeaders.rangeHeader);
      response.headers.set(HttpHeaders.acceptRangesHeader, 'bytes');
      if (range != null && range.startsWith('bytes=')) {
        final spec = range.substring('bytes='.length).split('-');
        final start = int.parse(spec[0]);
        final end = (spec.length > 1 && spec[1].isNotEmpty)
            ? int.parse(spec[1])
            : blob.length - 1;
        final chunk = Uint8List.sublistView(blob, start, end + 1);
        response.statusCode = HttpStatus.partialContent;
        response.headers.set(
          HttpHeaders.contentRangeHeader,
          'bytes $start-$end/${blob.length}',
        );
        response.headers.contentLength = chunk.length;
        addServed(chunk.length);
        await _writeSlow(response, chunk);
        await response.close();
      } else {
        response.headers.contentLength = blob.length;
        addServed(blob.length);
        await _writeSlow(response, blob);
        await response.close();
      }
    } else if (path == '/norange') {
      if (request.method == 'HEAD') {
        // Deliberately advertise no range support.
        response.headers.contentLength = blob.length;
        await response.close();
        return;
      }
      response.statusCode = HttpStatus.ok;
      response.headers.contentLength = blob.length;
      await _writeSlow(response, blob);
      await response.close();
    } else {
      response.statusCode = HttpStatus.notFound;
      await response.close();
    }
  } catch (_) {
    // Client hung up mid-stream (expected on pause/cancel); ignore.
    try {
      await response.close();
    } catch (_) {}
  }
}
