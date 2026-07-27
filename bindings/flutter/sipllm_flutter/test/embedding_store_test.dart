import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:sipllm_flutter/src/embedding/embedding_store.dart';
import 'package:test/test.dart';

/// Builds an L2-normalized [Float32List] from raw components, mirroring the
/// engine's contract that stored embeddings are unit vectors.
Float32List unit(List<double> xs) {
  var norm = 0.0;
  for (final x in xs) {
    norm += x * x;
  }
  norm = math.sqrt(norm);
  return Float32List.fromList([for (final x in xs) x / norm]);
}

void main() {
  group('EmbeddingStore.search', () {
    late EmbeddingStore store;

    setUp(() {
      store = EmbeddingStore.inMemory(dim: 4);
      store.add(text: 'x-axis', embedding: unit([1, 0, 0, 0]));
      store.add(text: 'y-axis', embedding: unit([0, 1, 0, 0]));
      store.add(text: 'z-axis', embedding: unit([0, 0, 1, 0]));
      store.add(text: 'diag-xy', embedding: unit([1, 1, 0, 0]));
    });

    tearDown(() => store.close());

    test('returns nearest neighbors ordered by descending cosine', () {
      final results = store.search(unit([1, 0, 0, 0]), topK: 3);

      expect(results, hasLength(3));
      expect(results[0].text, 'x-axis');
      // Score is descending.
      for (var i = 1; i < results.length; i++) {
        expect(results[i - 1].score, greaterThanOrEqualTo(results[i].score));
      }
      // Exact match against a unit vector has cosine ~1.0.
      expect(results[0].score, closeTo(1.0, 1e-6));
      // diag-xy shares the x component: cos = 1/sqrt(2).
      expect(results[1].text, 'diag-xy');
      expect(results[1].score, closeTo(1 / math.sqrt(2), 1e-6));
    });

    test('normalizes an unnormalized query defensively', () {
      final normalized = store.search(unit([1, 1, 0, 0]));
      final scaled = store.search(Float32List.fromList([5, 5, 0, 0]));

      expect(scaled[0].text, normalized[0].text);
      expect(scaled[0].score, closeTo(normalized[0].score, 1e-6));
      expect(scaled[0].score, closeTo(1.0, 1e-6));
    });

    test('topK caps the result count and non-positive topK yields empty', () {
      expect(store.search(unit([1, 0, 0, 0]), topK: 2), hasLength(2));
      expect(store.search(unit([1, 0, 0, 0]), topK: 0), isEmpty);
      // topK larger than the corpus returns everything.
      expect(store.search(unit([1, 0, 0, 0]), topK: 99), hasLength(4));
    });
  });

  group('dimension validation', () {
    test('add throws ArgumentError on dim mismatch', () {
      final store = EmbeddingStore.inMemory(dim: 4);
      addTearDown(store.close);
      expect(
        () => store.add(text: 'bad', embedding: Float32List.fromList([1, 2, 3])),
        throwsArgumentError,
      );
    });

    test('search throws ArgumentError on dim mismatch', () {
      final store = EmbeddingStore.inMemory(dim: 4);
      addTearDown(store.close);
      expect(
        () => store.search(Float32List.fromList([1, 0])),
        throwsArgumentError,
      );
    });

    test('addBatch validates every record before writing anything', () {
      final store = EmbeddingStore.inMemory(dim: 4);
      addTearDown(store.close);
      expect(
        () => store.addBatch([
          EmbeddingRecord(text: 'ok', embedding: unit([1, 0, 0, 0])),
          EmbeddingRecord(text: 'bad', embedding: Float32List.fromList([1, 0])),
        ]),
        throwsArgumentError,
      );
      // Nothing committed because validation ran before the transaction.
      expect(store.length, 0);
    });

    test('open rejects a mismatched dimension', () {
      final dir = Directory.systemTemp.createTempSync('embstore_dim');
      addTearDown(() => dir.deleteSync(recursive: true));
      final path = '${dir.path}/e.db';

      EmbeddingStore.open(path, dim: 4).close();
      expect(
        () => EmbeddingStore.open(path, dim: 8),
        throwsA(isA<StateError>()),
      );
    });

    test('constructors reject a non-positive dimension', () {
      expect(() => EmbeddingStore.inMemory(dim: 0), throwsArgumentError);
      expect(() => EmbeddingStore.open(':memory:', dim: -1), throwsArgumentError);
    });
  });

  group('metadata filtering', () {
    test('where restricts candidates by decoded metadata', () {
      final store = EmbeddingStore.inMemory(dim: 4);
      addTearDown(store.close);
      store.add(
        text: 'apple',
        embedding: unit([1, 0, 0, 0]),
        metadata: {'kind': 'fruit'},
      );
      store.add(
        text: 'carrot',
        embedding: unit([1, 0, 0, 0]),
        metadata: {'kind': 'veg'},
      );
      store.add(
        text: 'banana',
        embedding: unit([0.9, 0.1, 0, 0]),
        metadata: {'kind': 'fruit'},
      );

      final fruits = store.search(
        unit([1, 0, 0, 0]),
        where: (meta) => meta?['kind'] == 'fruit',
      );

      expect(fruits.map((m) => m.text), everyElement(isIn(['apple', 'banana'])));
      expect(fruits.any((m) => m.text == 'carrot'), isFalse);
      expect(fruits.first.metadata?['kind'], 'fruit');
    });
  });

  group('CRUD: addBatch, length, delete, clear, get', () {
    test('addBatch inserts all rows in one transaction', () {
      final store = EmbeddingStore.inMemory(dim: 4);
      addTearDown(store.close);
      store.addBatch([
        EmbeddingRecord(text: 'a', embedding: unit([1, 0, 0, 0])),
        EmbeddingRecord(
          text: 'b',
          embedding: unit([0, 1, 0, 0]),
          metadata: {'n': 2},
        ),
        EmbeddingRecord(text: 'c', embedding: unit([0, 0, 1, 0])),
      ]);
      expect(store.length, 3);
    });

    test('get reconstructs text, vector and metadata', () {
      final store = EmbeddingStore.inMemory(dim: 4);
      addTearDown(store.close);
      final vec = unit([1, 2, 3, 4]);
      final id = store.add(text: 'v', embedding: vec, metadata: {'tag': 't'});

      final rec = store.get(id)!;
      expect(rec.text, 'v');
      expect(rec.metadata?['tag'], 't');
      expect(rec.embedding.length, 4);
      for (var i = 0; i < 4; i++) {
        expect(rec.embedding[i], closeTo(vec[i], 1e-6));
      }
      expect(store.get(999999), isNull);
    });

    test('delete removes a single row; clear empties the store', () {
      final store = EmbeddingStore.inMemory(dim: 4);
      addTearDown(store.close);
      final a = store.add(text: 'a', embedding: unit([1, 0, 0, 0]));
      store.add(text: 'b', embedding: unit([0, 1, 0, 0]));
      expect(store.length, 2);

      store.delete(a);
      expect(store.length, 1);
      expect(store.get(a), isNull);
      // Deleting a non-existent id is a no-op.
      store.delete(a);
      expect(store.length, 1);

      store.clear();
      expect(store.length, 0);
      // Dimension survives clear().
      expect(store.dim, 4);
      expect(
        () => store.add(text: 'c', embedding: Float32List.fromList([1])),
        throwsArgumentError,
      );
    });
  });

  group('file-backed persistence', () {
    test('survives a close/reopen round-trip', () {
      final dir = Directory.systemTemp.createTempSync('embstore_persist');
      addTearDown(() => dir.deleteSync(recursive: true));
      final path = '${dir.path}/vectors.db';

      final w = EmbeddingStore.open(path, dim: 4);
      w.addBatch([
        EmbeddingRecord(
          text: 'north',
          embedding: unit([0, 1, 0, 0]),
          metadata: {'dir': 'N'},
        ),
        EmbeddingRecord(text: 'east', embedding: unit([1, 0, 0, 0])),
      ]);
      expect(w.length, 2);
      w.close();

      final r = EmbeddingStore.open(path, dim: 4);
      addTearDown(r.close);
      expect(r.length, 2);

      final hits = r.search(unit([0, 1, 0, 0]), topK: 1);
      expect(hits, hasLength(1));
      expect(hits.first.text, 'north');
      expect(hits.first.score, closeTo(1.0, 1e-6));
      expect(hits.first.metadata?['dir'], 'N');
    });
  });
}
