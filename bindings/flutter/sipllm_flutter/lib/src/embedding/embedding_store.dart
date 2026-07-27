// embedding_store.dart — a small, local, SQLite-backed vector store for
// on-device embeddings. Pure Dart on top of `package:sqlite3`, so it is
// unit-testable on the host VM (no Flutter engine required).
//
// The engine produces L2-normalized `Float32List` embeddings, so cosine
// similarity reduces to a plain dot product. Search is a full linear scan,
// which is entirely adequate at the scale we keep on a phone or watch (a few
// thousand rows). Approximate nearest-neighbor indexing (HNSW / IVF) is a
// future optimization; the public API would not need to change to add it.
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:sqlite3/sqlite3.dart';

/// A row to insert into an [EmbeddingStore].
class EmbeddingRecord {
  /// Creates a record. [embedding] must have length equal to the store's
  /// dimension; [metadata], when given, is persisted as JSON and must be
  /// JSON-encodable.
  const EmbeddingRecord({
    required this.text,
    required this.embedding,
    this.metadata,
  });

  /// The source text the embedding was computed from.
  final String text;

  /// The embedding vector. Expected to be L2-normalized by the engine.
  final Float32List embedding;

  /// Optional JSON-encodable metadata stored alongside the vector.
  final Map<String, Object?>? metadata;
}

/// A search hit, ordered by descending [score].
class EmbeddingMatch {
  /// Creates a match.
  const EmbeddingMatch({
    required this.id,
    required this.text,
    required this.score,
    this.metadata,
  });

  /// The row id of the matched embedding.
  final int id;

  /// The stored text for the matched embedding.
  final String text;

  /// Cosine similarity (dot product of L2-normalized vectors) with the query,
  /// in `[-1, 1]`. Higher is more similar.
  final double score;

  /// The metadata decoded from JSON, or `null` if none was stored.
  final Map<String, Object?>? metadata;
}

/// A local SQLite-backed store of text + embedding vectors with cosine
/// similarity search.
///
/// The on-disk schema is a single `embeddings` table plus a `store_meta` row
/// recording the fixed vector [dim]. Vectors are stored as little-endian
/// float32 BLOBs and reconstructed in place via [Float32List.view].
class EmbeddingStore {
  EmbeddingStore._(this._db, this.dim);

  final Database _db;

  /// The fixed dimensionality of every vector in this store.
  final int dim;

  /// Opens (creating if absent) a file-backed store at [path].
  ///
  /// If the database already exists, its recorded dimension must equal [dim];
  /// otherwise a [StateError] is thrown. Passing `':memory:'` as [path] yields
  /// a private in-memory database (prefer [inMemory] for clarity).
  static EmbeddingStore open(String path, {required int dim}) {
    if (dim <= 0) {
      throw ArgumentError.value(dim, 'dim', 'must be a positive integer');
    }
    final db = sqlite3.open(path);
    return _init(db, dim);
  }

  /// Opens a private, ephemeral in-memory store. Data is discarded on [close].
  static EmbeddingStore inMemory({required int dim}) {
    if (dim <= 0) {
      throw ArgumentError.value(dim, 'dim', 'must be a positive integer');
    }
    final db = sqlite3.openInMemory();
    return _init(db, dim);
  }

  static EmbeddingStore _init(Database db, int dim) {
    try {
      db.execute('PRAGMA journal_mode = WAL;');
      db.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
          id   INTEGER PRIMARY KEY AUTOINCREMENT,
          text TEXT    NOT NULL,
          dim  INTEGER NOT NULL,
          vec  BLOB    NOT NULL,
          meta TEXT
        );
      ''');
      db.execute('''
        CREATE TABLE IF NOT EXISTS store_meta (
          key   TEXT PRIMARY KEY,
          value INTEGER NOT NULL
        );
      ''');

      final existing = db.select(
        'SELECT value FROM store_meta WHERE key = ?;',
        const ['dim'],
      );
      if (existing.isEmpty) {
        db.execute(
          'INSERT INTO store_meta (key, value) VALUES (?, ?);',
          ['dim', dim],
        );
      } else {
        final stored = existing.first['value'] as int;
        if (stored != dim) {
          throw StateError(
            'EmbeddingStore dimension mismatch: database was created with '
            'dim=$stored but was opened with dim=$dim.',
          );
        }
      }
    } catch (_) {
      db.dispose();
      rethrow;
    }
    return EmbeddingStore._(db, dim);
  }

  /// The number of embeddings currently stored.
  int get length {
    final row = _db.select('SELECT COUNT(*) AS n FROM embeddings;');
    return row.first['n'] as int;
  }

  /// Inserts one embedding and returns its new row id.
  ///
  /// Throws [ArgumentError] if `embedding.length != dim`.
  int add({
    required String text,
    required Float32List embedding,
    Map<String, Object?>? metadata,
  }) {
    _checkDim(embedding);
    _db.execute(
      'INSERT INTO embeddings (text, dim, vec, meta) VALUES (?, ?, ?, ?);',
      [text, dim, _blobOf(embedding), _encodeMeta(metadata)],
    );
    return _db.lastInsertRowId;
  }

  /// Inserts many embeddings in a single transaction using a prepared
  /// statement. Either all rows are committed or, on error, none are.
  ///
  /// Throws [ArgumentError] if any record's embedding length differs from
  /// [dim] (checked before any write begins).
  void addBatch(List<EmbeddingRecord> records) {
    if (records.isEmpty) return;
    for (final r in records) {
      _checkDim(r.embedding);
    }
    final stmt = _db.prepare(
      'INSERT INTO embeddings (text, dim, vec, meta) VALUES (?, ?, ?, ?);',
    );
    try {
      _db.execute('BEGIN;');
      try {
        for (final r in records) {
          stmt.execute([
            r.text,
            dim,
            _blobOf(r.embedding),
            _encodeMeta(r.metadata),
          ]);
        }
        _db.execute('COMMIT;');
      } catch (_) {
        _db.execute('ROLLBACK;');
        rethrow;
      }
    } finally {
      stmt.dispose();
    }
  }

  /// Returns the top [topK] matches for [query] by descending cosine
  /// similarity.
  ///
  /// The query is L2-normalized defensively so scores remain true cosine
  /// values even if the caller passes an unnormalized vector. An optional
  /// [where] predicate filters candidates by their decoded metadata before
  /// scoring; rows for which it returns `false` are skipped.
  ///
  /// Throws [ArgumentError] if `query.length != dim`.
  List<EmbeddingMatch> search(
    Float32List query, {
    int topK = 5,
    bool Function(Map<String, Object?>? meta)? where,
  }) {
    _checkDim(query);
    if (topK <= 0) return const [];

    final q = _normalized(query);
    final rows = _db.select(
      'SELECT id, text, vec, meta FROM embeddings;',
    );

    final matches = <EmbeddingMatch>[];
    for (final row in rows) {
      final meta = _decodeMeta(row['meta'] as String?);
      if (where != null && !where(meta)) continue;

      final blob = row['vec'] as Uint8List;
      final vec = _vectorOf(blob);
      final score = _dot(q, vec);
      matches.add(EmbeddingMatch(
        id: row['id'] as int,
        text: row['text'] as String,
        score: score,
        metadata: meta,
      ));
    }

    matches.sort((a, b) => b.score.compareTo(a.score));
    if (matches.length > topK) {
      return matches.sublist(0, topK);
    }
    return matches;
  }

  /// Returns the record stored at [id], or `null` if there is no such row.
  EmbeddingRecord? get(int id) {
    final rows = _db.select(
      'SELECT text, vec, meta FROM embeddings WHERE id = ?;',
      [id],
    );
    if (rows.isEmpty) return null;
    final row = rows.first;
    return EmbeddingRecord(
      text: row['text'] as String,
      embedding: _vectorOf(row['vec'] as Uint8List),
      metadata: _decodeMeta(row['meta'] as String?),
    );
  }

  /// Deletes the row with the given [id]. A no-op if it does not exist.
  void delete(int id) {
    _db.execute('DELETE FROM embeddings WHERE id = ?;', [id]);
  }

  /// Removes every embedding, leaving the store's dimension unchanged.
  void clear() {
    _db.execute('DELETE FROM embeddings;');
  }

  /// Closes the underlying database. The store must not be used afterwards.
  void close() {
    _db.dispose();
  }

  void _checkDim(Float32List v) {
    if (v.length != dim) {
      throw ArgumentError.value(
        v.length,
        'embedding.length',
        'must equal store dimension ($dim)',
      );
    }
  }

  /// Views a [Float32List] as its little-endian byte BLOB. On the little-endian
  /// ARM/x64 targets we run on, the in-memory layout is already the storage
  /// layout, so this is a zero-copy view.
  static Uint8List _blobOf(Float32List v) =>
      Uint8List.view(v.buffer, v.offsetInBytes, v.lengthInBytes);

  /// Reconstructs a [Float32List] from a stored BLOB. SQLite returns a fresh,
  /// aligned [Uint8List] whose backing buffer we can view directly.
  static Float32List _vectorOf(Uint8List blob) => Float32List.view(
        blob.buffer,
        blob.offsetInBytes,
        blob.lengthInBytes ~/ Float32List.bytesPerElement,
      );

  static String? _encodeMeta(Map<String, Object?>? meta) =>
      meta == null ? null : jsonEncode(meta);

  static Map<String, Object?>? _decodeMeta(String? json) {
    if (json == null) return null;
    final decoded = jsonDecode(json);
    return (decoded as Map).cast<String, Object?>();
  }

  static double _dot(Float32List a, Float32List b) {
    var sum = 0.0;
    for (var i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  /// Returns an L2-normalized copy of [v]. A zero vector is returned unchanged.
  static Float32List _normalized(Float32List v) {
    var norm = 0.0;
    for (var i = 0; i < v.length; i++) {
      norm += v[i] * v[i];
    }
    norm = norm > 0 ? 1.0 / math.sqrt(norm) : 0.0;
    if (norm == 0.0) return Float32List.fromList(v);
    final out = Float32List(v.length);
    for (var i = 0; i < v.length; i++) {
      out[i] = v[i] * norm;
    }
    return out;
  }
}
