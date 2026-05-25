import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

import '../models.dart';
import 'app_logger.dart';
import 'retrieval_ranker.dart';
import 'text_chunker.dart';

class DocumentStore {
  DocumentStore({
    this.chunker = const TextChunker(),
    this.ranker = const RetrievalRanker(),
  });

  final TextChunker chunker;
  final RetrievalRanker ranker;
  Database? _db;
  bool _ftsAvailable = true;
  bool _ftsProvidesRank = true;

  Future<void> open() async {
    if (_db != null) return;
    final stopwatch = Stopwatch()..start();
    final directory = await getApplicationDocumentsDirectory();
    final path = p.join(directory.path, 'local_doc_qa.db');
    AppLogger.info('store.open.start', {'path': path});
    _db = await openDatabase(
      path,
      version: 3,
      onCreate: (db, version) async {
        AppLogger.info('store.database.create', {'version': version});
        await _createTables(db);
      },
      onUpgrade: (db, oldVersion, newVersion) async {
        AppLogger.info('store.database.upgrade', {
          'oldVersion': oldVersion,
          'newVersion': newVersion,
        });
        if (oldVersion < 2) {
          await db.execute('ALTER TABLE chunks ADD COLUMN embedding BLOB');
          await db.execute(
            'ALTER TABLE chunks ADD COLUMN embedding_model TEXT',
          );
        }
        if (oldVersion < 3) {
          await _createFtsIndex(db);
          await _rebuildFtsIndex(db);
        }
      },
    );
    await _ensureFtsIndex();
    AppLogger.info('store.open.done', {
      'durationMs': stopwatch.elapsedMilliseconds,
      'ftsAvailable': _ftsAvailable,
      'ftsProvidesRank': _ftsProvidesRank,
    });
  }

  Future<void> _createTables(Database db) async {
    await db.execute('''
          CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            indexed_at INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL
          )
        ''');
    await db.execute('''
          CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            embedding_model TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
          )
        ''');
    await db.execute(
      'CREATE INDEX chunks_document_id_index ON chunks(document_id)',
    );
    await _createFtsIndex(db);
  }

  Future<void> _createFtsIndex(Database db) async {
    final existing = await db.query(
      'sqlite_master',
      columns: ['sql'],
      where: 'name = ?',
      whereArgs: ['chunks_fts'],
      limit: 1,
    );
    if (existing.isNotEmpty) {
      final sql = (existing.first['sql'] as String? ?? '').toLowerCase();
      _ftsAvailable = true;
      _ftsProvidesRank = sql.contains('using fts5');
      AppLogger.info('store.fts.detected', {
        'ftsAvailable': _ftsAvailable,
        'ftsProvidesRank': _ftsProvidesRank,
      });
      return;
    }

    if (!await _sqliteSupportsFts5(db)) {
      AppLogger.warn('store.fts5.unavailable');
      await _createFts4Index(db);
      return;
    }

    try {
      await db.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
          chunk_id UNINDEXED,
          document_id UNINDEXED,
          title,
          content,
          tokenize = 'unicode61 remove_diacritics 2'
        )
      ''');
      _ftsAvailable = true;
      _ftsProvidesRank = true;
      AppLogger.info('store.fts5.created');
    } on DatabaseException {
      AppLogger.warn('store.fts5.create_failed_using_fts4');
      await _createFts4Index(db);
    }
  }

  Future<bool> _sqliteSupportsFts5(Database db) async {
    try {
      final rows = await db.rawQuery('PRAGMA compile_options');
      return rows.any(
        (row) => row.values.any(
          (value) => value.toString().toUpperCase() == 'ENABLE_FTS5',
        ),
      );
    } on DatabaseException {
      return false;
    }
  }

  Future<void> _createFts4Index(Database db) async {
    try {
      await db.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts4(
          chunk_id,
          document_id,
          title,
          content,
          notindexed=chunk_id,
          notindexed=document_id,
          tokenize=unicode61
        )
      ''');
      _ftsAvailable = true;
      _ftsProvidesRank = false;
      AppLogger.info('store.fts4.created');
    } on DatabaseException {
      _ftsAvailable = false;
      _ftsProvidesRank = false;
      AppLogger.warn('store.fts.unavailable');
    }
  }

  Future<void> _ensureFtsIndex() async {
    final db = _requireDb();
    await _createFtsIndex(db);
    if (!_ftsAvailable) return;

    final chunkCount =
        Sqflite.firstIntValue(
          await db.rawQuery('SELECT COUNT(*) FROM chunks'),
        ) ??
        0;
    if (chunkCount == 0) return;

    try {
      final ftsCount =
          Sqflite.firstIntValue(
            await db.rawQuery('SELECT COUNT(*) FROM chunks_fts'),
          ) ??
          0;
      if (ftsCount == 0) {
        AppLogger.warn('store.fts.rebuild_needed', {
          'chunkCount': chunkCount,
          'ftsCount': ftsCount,
        });
        await _rebuildFtsIndex(db);
      }
    } on DatabaseException {
      _ftsAvailable = false;
      _ftsProvidesRank = false;
      AppLogger.warn('store.fts.disabled_after_check');
    }
  }

  Future<void> _rebuildFtsIndex(Database db) async {
    if (!_ftsAvailable) return;
    final stopwatch = Stopwatch()..start();
    AppLogger.info('store.fts.rebuild.start');
    await db.delete('chunks_fts');
    await db.execute('''
      INSERT INTO chunks_fts(chunk_id, document_id, title, content)
      SELECT chunks.id, chunks.document_id, documents.name, chunks.content
      FROM chunks
      INNER JOIN documents ON documents.id = chunks.document_id
    ''');
    AppLogger.info('store.fts.rebuild.done', {
      'durationMs': stopwatch.elapsedMilliseconds,
    });
  }

  Future<void> upsertDocument(
    ExtractedDocument document, {
    Future<List<List<double>>?> Function(List<String> chunks)? embedChunks,
    String? embeddingModel,
  }) async {
    final db = _requireDb();
    final stopwatch = Stopwatch()..start();
    AppLogger.info('store.upsert.start', {
      'document': document.name,
      'sourcePath': document.sourcePath,
      'bytes': document.bytes,
      'textChars': document.text.length,
      'textPreview': AppLogger.preview(document.text),
      'embeddingRequested': embedChunks != null,
      'embeddingModel': embeddingModel,
    });
    final hash = sha256.convert(utf8.encode(document.text)).toString();
    final id = hash;
    final chunks = chunker.chunk(document.text);
    AppLogger.info('store.upsert.chunking.done', {
      'document': document.name,
      'documentId': _shortId(id),
      'chunkCount': chunks.length,
      'chunkSize': chunker.chunkSize,
      'overlap': chunker.overlap,
      'chunkStats': AppLogger.textStats(chunks),
      'sampleChunks': _chunkSamples(chunks),
    });
    final now = DateTime.now().millisecondsSinceEpoch;
    final embeddings = await _embedForIndexing(chunks, embedChunks);
    AppLogger.info('store.upsert.embedding.done', {
      'document': document.name,
      'embeddingCount': embeddings?.length ?? 0,
      'embeddingDimension': embeddings == null || embeddings.isEmpty
          ? 0
          : embeddings.first.length,
    });

    await db.transaction((txn) async {
      await txn.delete('chunks', where: 'document_id = ?', whereArgs: [id]);
      await _deleteFtsRows(txn, documentId: id);
      await txn.insert('documents', {
        'id': id,
        'name': document.name,
        'source_path': document.sourcePath,
        'sha256': hash,
        'bytes': document.bytes,
        'indexed_at': now,
        'chunk_count': chunks.length,
      }, conflictAlgorithm: ConflictAlgorithm.replace);

      final batch = txn.batch();
      for (var i = 0; i < chunks.length; i++) {
        batch.insert('chunks', {
          'id': '$id:$i',
          'document_id': id,
          'chunk_index': i,
          'content': chunks[i],
          'embedding': embeddings == null
              ? null
              : _encodeEmbedding(embeddings[i]),
          'embedding_model': embeddings == null ? null : embeddingModel,
        });
        if (_ftsAvailable) {
          batch.insert('chunks_fts', {
            'chunk_id': '$id:$i',
            'document_id': id,
            'title': document.name,
            'content': chunks[i],
          });
        }
      }
      await batch.commit(noResult: true);
    });
    AppLogger.info('store.upsert.done', {
      'document': document.name,
      'documentId': _shortId(id),
      'chunkCount': chunks.length,
      'hasEmbeddings': embeddings != null,
      'durationMs': stopwatch.elapsedMilliseconds,
    });
  }

  Future<List<IndexedDocument>> listDocuments() async {
    final rows = await _requireDb().query(
      'documents',
      orderBy: 'indexed_at DESC',
    );
    return rows.map(IndexedDocument.fromMap).toList(growable: false);
  }

  Future<IndexStats> stats() async {
    final db = _requireDb();
    final docCount =
        Sqflite.firstIntValue(
          await db.rawQuery('SELECT COUNT(*) FROM documents'),
        ) ??
        0;
    final chunkCount =
        Sqflite.firstIntValue(
          await db.rawQuery('SELECT COUNT(*) FROM chunks'),
        ) ??
        0;
    final embeddedChunkCount =
        Sqflite.firstIntValue(
          await db.rawQuery(
            'SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL',
          ),
        ) ??
        0;
    final totalBytes =
        Sqflite.firstIntValue(
          await db.rawQuery('SELECT COALESCE(SUM(bytes), 0) FROM documents'),
        ) ??
        0;
    return IndexStats(
      documentCount: docCount,
      chunkCount: chunkCount,
      embeddedChunkCount: embeddedChunkCount,
      totalBytes: totalBytes,
    );
  }

  Future<List<SearchHit>> search(
    String query, {
    int topK = 5,
    List<double>? queryEmbedding,
    int contextWindow = 1,
  }) async {
    final db = _requireDb();
    final stopwatch = Stopwatch()..start();
    final mode = queryEmbedding == null ? 'keyword' : 'hybrid_vector';
    AppLogger.info('retrieval.search.start', {
      'mode': mode,
      'queryChars': query.length,
      'queryPreview': AppLogger.preview(query),
      'topK': topK,
      'contextWindow': contextWindow,
      'queryEmbeddingDim': queryEmbedding?.length ?? 0,
      'ftsAvailable': _ftsAvailable,
      'ftsProvidesRank': _ftsProvidesRank,
    });
    final keywordLimit = math.max(topK * ranker.candidateMultiplier * 4, 40);
    final keywordCandidates = queryEmbedding == null
        ? await _keywordCandidates(query, limit: keywordLimit)
        : const _KeywordCandidates.empty();
    AppLogger.info('retrieval.search.keyword_candidates.done', {
      'mode': mode,
      'keywordLimit': keywordLimit,
      'keywordCandidateCount': keywordCandidates.chunkIds.length,
      'sampleChunkIds': keywordCandidates.chunkIds.take(12).toList(),
    });
    final rows = queryEmbedding == null
        ? _ftsAvailable
              ? await _loadKeywordCandidateRows(keywordCandidates.chunkIds)
              : await db.query('chunks')
        : await db.query('chunks');
    AppLogger.info('retrieval.search.rows_loaded', {
      'mode': mode,
      'rowCount': rows.length,
    });
    final documentIds = rows
        .map((row) => row['document_id'] as String)
        .toSet()
        .toList(growable: false);
    final docs = await _loadDocumentsByIds(documentIds);
    final candidates = <RetrievalCandidate>[];
    for (final row in rows) {
      final chunk = DocumentChunk.fromMap(row);
      final document = docs[chunk.documentId];
      if (document == null) continue;
      candidates.add(
        RetrievalCandidate(
          document: document,
          chunk: chunk,
          keywordScore: keywordCandidates.scores[chunk.id],
        ),
      );
    }

    final hits = ranker.rank(
      query: query,
      candidates: candidates,
      queryEmbedding: queryEmbedding,
      topK: topK,
    );
    final expandedHits = await _attachContextWindows(
      hits,
      contextWindow: contextWindow,
    );
    AppLogger.info('retrieval.search.done', {
      'mode': mode,
      'candidateCount': candidates.length,
      'hitCount': expandedHits.length,
      'durationMs': stopwatch.elapsedMilliseconds,
      'hits': _hitSummaries(expandedHits),
    });
    return expandedHits;
  }

  Future<int> countChunksMissingEmbeddings() async {
    return Sqflite.firstIntValue(
          await _requireDb().rawQuery(
            'SELECT COUNT(*) FROM chunks WHERE embedding IS NULL',
          ),
        ) ??
        0;
  }

  Future<int> backfillMissingEmbeddings({
    required Future<List<List<double>>?> Function(List<String> chunks)
    embedChunks,
    required String embeddingModel,
    int batchSize = 8,
    void Function(int completed, int total)? onProgress,
  }) async {
    final db = _requireDb();
    final total = await countChunksMissingEmbeddings();
    AppLogger.info('store.embedding_backfill.start', {
      'totalMissing': total,
      'batchSize': batchSize,
      'embeddingModel': embeddingModel,
    });
    if (total == 0) {
      onProgress?.call(0, 0);
      AppLogger.info('store.embedding_backfill.done', {
        'completed': 0,
        'total': 0,
      });
      return 0;
    }

    var completed = 0;
    while (true) {
      final rows = await db.query(
        'chunks',
        columns: ['id', 'content'],
        where: 'embedding IS NULL',
        orderBy: 'document_id ASC, chunk_index ASC',
        limit: batchSize,
      );
      if (rows.isEmpty) break;

      final contents = rows
          .map((row) => row['content'] as String)
          .toList(growable: false);
      AppLogger.info('store.embedding_backfill.batch.start', {
        'completed': completed,
        'batchRows': rows.length,
        'chunkIds': rows.map((row) => row['id']).take(8).toList(),
        'chunkStats': AppLogger.textStats(contents),
        'sampleChunks': _chunkSamples(contents),
      });
      final embeddings = await embedChunks(contents);
      if (embeddings == null || embeddings.length != rows.length) {
        AppLogger.warn('store.embedding_backfill.batch.invalid_embeddings', {
          'expected': rows.length,
          'actual': embeddings?.length ?? 0,
        });
        break;
      }

      await db.transaction((txn) async {
        final batch = txn.batch();
        for (var i = 0; i < rows.length; i++) {
          batch.update(
            'chunks',
            {
              'embedding': _encodeEmbedding(embeddings[i]),
              'embedding_model': embeddingModel,
            },
            where: 'id = ?',
            whereArgs: [rows[i]['id']],
          );
        }
        await batch.commit(noResult: true);
      });

      completed += rows.length;
      onProgress?.call(completed, total);
      AppLogger.info('store.embedding_backfill.batch.done', {
        'completed': completed,
        'total': total,
        'embeddingDimension': embeddings.isEmpty ? 0 : embeddings.first.length,
      });
    }

    AppLogger.info('store.embedding_backfill.done', {
      'completed': completed,
      'total': total,
    });
    return completed;
  }

  Future<void> clear() async {
    final db = _requireDb();
    AppLogger.warn('store.clear.start');
    await db.transaction((txn) async {
      await _deleteAllFtsRows(txn);
      await txn.delete('chunks');
      await txn.delete('documents');
    });
    AppLogger.warn('store.clear.done');
  }

  Future<void> close() async {
    AppLogger.info('store.close.start');
    await _db?.close();
    _db = null;
    AppLogger.info('store.close.done');
  }

  Database _requireDb() {
    final db = _db;
    if (db == null) {
      throw StateError('DocumentStore.open must be called before use.');
    }
    return db;
  }

  Future<List<List<double>>?> _embedForIndexing(
    List<String> chunks,
    Future<List<List<double>>?> Function(List<String> chunks)? embedChunks,
  ) async {
    if (embedChunks == null || chunks.isEmpty) return null;
    AppLogger.info('store.embed_for_indexing.start', {
      'chunkCount': chunks.length,
      'chunkStats': AppLogger.textStats(chunks),
    });
    final embeddings = await embedChunks(chunks);
    if (embeddings == null || embeddings.length != chunks.length) {
      AppLogger.warn('store.embed_for_indexing.invalid_result', {
        'expected': chunks.length,
        'actual': embeddings?.length ?? 0,
      });
      return null;
    }
    AppLogger.info('store.embed_for_indexing.done', {
      'embeddingCount': embeddings.length,
      'embeddingDimension': embeddings.isEmpty ? 0 : embeddings.first.length,
    });
    return embeddings;
  }

  Uint8List _encodeEmbedding(List<double> embedding) {
    final data = ByteData(embedding.length * 4);
    for (var i = 0; i < embedding.length; i++) {
      data.setFloat32(i * 4, embedding[i], Endian.little);
    }
    return data.buffer.asUint8List();
  }

  Future<List<SearchHit>> _attachContextWindows(
    List<SearchHit> hits, {
    required int contextWindow,
  }) async {
    if (contextWindow <= 0 || hits.isEmpty) return hits;

    final expanded = <SearchHit>[];
    for (final hit in hits) {
      final start = math.max(0, hit.chunk.index - contextWindow);
      final end = hit.chunk.index + contextWindow;
      final rows = await _requireDb().query(
        'chunks',
        columns: ['chunk_index', 'content'],
        where: 'document_id = ? AND chunk_index BETWEEN ? AND ?',
        whereArgs: [hit.document.id, start, end],
        orderBy: 'chunk_index ASC',
      );
      final context = rows
          .map((row) => (row['content'] as String).trim())
          .where((content) => content.isNotEmpty)
          .join('\n\n');
      AppLogger.info('retrieval.context_window.attached', {
        'document': hit.document.name,
        'chunkIndex': hit.chunk.index,
        'windowStart': start,
        'windowEnd': end,
        'rowCount': rows.length,
        'contextChars': context.length,
      });
      expanded.add(
        SearchHit(
          document: hit.document,
          chunk: hit.chunk,
          score: hit.score,
          keywordScore: hit.keywordScore,
          vectorScore: hit.vectorScore,
          context: context.isEmpty ? null : context,
        ),
      );
    }

    return expanded;
  }

  Future<Map<String, IndexedDocument>> _loadDocumentsByIds(
    List<String> ids,
  ) async {
    if (ids.isEmpty) return const {};
    if (ids.length > 900) {
      final rows = await _requireDb().query('documents');
      return {
        for (final row in rows)
          row['id'] as String: IndexedDocument.fromMap(row),
      };
    }

    final rows = await _requireDb().query(
      'documents',
      where: 'id IN (${List.filled(ids.length, '?').join(',')})',
      whereArgs: ids,
    );
    return {
      for (final row in rows) row['id'] as String: IndexedDocument.fromMap(row),
    };
  }

  Future<List<Map<String, Object?>>> _loadKeywordCandidateRows(
    List<String> chunkIds,
  ) async {
    if (chunkIds.isEmpty) return const [];
    return _requireDb().query(
      'chunks',
      where: 'id IN (${List.filled(chunkIds.length, '?').join(',')})',
      whereArgs: chunkIds,
    );
  }

  Future<_KeywordCandidates> _keywordCandidates(
    String query, {
    required int limit,
  }) async {
    if (!_ftsAvailable || limit <= 0) {
      return const _KeywordCandidates.empty();
    }

    final terms = _ftsTerms(query);
    if (terms.isEmpty) {
      AppLogger.warn('retrieval.fts.terms_empty', {
        'queryPreview': AppLogger.preview(query),
      });
      return const _KeywordCandidates.empty();
    }

    final exactQuery = terms.map((term) => '$term*').join(' ');
    AppLogger.info('retrieval.fts.query.start', {
      'terms': terms,
      'exactQuery': exactQuery,
      'limit': limit,
      'ftsAvailable': _ftsAvailable,
      'ftsProvidesRank': _ftsProvidesRank,
    });
    var rows = await _queryFts(exactQuery, limit: limit);
    if (rows.isEmpty && terms.length > 1) {
      final broadQuery = terms.map((term) => '$term*').join(' OR ');
      AppLogger.info('retrieval.fts.query.broad_retry', {
        'broadQuery': broadQuery,
      });
      rows = await _queryFts(broadQuery, limit: limit);
    }

    final chunkIds = rows
        .map((row) => row['chunk_id'] as String)
        .toList(growable: false);
    AppLogger.info('retrieval.fts.query.done', {
      'terms': terms,
      'rowCount': rows.length,
      'chunkIds': chunkIds.take(12).toList(),
    });
    return _KeywordCandidates(chunkIds: chunkIds, scores: const {});
  }

  Future<List<Map<String, Object?>>> _queryFts(
    String matchQuery, {
    required int limit,
  }) async {
    try {
      final sql = _ftsProvidesRank
          ? '''
        SELECT chunk_id
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY bm25(chunks_fts)
        LIMIT ?
        '''
          : '''
        SELECT chunk_id
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        LIMIT ?
        ''';
      return await _requireDb().rawQuery(sql, [matchQuery, limit]);
    } on DatabaseException catch (e, st) {
      AppLogger.error('retrieval.fts.query.error', e, st, {
        'matchQuery': matchQuery,
        'limit': limit,
      });
      _ftsAvailable = false;
      return const [];
    }
  }

  Future<void> _deleteFtsRows(
    DatabaseExecutor executor, {
    required String documentId,
  }) async {
    if (!_ftsAvailable) return;
    await executor.delete(
      'chunks_fts',
      where: 'document_id = ?',
      whereArgs: [documentId],
    );
  }

  Future<void> _deleteAllFtsRows(DatabaseExecutor executor) async {
    if (!_ftsAvailable) return;
    await executor.delete('chunks_fts');
  }

  List<Map<String, Object?>> _chunkSamples(List<String> chunks) {
    return [
      for (var i = 0; i < chunks.length && i < 3; i++)
        {
          'index': i,
          'chars': chunks[i].length,
          'preview': AppLogger.preview(chunks[i], 140),
        },
    ];
  }

  List<Map<String, Object?>> _hitSummaries(List<SearchHit> hits) {
    return [
      for (var i = 0; i < hits.length && i < 8; i++) _hitSummary(hits[i], i),
    ];
  }

  Map<String, Object?> _hitSummary(SearchHit hit, int index) {
    return {
      'rank': index + 1,
      'document': hit.document.name,
      'documentId': _shortId(hit.document.id),
      'chunkId': _shortId(hit.chunk.id),
      'chunkIndex': hit.chunk.index,
      'score': AppLogger.score(hit.score),
      'keywordScore': AppLogger.score(hit.keywordScore),
      'vectorScore': AppLogger.score(hit.vectorScore),
      'chunkChars': hit.chunk.content.length,
      'contextChars': hit.contextText.length,
      'preview': AppLogger.preview(hit.contextText, 140),
    };
  }

  String _shortId(String value) {
    return value.length <= 12 ? value : value.substring(0, 12);
  }
}

class _KeywordCandidates {
  const _KeywordCandidates({required this.chunkIds, required this.scores});

  const _KeywordCandidates.empty() : chunkIds = const [], scores = const {};

  final List<String> chunkIds;
  final Map<String, double> scores;
}

List<String> _ftsTerms(String value) {
  return RegExp(r'[a-zA-Z0-9_]{2,}')
      .allMatches(value.toLowerCase())
      .map((match) => match.group(0)!)
      .where((term) => !_ftsStopWords.contains(term))
      .toList(growable: false);
}

const Set<String> _ftsStopWords = {
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'for',
  'from',
  'how',
  'in',
  'is',
  'it',
  'of',
  'on',
  'or',
  'that',
  'the',
  'this',
  'to',
  'was',
  'what',
  'when',
  'where',
  'which',
  'who',
  'why',
  'with',
};
