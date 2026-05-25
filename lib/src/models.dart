import 'dart:convert';

import 'package:flutter/foundation.dart';

@immutable
class ExtractedDocument {
  const ExtractedDocument({
    required this.name,
    required this.sourcePath,
    required this.bytes,
    required this.text,
  });

  final String name;
  final String sourcePath;
  final int bytes;
  final String text;
}

@immutable
class IndexedDocument {
  const IndexedDocument({
    required this.id,
    required this.name,
    required this.sourcePath,
    required this.sha256,
    required this.bytes,
    required this.indexedAt,
    required this.chunkCount,
  });

  factory IndexedDocument.fromMap(Map<String, Object?> map) {
    return IndexedDocument(
      id: map['id'] as String,
      name: map['name'] as String,
      sourcePath: map['source_path'] as String,
      sha256: map['sha256'] as String,
      bytes: map['bytes'] as int,
      indexedAt: DateTime.fromMillisecondsSinceEpoch(map['indexed_at'] as int),
      chunkCount: map['chunk_count'] as int,
    );
  }

  final String id;
  final String name;
  final String sourcePath;
  final String sha256;
  final int bytes;
  final DateTime indexedAt;
  final int chunkCount;
}

@immutable
class DocumentChunk {
  const DocumentChunk({
    required this.id,
    required this.documentId,
    required this.index,
    required this.content,
    this.embedding,
    this.embeddingModel,
  });

  factory DocumentChunk.fromMap(Map<String, Object?> map) {
    return DocumentChunk(
      id: map['id'] as String,
      documentId: map['document_id'] as String,
      index: map['chunk_index'] as int,
      content: map['content'] as String,
      embedding: _parseEmbedding(map['embedding']),
      embeddingModel: map['embedding_model'] as String?,
    );
  }

  final String id;
  final String documentId;
  final int index;
  final String content;
  final List<double>? embedding;
  final String? embeddingModel;
}

@immutable
class SearchHit {
  const SearchHit({
    required this.document,
    required this.chunk,
    required this.score,
    this.keywordScore = 0,
    this.vectorScore = 0,
    this.context,
  });

  final IndexedDocument document;
  final DocumentChunk chunk;
  final double score;
  final double keywordScore;
  final double vectorScore;
  final String? context;

  String get contextText => context ?? chunk.content;
}

@immutable
class IndexStats {
  const IndexStats({
    required this.documentCount,
    required this.chunkCount,
    required this.embeddedChunkCount,
    required this.totalBytes,
  });

  const IndexStats.empty()
    : documentCount = 0,
      chunkCount = 0,
      embeddedChunkCount = 0,
      totalBytes = 0;

  final int documentCount;
  final int chunkCount;
  final int embeddedChunkCount;
  final int totalBytes;
}

enum ChatRole { user, assistant }

@immutable
class ChatTurn {
  ChatTurn({
    required this.role,
    required this.text,
    this.sources = const [],
    this.pending = false,
  }) : id = UniqueKey().toString();

  ChatTurn.user(String text) : this(role: ChatRole.user, text: text);

  ChatTurn.assistant(
    String text, {
    List<SearchHit> sources = const [],
    bool pending = false,
  }) : this(
         role: ChatRole.assistant,
         text: text,
         sources: sources,
         pending: pending,
       );

  final String id;
  final ChatRole role;
  final String text;
  final List<SearchHit> sources;
  final bool pending;
}

List<double>? _parseEmbedding(Object? rawValue) {
  if (rawValue == null) return null;
  if (rawValue is Uint8List) return _parseEmbeddingBytes(rawValue);
  if (rawValue is List<int>) {
    return _parseEmbeddingBytes(Uint8List.fromList(rawValue));
  }
  if (rawValue is! String || rawValue.isEmpty) return null;

  Object? decoded;
  try {
    decoded = jsonDecode(rawValue);
  } on FormatException {
    return null;
  }

  if (decoded is! List) return null;
  return decoded
      .whereType<num>()
      .map((value) => value.toDouble())
      .toList(growable: false);
}

List<double>? _parseEmbeddingBytes(Uint8List bytes) {
  if (bytes.isEmpty || bytes.lengthInBytes % 4 != 0) return null;
  final data = ByteData.sublistView(bytes);
  return List.generate(
    bytes.lengthInBytes ~/ 4,
    (index) => data.getFloat32(index * 4, Endian.little),
    growable: false,
  );
}
