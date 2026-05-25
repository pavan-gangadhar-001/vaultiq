import 'package:flutter_test/flutter_test.dart';
import 'package:vaultiq/src/models.dart';
import 'package:vaultiq/src/services/retrieval_ranker.dart';

void main() {
  test('BM25 ranks exact keyword matches', () {
    const ranker = RetrievalRanker(maxChunksPerDocument: 10);

    final hits = ranker.rank(
      query: 'refund policy',
      candidates: [
        _candidate(
          documentName: 'policy.md',
          content: 'Refund policy requests must include the original invoice.',
        ),
        _candidate(
          documentName: 'menu.md',
          content: 'Lunch menus rotate weekly in the office cafeteria.',
        ),
      ],
      topK: 2,
    );

    expect(hits, isNotEmpty);
    expect(hits.first.document.name, 'policy.md');
    expect(hits.first.keywordScore, greaterThan(0));
    expect(hits.first.vectorScore, 0);
  });

  test(
    'vector retrieval finds semantic candidates without keyword overlap',
    () {
      const ranker = RetrievalRanker(maxChunksPerDocument: 10);

      final hits = ranker.rank(
        query: 'perks',
        queryEmbedding: const [1, 0],
        candidates: [
          _candidate(
            documentName: 'lunch.md',
            content: 'Cafeteria lunch credits are refreshed every Monday.',
            embedding: const [1, 0],
          ),
          _candidate(
            documentName: 'parking.md',
            content: 'Parking garage passes expire at the end of each month.',
            embedding: const [0, 1],
          ),
        ],
        topK: 2,
      );

      expect(hits, isNotEmpty);
      expect(hits.first.document.name, 'lunch.md');
      expect(hits.first.vectorScore, greaterThan(0));
    },
  );

  test('hybrid ranking keeps keyword and vector evidence', () {
    const ranker = RetrievalRanker(maxChunksPerDocument: 10);

    final hits = ranker.rank(
      query: 'revenue',
      queryEmbedding: const [1, 0],
      candidates: [
        _candidate(
          documentName: 'keyword.md',
          content: 'Quarterly revenue is reported after audit close.',
          embedding: const [0, 1],
        ),
        _candidate(
          documentName: 'semantic.md',
          content: 'Q4 income increased after enterprise renewals.',
          embedding: const [1, 0],
        ),
      ],
      topK: 2,
    );

    expect(
      hits.map((hit) => hit.document.name),
      containsAll(['keyword.md', 'semantic.md']),
    );
    expect(hits.any((hit) => hit.keywordScore > 0), isTrue);
    expect(hits.any((hit) => hit.vectorScore > 0), isTrue);
  });

  test('MMR can prefer a less redundant second hit', () {
    const ranker = RetrievalRanker(maxChunksPerDocument: 10, mmrLambda: 0.45);

    final hits = ranker.rank(
      query: 'refund policy',
      candidates: [
        _candidate(
          documentName: 'a.md',
          content: 'Refund policy requests need the original invoice.',
        ),
        _candidate(
          documentName: 'b.md',
          content: 'Refund policy requests need the original invoice.',
        ),
        _candidate(
          documentName: 'c.md',
          content: 'Refund policy exceptions are reviewed by finance.',
        ),
      ],
      topK: 2,
    );

    expect(hits, hasLength(2));
    final names = hits.map((hit) => hit.document.name).toSet();
    expect(names, contains('c.md'));
    expect(names.contains('a.md') && names.contains('b.md'), isFalse);
  });
}

RetrievalCandidate _candidate({
  required String documentName,
  required String content,
  List<double>? embedding,
}) {
  final id = documentName;
  return RetrievalCandidate(
    document: IndexedDocument(
      id: id,
      name: documentName,
      sourcePath: documentName,
      sha256: id,
      bytes: content.length,
      indexedAt: DateTime(2026),
      chunkCount: 1,
    ),
    chunk: DocumentChunk(
      id: '$id:0',
      documentId: id,
      index: 0,
      content: content,
      embedding: embedding,
      embeddingModel: embedding == null ? null : 'test',
    ),
  );
}
