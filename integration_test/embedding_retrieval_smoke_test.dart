import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:vaultiq/src/model_catalog.dart';
import 'package:vaultiq/src/models.dart';
import 'package:vaultiq/src/services/local_ai_service.dart';
import 'package:vaultiq/src/services/retrieval_ranker.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('real embedder supports hybrid retrieval', (tester) async {
    final ai = LocalAiService();
    await ai.initialize();

    final supportedAbis = await ai.supportedAbis();
    // ignore: avoid_print
    print('Android supported ABIs: ${supportedAbis.join(', ')}');
    if (!await ai.supportsLocalEmbedder()) {
      // ignore: avoid_print
      print(
        'Skipping real embedding generation on unsupported ABI: ${supportedAbis.join(', ')}',
      );
      await ai.dispose();
      return;
    }

    if (!ai.hasActiveEmbedder) {
      await ai.installEmbedderFromNetwork(
        model: DownloadableEmbeddingModel.gecko256,
        onProgress: (modelProgress, tokenizerProgress) {
          // ignore: avoid_print
          print(
            'Embedding download progress: model $modelProgress%, tokenizer $tokenizerProgress%',
          );
        },
      );
    }

    final texts = <String>[
      'Cafeteria lunch credits are a workplace meal benefit. Employees can use lunch credits for food on weekdays.',
      'Android Gradle builds require matching SDK platforms and accepted Android SDK licenses.',
      'Travel reimbursements require receipts and manager approval before finance can process payment.',
    ];

    final documentEmbeddings = await ai.embedDocuments(texts);
    final queryEmbedding = await ai.embedQuery(
      'Where are meal benefits and lunch credits described?',
    );

    expect(documentEmbeddings, hasLength(texts.length));
    expect(queryEmbedding.length, greaterThan(100));
    expect(
      documentEmbeddings.every(
        (embedding) => embedding.length == queryEmbedding.length,
      ),
      isTrue,
    );

    const ranker = RetrievalRanker(maxChunksPerDocument: 10);
    final hits = ranker.rank(
      query: 'meal benefits lunch credits',
      candidates: [
        for (var i = 0; i < texts.length; i++)
          _candidate(
            id: 'doc-$i',
            name: ['benefits.md', 'android.md', 'finance.md'][i],
            content: texts[i],
            embedding: documentEmbeddings[i],
          ),
      ],
      queryEmbedding: queryEmbedding,
      topK: 3,
    );

    expect(hits, isNotEmpty);
    expect(hits.first.document.name, 'benefits.md');
    expect(hits.first.keywordScore, greaterThan(0));
    expect(hits.first.vectorScore, greaterThan(0));

    await ai.dispose();
  });
}

RetrievalCandidate _candidate({
  required String id,
  required String name,
  required String content,
  required List<double> embedding,
}) {
  return RetrievalCandidate(
    document: IndexedDocument(
      id: id,
      name: name,
      sourcePath: name,
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
      embeddingModel: DownloadableEmbeddingModel.gecko256.id,
    ),
  );
}
