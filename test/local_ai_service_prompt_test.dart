import 'package:flutter_test/flutter_test.dart';
import 'package:vaultiq/src/model_catalog.dart';
import 'package:vaultiq/src/models.dart';
import 'package:vaultiq/src/services/local_ai_service.dart';

void main() {
  test('prompt builder respects model context limits', () {
    final service = LocalAiService();
    final hits = List.generate(
      5,
      (index) => _hit(
        documentName: 'document_$index.md',
        content: List.filled(
          80,
          'This is a long retrieved sentence from document $index.',
        ).join(' '),
      ),
    );

    final prompt = service.buildPromptForTesting(
      question: List.filled(40, 'What does the document say?').join(' '),
      hits: hits,
      config: const LocalInferenceConfig(
        maxTokens: 1024,
        reservedOutputTokens: 320,
        maxPromptHits: 2,
        maxContextChars: 250,
        maxExcerptChars: 120,
        maxQuestionChars: 90,
      ),
    );

    expect(
      RegExp(r'^Excerpt \d+', multiLine: true).allMatches(prompt),
      hasLength(2),
    );
    expect(prompt, contains('Source file: document_0.md'));
    expect(prompt, isNot(contains('keyword')));
    expect(prompt, isNot(contains('semantic')));
    expect(prompt.length, lessThanOrEqualTo(840));
    expect(prompt, contains('Question: What does the document say?'));
    expect(prompt, contains('...'));
  });

  test('prompt builder uses expanded retrieval context when available', () {
    final service = LocalAiService();
    final prompt = service.buildPromptForTesting(
      question: 'What is the approval step?',
      hits: [
        _hit(
          documentName: 'handbook.md',
          content: 'The selected chunk only says submit the request.',
          context:
              'First collect manager approval. Then submit the request. Finally archive the receipt.',
        ),
      ],
      config: const LocalInferenceConfig(
        maxTokens: 1024,
        reservedOutputTokens: 320,
        maxPromptHits: 1,
        maxContextChars: 220,
        maxExcerptChars: 220,
      ),
    );

    expect(prompt, contains('manager approval'));
    expect(prompt, contains('archive the receipt'));
  });

  test('prompt builder allows broad synthesis from relevant excerpts', () {
    final service = LocalAiService();
    final prompt = service.buildPromptForTesting(
      question: 'tell me about pavan',
      hits: [
        _hit(
          documentName: 'Pavan_Gangadhar__FDE.pdf',
          content:
              'PAVAN GANGADHAR Forward Deployed Engineer. Professional summary: AI Agent Developer and Cloud Engineer building autonomous multi-agent systems and LLM-powered applications.',
        ),
      ],
      config: const LocalInferenceConfig(
        maxTokens: 1024,
        reservedOutputTokens: 320,
        maxPromptHits: 1,
        maxContextChars: 260,
        maxExcerptChars: 260,
      ),
    );

    expect(prompt, contains('Broad: synthesize relevant facts'));
    expect(prompt, contains('exact sentence match not required'));
    expect(prompt, contains('If no relevant facts'));
    expect(prompt, isNot(contains('If the exact answer is missing')));
    expect(prompt, contains('PAVAN GANGADHAR'));
    expect(prompt, contains('Forward Deployed Engineer'));
    expect(prompt, contains('Question: tell me about pavan'));
  });

  test('prompt builder focuses expanded context on requested identifier', () {
    final service = LocalAiService();
    final prompt = service.buildPromptForTesting(
      question: 'For CASE-002, answer with only the owner team.',
      hits: [
        _hit(
          documentName: 'eval.md',
          content: 'Record ID: CASE-002. Owner team: Quartz Finance.',
          context: '''
## CASE-001: Beacon Ledger
Record ID: CASE-001.
Owner team: Blue Harbor.

## CASE-002: Atlas Relay
Record ID: CASE-002.
Owner team: Quartz Finance.
Deadline: 2026-04-10.
''',
        ),
      ],
      config: const LocalInferenceConfig(
        maxTokens: 1024,
        reservedOutputTokens: 320,
        maxPromptHits: 1,
        maxContextChars: 220,
        maxExcerptChars: 220,
      ),
    );

    expect(prompt, contains('CASE-002'));
    expect(prompt, contains('Owner team: Quartz Finance'));
    expect(prompt, isNot(contains('Blue Harbor')));
  });

  test('extractive answer copies exact fields for identifier questions', () {
    final service = LocalAiService();
    final answer = service.extractiveAnswerForTesting(
      question:
          'For CASE-002, provide the approval path and primary region in one concise sentence.',
      hits: [
        _hit(
          documentName: 'eval.md',
          content: 'Record ID: CASE-002.',
          context: '''
## CASE-002: Atlas Relay
Record ID: CASE-002.
Owner team: Quartz Finance.
Approval path: finance director approval.
Primary region: Seattle.
Deadline: 2026-04-10.
''',
        ),
      ],
    );

    expect(answer, contains('finance director approval'));
    expect(answer, contains('Seattle'));
    expect(answer, contains('[eval.md]'));
  });

  test('extractive answer skips partial identifier overlaps', () {
    final service = LocalAiService();
    final answer = service.extractiveAnswerForTesting(
      question: 'What budget cap is listed for CASE-025?',
      hits: [
        _hit(
          documentName: 'eval.md',
          content:
              'Operational note: when answering questions about CASE-025, use exact values.',
        ),
        _hit(
          documentName: 'eval.md',
          content: '''
## CASE-025: Beacon Ledger
Record ID: CASE-025.
Owner team: Blue Harbor.
Budget cap: \$2375.
Deadline: 2026-08-22.
''',
        ),
      ],
    );

    expect(answer, contains('\$2375'));
    expect(answer, contains('[eval.md]'));
  });

  test('extractive answer reports missing identifiers', () {
    final service = LocalAiService();
    final answer = service.extractiveAnswerForTesting(
      question:
          'What is the owner team for CASE-090? If the local file does not say, state that it is missing.',
      hits: [
        _hit(
          documentName: 'eval.md',
          content: 'Record ID: CASE-001. Owner team: Blue Harbor.',
        ),
      ],
    );

    expect(answer, contains('CASE-090 is missing'));
    expect(answer, contains('[eval.md]'));
  });
}

SearchHit _hit({
  required String documentName,
  required String content,
  String? context,
}) {
  return SearchHit(
    document: IndexedDocument(
      id: documentName,
      name: documentName,
      sourcePath: documentName,
      sha256: documentName,
      bytes: content.length,
      indexedAt: DateTime(2026),
      chunkCount: 1,
    ),
    chunk: DocumentChunk(
      id: '$documentName:0',
      documentId: documentName,
      index: 0,
      content: content,
    ),
    score: 1,
    context: context,
  );
}
