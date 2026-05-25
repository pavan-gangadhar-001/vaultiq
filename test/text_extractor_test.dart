import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:vaultiq/src/services/import_service.dart';
import 'package:vaultiq/src/services/text_extractor.dart';

void main() {
  test('extracts readable text from DOCX files', () async {
    final file = File(
      'test_corpus/public_sample_docs/docx_01_public_bicycle_counter_report.docx',
    );

    final extracted = await TextExtractor().extract(
      ImportCandidate(
        name: file.uri.pathSegments.last,
        sourcePath: file.path,
        path: file.path,
      ),
    );

    expect(extracted.text, contains('Public Bicycle Counter Report'));
    expect(extracted.text, contains('BIKE-603'));
    expect(extracted.text, contains('Harbor Trail'));
  });
}
