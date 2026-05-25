import 'package:flutter_test/flutter_test.dart';
import 'package:vaultiq/src/model_catalog.dart';
import 'package:vaultiq/src/services/local_ai_service.dart';

void main() {
  test('gecko input cap leaves room under the 256 token model limit', () {
    expect(DownloadableEmbeddingModel.gecko256.maxSequenceLength, 256);
    expect(DownloadableEmbeddingModel.gecko256.maxInputChars, 220);
  });

  test('embedding input clipping applies the safe cap', () {
    final service = LocalAiService();
    final clipped = service.clipEmbeddingInputForTesting('a' * 500, 900);

    expect(clipped.length, 220);
  });

  test('embedding input clipping respects stricter caller limits', () {
    final service = LocalAiService();
    final clipped = service.clipEmbeddingInputForTesting('a' * 500, 80);

    expect(clipped.length, 80);
  });
}
