import 'package:flutter_test/flutter_test.dart';
import 'package:vaultiq/src/services/text_chunker.dart';

void main() {
  test('chunk keeps short text as one chunk', () {
    const chunker = TextChunker(chunkSize: 20, overlap: 4);

    expect(chunker.chunk('short note'), ['short note']);
  });

  test('chunk splits long text with overlap', () {
    const chunker = TextChunker(chunkSize: 10, overlap: 3);

    final chunks = chunker.chunk('0123456789abcdefghij');

    expect(chunks.length, greaterThan(1));
    expect(chunks.first, '0123456789');
    expect(chunks[1].startsWith('789'), isTrue);
  });
}
