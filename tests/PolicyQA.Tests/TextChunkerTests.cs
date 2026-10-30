using Xunit;

public class TextChunkerTests
{
    [Fact]
    public void ShortText_ReturnsSingleChunk()
    {
        var chunks = TextChunker.Chunk("one two three", chunkSize: 10, overlap: 2).ToList();

        Assert.Single(chunks);
        Assert.Equal("one two three", chunks[0]);
    }

    [Fact]
    public void LongText_ProducesOverlappingChunks()
    {
        // 25 words, chunks of 10 with overlap 2 → starts at word 1, 9, 17
        var text = string.Join(' ', Enumerable.Range(1, 25).Select(i => $"w{i}"));

        var chunks = TextChunker.Chunk(text, chunkSize: 10, overlap: 2).ToList();

        Assert.Equal(3, chunks.Count);
        Assert.EndsWith("w9 w10", chunks[0]);   // last 2 words of chunk 1...
        Assert.StartsWith("w9 w10", chunks[1]); // ...repeat at the start of chunk 2
        Assert.EndsWith("w25", chunks[^1]);
    }

    [Fact]
    public void EmptyText_ReturnsNoChunks()
    {
        Assert.Empty(TextChunker.Chunk("   "));
    }

    [Fact]
    public void OverlapNotSmallerThanChunkSize_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TextChunker.Chunk("a b c", chunkSize: 10, overlap: 10));
    }
}
