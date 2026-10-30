using Xunit;

public class VectorStoreTests
{
    private static DocumentChunk Chunk(string file, string hash, float[] vector) =>
        new(Guid.NewGuid(), file, hash, 1, "text", vector);

    [Fact]
    public void CosineSimilarity_MatchesHandCalculation()
    {
        // Same example as the README: [3,4] vs [4,3] = 24/25 = 0.96; [3,4] vs [4,-3] = 0
        Assert.Equal(0.96, InMemoryVectorStore.CosineSimilarity([3f, 4f], [4f, 3f]), precision: 2);
        Assert.Equal(0.0, InMemoryVectorStore.CosineSimilarity([3f, 4f], [4f, -3f]), precision: 2);
    }

    [Fact]
    public void Search_ReturnsMostSimilarChunkFirst()
    {
        var store = new InMemoryVectorStore();
        var close = Chunk("a.pdf", "h1", [1f, 0f]);
        var far = Chunk("a.pdf", "h1", [0f, 1f]);
        store.ReplaceDocument("a.pdf", [far, close]);

        var results = store.Search([0.9f, 0.1f], topK: 1);

        Assert.Single(results);
        Assert.Equal(close.Id, results[0].Chunk.Id);
    }

    [Fact]
    public void ReplaceDocument_RemovesPreviousVersion()
    {
        var store = new InMemoryVectorStore();
        store.ReplaceDocument("a.pdf", [Chunk("a.pdf", "v1", [1f, 0f]), Chunk("a.pdf", "v1", [0f, 1f])]);

        int removed = store.ReplaceDocument("a.pdf", [Chunk("a.pdf", "v2", [1f, 0f])]);

        Assert.Equal(2, removed);
        Assert.Equal(1, store.Count);
        Assert.False(store.HasDocument("v1"));
        Assert.True(store.HasDocument("v2"));
    }
}
