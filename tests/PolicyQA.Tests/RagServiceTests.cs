using System.Globalization;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

/// <summary>
/// The LLM and embeddings are faked through their interfaces:
/// tests are fast, free, deterministic and never call Gemini.
/// </summary>
public class RagServiceTests
{
    private class FakeEmbedder(float[] queryVector) : IEmbeddingClient
    {
        public Task<IReadOnlyList<float[]>> EmbedDocumentsAsync(IReadOnlyList<string> texts, CancellationToken ct = default) =>
            Task.FromResult<IReadOnlyList<float[]>>(texts.Select(_ => queryVector).ToList());

        public Task<float[]> EmbedQueryAsync(string text, CancellationToken ct = default) =>
            Task.FromResult(queryVector);
    }

    private class FakeLlm : ILlmClient
    {
        public int Calls { get; private set; }
        public string LastInput { get; private set; } = "";

        public Task<LlmResult> GenerateAsync(string systemInstruction, string input, CancellationToken ct = default)
        {
            Calls++;
            LastInput = input;
            return Task.FromResult(new LlmResult("Employees get 15 days [1].", 100, 10));
        }
    }

    private static (RagService Rag, FakeLlm Llm) Create(float[] queryVector, double minScore)
    {
        var config = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["Rag:TopK"] = "3",
                ["Rag:MinScore"] = minScore.ToString(CultureInfo.InvariantCulture),
                ["Pricing:InputPerMillionTokens"] = "1",
                ["Pricing:OutputPerMillionTokens"] = "2"
            })
            .Build();

        var store = new InMemoryVectorStore();
        store.ReplaceDocument("policy.pdf",
        [
            new DocumentChunk(Guid.NewGuid(), "policy.pdf", "hash", 1, "Employees get 15 vacation days.", [1f, 0f])
        ]);

        var llm = new FakeLlm();
        var rag = new RagService(new FakeEmbedder(queryVector), store, llm,
            new UsageTracker(config), config, NullLogger<RagService>.Instance);

        return (rag, llm);
    }

    [Fact]
    public async Task WeakMatch_BelowThreshold_SkipsLlm()
    {
        var (rag, llm) = Create(queryVector: [0f, 1f], minScore: 0.5); // similarity 0 with the chunk

        var result = await rag.AskAsync("What is the capital of France?");

        Assert.True(result.SkippedLlm);
        Assert.Equal(0, llm.Calls);
        Assert.Equal(RagService.NotFound, result.Answer);
        Assert.Equal(0, result.InputTokens);
    }

    [Fact]
    public async Task StrongMatch_CallsLlmWithNumberedSources()
    {
        var (rag, llm) = Create(queryVector: [1f, 0f], minScore: 0.5); // similarity 1

        var result = await rag.AskAsync("How many vacation days?");

        Assert.False(result.SkippedLlm);
        Assert.Equal(1, llm.Calls);
        Assert.Contains("[1] (policy.pdf, page 1)", llm.LastInput);
        Assert.Contains("Question: How many vacation days?", llm.LastInput);
    }

    [Fact]
    public async Task EstimatedCost_UsesConfiguredPrices()
    {
        var (rag, _) = Create(queryVector: [1f, 0f], minScore: 0.0);

        var result = await rag.AskAsync("How many vacation days?");

        // 100 input tokens at $1/M + 10 output tokens at $2/M
        Assert.Equal(100m / 1_000_000m + 20m / 1_000_000m, result.EstimatedCostUsd);
    }
}
