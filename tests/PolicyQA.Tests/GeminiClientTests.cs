using System.Text.Json;
using Xunit;

public class GeminiClientTests
{
    [Fact]
    public void ParseResponse_ReadsModelOutputAndTokens_IgnoringThoughts()
    {
        const string json = """
            {
              "steps": [
                { "type": "thought", "content": [ { "type": "text", "text": "thinking..." } ] },
                { "type": "model_output", "content": [
                    { "type": "text", "text": "Hello" },
                    { "type": "text", "text": " world" }
                ] }
              ],
              "usage": { "total_input_tokens": 12, "total_output_tokens": 3 }
            }
            """;

        using var doc = JsonDocument.Parse(json);
        var result = GeminiClient.ParseResponse(doc.RootElement);

        Assert.Equal("Hello world", result.Text);
        Assert.Equal(12, result.InputTokens);
        Assert.Equal(3, result.OutputTokens);
    }

    [Fact]
    public void ParseResponse_MissingFields_ReturnsEmptyResult()
    {
        using var doc = JsonDocument.Parse("{}");
        var result = GeminiClient.ParseResponse(doc.RootElement);

        Assert.Equal("", result.Text);
        Assert.Equal(0, result.InputTokens);
    }
}
